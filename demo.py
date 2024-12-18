import json
import os
import queue
import threading
import time
import urllib.request
import wave

import cv2 as cv
import google.generativeai as genai
import keyboard
import numpy as np
import pyaudio
import pyrealsense2 as rs
import serial
import typing_extensions as typing
from openai import OpenAI

######## GENERIC #########
# Function to read an API key from a file
def read_api_key(file_path):
    try:
        with open(file_path, 'r') as file:
            # read the first line
            api_key = file.readline().strip()
            return api_key
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Something went wrong: {e}")
        return None

class Entity(typing.TypedDict):
    entity_name: str
    entity_type: str
    entity_location: list[int]

def draw_bbox(input_img, bbox, img_width, img_height, entity_name):
    bbox = list(bbox)
    bbox[0] = bbox[0] * img_height
    bbox[1] = bbox[1] * img_width
    bbox[2] = bbox[2] * img_height
    bbox[3] = bbox[3] * img_width
    bbox = [int(coord) for coord in bbox]
    framed_img = cv.rectangle(input_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0,255,0), thickness=2)
    framed_img = cv.putText(framed_img, entity_name, (bbox[1], bbox[0]-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
######## AUDIO CLASSES #########
# AUDIO CONSTANTS
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

# Audio Recording Class
class AudioRecorder:
    def __init__(self):
        # print("\nInitializing Audio Recorder...")
        self.pAudio = pyaudio.PyAudio()
        self.index = 1
        # self.get_device_index() uncomment to select your microphone array index
        self.stream = self.pAudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.index
        )
        self.frames = []
        # print("Audio Recorder Initialized!")

    def get_device_index(self):
        # Get Device List for Recording
        print("----------------record device list----------------")
        info = self.pAudio.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")
        for i in range(0, numdevices):
            if (self.pAudio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print("Input Device id ", i, " - ", self.pAudio.get_device_info_by_host_api_device_index(0, i).get('name'))
        print("--------------------------------------------------")
        self.index = int(input())
        print(f"Recording via index: {self.index}\n")

    def record_audio(self):
        self.frames = []
        try:
            print("Press 'r' to start recording & release to stop.")
            while True:
                if keyboard.is_pressed('r'):
                    #  print("Recording Started...")
                    while keyboard.is_pressed('r'):
                        data = self.stream.read(CHUNK)
                        self.frames.append(data)
                    # print("Recording Stopped.\n")
                    break
        except KeyboardInterrupt:
            print("Recording Interrupted.")
        except Exception as e:
            print("Something went wrong when recording audio.")
            raise e
        
        self.stream.stop_stream()
        self.stream.close()
        self.pAudio.terminate()
        print("Audio Recording Complete.\n")

    def save_audio_recording(self, wave_output_file):
        if self.frames:
            wave_output_path = os.path.join(os.getcwd(), wave_output_file)
            waveFile = wave.open(wave_output_path, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(self.pAudio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(self.frames))
            waveFile.close()
            print(f"Audio saved to {wave_output_path}")
        else:
            print("No Audio was recorded.")

# Audio Player Class
class AudioPlayer:
    def __init__(self):
        # print("\nInitializing Audio Player...")
        self.audio_queue = queue.Queue()
        self.playback_complete = threading.Event()
        self.audio_added = threading.Event()

        # Initializing PyAudio
        self.pAudio = pyaudio.PyAudio()
        self.stream = self.pAudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True    
        )

        # Start Audio playback thread
        self.audio_thread = threading.Thread(target=self.play_audio)
        self.audio_thread.start()
        # print("Audio Player Initialized")

    def play_audio(self):
        while not self.playback_complete.is_set():
            # Give a way to stop audio
            if keyboard.is_pressed('q'):
                print("Audio playback ended by User.")
                break

            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.stream.write(audio_chunk)
            except queue.Empty:
                if self.audio_added.is_set() and self.audio_queue.empty():
                    break # All audio has been added and the queue is empty, so we're done
                continue

        self.stream.stop_stream()
        self.stream.close()
        self.pAudio.terminate()
        # print("Audio playback complete.") if debug else None

    def add_audio(self, audio_data):
        for chunk in audio_data:
            self.audio_queue.put(chunk)
        self.audio_added.set()  # Signal that audio has been added

    def wait_for_completion(self):
        self.audio_thread.join()

######### FACIAL TRACKING + REALSENSE ##########

# FaceTracking Class using YuNet
class FacialTracker:
    def __init__(self):
        try:
            print("Initializing Facial Tracker... Grabbing YuNet Model.")
            self.model_name = "face_detection_yunet_2023mar.onnx"
            self.model_path = os.path.join(os.getcwd(), self.model_name)
            self.model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            
            # download model if it doesn't already exist
            if not os.path.exists(self.model_path):
                print(f"Model not found at {self.model_path}")
                print("Downloading Model...")
                urllib.request.urlretrieve(self.model_url, self.model_path)
                print(f"Model Downloaded to {self.model_path}")
            else:
                print(f"Found model at {self.model_path}")
            
            # Verify file is readable
            if os.path.exists(self.model_path):
                print(f"Model File Size: {os.path.getsize(self.model_path)} bytes")
            else:
                print("Model file still not found after downloading.")

            # Create Detector
            self.face_YN_detector = cv.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),
                0.9,
            )
        except Exception as e:
            print(f"Could not load YN face detector: {e}")
        finally:
            print("YuNet Face Detector Model Created.")

        try:
            print("\nEstablishing Serial Connection to Arduino.")
            self.arduino_port = "COM9"
            self.baud_rate = 9600
            self.timeout = 5.0
            
            # Establish serial connection to Arduino
            self.arduino = serial.Serial(port=self.arduino_port, baudrate=self.baud_rate, timeout=self.timeout)
            time.sleep(self.timeout) # Wait for the arduino to Initialize
            print("Connected to Arduino!")
            self.current_angle = 90
            self.write_to_servo(self.current_angle)
        except serial.SerialException as  e:
            print(f"Error: Could not open serial port {self.arduino_port}: {e}")
            raise e
        finally:
            self.close_port()
    
    def detect_and_display(self, frame):
        self.face_YN_detector.setInputSize((frame.shape[1], frame.shape[0]))
        frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        # Detect Faces
        _, faces = self.face_YN_detector.detect(frame) # Using YuNet Model
        bbox = []
        
        # Only modify frames if faces are detected
        if not faces is None and faces.shape[0] >= 1:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                frame = cv.rectangle(frame, (x,y), (x+w, y+h), color=(0,0,255), thickness=2)
                bbox.append((x, y, w, h))

        cv.imshow("Facial Detection Test", frame)
        return bbox
    
    def calculate_servo_angle(self, bbox_center_x, image_center_x):
        error = bbox_center_x - image_center_x
        adjustment = 0.01 * error
        new_angle = self.current_angle - adjustment
        self.current_angle = max(0, min(180, new_angle))
        return self.current_angle
    
    def write_to_servo(self, value):
        self.arduino.write(f"{value}\n".encode())

    def close_port(self):
        if 'arduino' in locals() and self.arduino.is_open:
            self.arduino.close()
            print("Serial connection closed.")

# RealSense Class
class Sensor:
    def __init__(self):
        # Configure Streams from realsense
        print("\nInitializing Realsense & Establishing Pipeline...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device for setting supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        # Check to make sure that the camera is RGB
        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The Demo requires a Depth Camera with a Color sensor.")
            exit(0)
        print("Realsense Initialized!\n")

    def start_pipeline(self):
        self.pipeline.start(self.config)

    def get_image_from_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        else:
            return cv.cvtColor(np.asanyarray(color_frame.get_data()), cv.COLOR_BGR2RGB)
        
    def end_session(self):
        print("Stopping Realsense Pipeline & Destroying Windows...")
        cv.destroyAllWindows()
        self.pipeline.stop()

######## AGENTS ########
# Chat Voice Agent (OpenAI Text-to-Speech - Voice: 'Onyx')
class VoiceAgent:
    def __init__(self):
        print("\nInitializing Voice Agent with OpenAI Servers...")
        self._api_key_path = "C:/Users/mucke/Pitt/API_KEYS/openai_api.txt"
        self._api_key = read_api_key(self._api_key_path)
        self.client = OpenAI(api_key=self._api_key)

        # define model parameters
        self.model="tts-1" # use tts-1-hd for higher quality response
        self.voice="onyx"
        self.response_format = "pcm"

    def text_to_speech(self, text):
        audio_response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format=self.response_format,
        )
        return audio_response.iter_bytes(chunk_size=CHUNK)
        
# Generic Gemini Agent Class
class GeminiAgent:
    def __init__(self, model="models/gemini-1.5-flash", response_mime_type=None, response_schema=None):
        print("Initializing Prompt Agent with Google Gemini Servers")
        self._api_key_path = "C:/Users/mucke/Pitt/API_KEYS/gemini_api.txt"
        self._api_key = read_api_key(self._api_key_path)
        genai.configure(api_key=self._api_key)

        # Model Configurations
        self.model = model
        self.temperature = 1.0
        self.response_mime_type=response_mime_type
        self.response_schema=response_schema
        self.conf = genai.types.GenerationConfig(temperature=self.temperature)
        self.conf.response_mime_type = response_mime_type if self.response_mime_type else None
        self.conf.response_schema = response_schema if self.response_schema else None
        self.chat_model = genai.GenerativeModel(self.model, generation_config=self.conf)
        self.chat = self.chat_model.start_chat(history=[])

    def set_mime_type(self, mime_type):
        self.response_mime_type = mime_type
        self.adjust_conf()

    def set_response_schema(self, schema):
        self.response_schema=schema
        self.adjust_conf()

    def set_model(self, text):
        self.model = text

    def adjust_conf(self, preserve_history):
        if self.response_mime_type and self.response_schema:
            self.conf = genai.types.GenerationConfig(
                temperature=self.temperature,
                response_mime_type=self.response_mime_type,
                response_schema=self.response_schema
            )
        elif self.response_mime_type and not self.response_schema:
            self.conf = genai.types.GenerationConfig(
                temperature=self.temperature,
                response_mime_type=self.response_mime_type,
            )
        elif not self.response_mime_type and self.response_schema:
            self.conf = genai.types.GenerationConfig(
                temperature=self.temperature,
                response_schema=self.response_schema
            )
        else:
            self.conf = genai.types.GenerationConfig(temperature=self.temperature)

        self.reset_model(preserve_history)

    def reset_model(self, preserve_history):
        hist = self.chat.history if preserve_history else []
        self.chat_model = genai.GenerativeModel(self.model, generation_config=self.conf)
        self.chat = self.chat_model.start_chat(history=hist)

    def prompt_model(self, prompt):
        return self.chat.send_message(prompt)

    def get_hist(self):
        return self.chat.history


if __name__ == "__main__":
    face_tracker = FacialTracker()
    sensor = Sensor()
    voice_agent = VoiceAgent()
    prompt_agent = GeminiAgent(model="models/gemini-1.5-flash")
    classifier_agent = GeminiAgent(model="models/gemini-1.5-pro", response_mime_type="application/json", response_schema=list[Entity])
    conversation_agent = GeminiAgent(model="models/gemini-1.5-pro")
    wave_output_file = "prompt.wav"
    keypress = ord('0')
    
    try:
        print("Press 'q' in Window to exit...")
        sensor.start_pipeline()

        while not (keypress & 0xFF) == ord('q'):
            # Generate Audio Prompt
            try:
                recorder = AudioRecorder()
                recorder.record_audio()
                recorder.save_audio_recording(wave_output_file)
            except Exception as e:
                print(f"Something went wrong with audio recording: {e}")

            # Generate Text Prompt from Audio
            prompt = prompt_agent.prompt_model([
                "Your job is to accurately transcribe audio files. Do not make anything up, and provide only complete words spoken as text.",
                {'mime_type': 'audio/wav', 'data': open(os.path.join(os.getcwd(), wave_output_file), 'rb').read()}
            ])
            print(f"Transcribed Text Prompt: {prompt.text}")

            # Grab Frames from Scene 
            cv.namedWindow("Facial Detection Test", cv.WINDOW_AUTOSIZE)
            input_images = []
            for i in range(0,100):
                color_image = sensor.get_image_from_frames()
                bboxes = face_tracker.detect_and_display(color_image)
                keypress = cv.waitKey(1)

                # Calculate BBOX Center & Image Center
                if bboxes:
                    bbox_center_x = bboxes[0][0] + (bboxes[0][2] // 2)
                    image_center_x = color_image.shape[1] / 2
                    new_angle = face_tracker.calculate_servo_angle(bbox_center_x, image_center_x)

                    # Check that we have a valid angle
                    if 0 <= new_angle <= 180:
                        face_tracker.write_to_servo(new_angle)

                    # Draw BBOX
                    # input_image = draw_bbox(color_image, bbox=bboxes[0], img_width=color_image.shape[1], img_height=color_image.shape[0], entity_name="person")
                    _, input_image = cv.imencode(".png", color_image)
                    input_images.append(input_image.tobytes())
            
            # Pass frames into classifer agent
            classifier_agent.prompt_model(["""
                You are an identifier/classifier, your job is to identify people and objects within images with as much specificity as possible.
                Provide answers that are descriptive yet concise and accurate, and do not hallucinate or make up any answers.
                
                For the entity name:
                    if the entity is a person, fill with some random identifier ID unless provided with a specific name.
                    if the entity is an object, fill with the name of the object (ie phone, pen, apple, etc).
                            Try to be specific as possible when identifying objects (iPhone vs. Phone, Red Shirt vs Shirt, Black Marker vs Pen)     
                For the entity location, use a list in [ymin, xmin, ymax, xmax] format.
                For the entity type, use 'person' for people, and 'item' for objects.
                            
                List all relevant entities (people and objects) within the frame that are relevant to the query.
            """])
            classifier_response = classifier_agent.prompt_model([
                {'mime_type': 'image/png', 'data': input_images[-1]},
                {'mime_type': 'image/png', 'data': input_images[-2]},
                prompt.text
            ])
            print(f"Model Response:\n{classifier_response.text}")

            # Pass History into Conversational Agent
            conversation_agent.prompt_model(["""
                You are an expert in conversations. You try to be as helpful and kind as possible to everyone you interact with. Given a prompt and
                some additional context of the scene in the form of JSON formatted text, please try to faciliate a conversation with the user that
                will best answer their prompt in a concise yet enjoyable manner.

                Your response should only be text answering the prompt with the information you have.                                
            """])
            conversation_response = conversation_agent.prompt_model([
                prompt.text,
                classifier_response.text,
                {'mime_type': 'image/png', 'data': input_images[-1]},
                {'mime_type': 'image/png', 'data': input_images[-2]},
            ])
            print(f"Conversation Agent: \n{conversation_response.text}")

            # Convert Text to Speech
            audio_data = voice_agent.text_to_speech(conversation_response.text)
            player = AudioPlayer()
            player.add_audio(audio_data)
            player.wait_for_completion()
            
    except Exception as e:
        print(f"Error: Something went wrong: {e}")
    finally:
        print("Exiting Program...")
        sensor.end_session()
        face_tracker.close_port()


