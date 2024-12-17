from openai import OpenAI
import google.generativeai as genai
import os
from io import BytesIO
import pyaudio
import queue
import threading
import wave
import keyboard

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024


# Class for handling audio input
class AudioRecorder:
    def __init__(self):
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
                    print("Recording Started...")
                    while keyboard.is_pressed('r'):
                        data = self.stream.read(CHUNK)
                        self.frames.append(data)
                    print("Recording Stopped.\n")
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


# Class for handling audio output
class AudioPlayer:
    def __init__(self):
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
        print("Audio playback complete.")

    def add_audio(self, audio_data):
        for chunk in audio_data:
            self.audio_queue.put(chunk)
        self.audio_added.set()  # Signal that audio has been added

    def wait_for_completion(self):
        self.audio_thread.join()

# Use OpenAI API for text to speech
def text_to_speech(text):
    audio_response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text,
        response_format="pcm"
    )
    return audio_response.iter_bytes(chunk_size=1024)


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
    
if __name__ == "__main__":
    OPENAI_KEY_PATH = "C:/Users/mucke/Pitt/API_KEYS/openai_api.txt"
    GEMINI_KEY_PATH = "C:/Users/mucke/Pitt/API_KEYS/gemini_api.txt"
    WAVE_OUTPUT_FILE = "prompt.wav"

    # Set Up OpenAI
    openai_key = read_api_key(OPENAI_KEY_PATH)
    client = OpenAI(api_key=openai_key)

    
    # Set Up Gemini
    gemini_key = read_api_key(GEMINI_KEY_PATH)
    genai.configure(api_key=gemini_key)
    conf = genai.types.GenerationConfig(
        temperature=1.0,   
    )
    chat_model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config=conf)
    chat = chat_model.start_chat(history=[])

    # Generate Audio Prompt
    try:
        recorder = AudioRecorder()
        recorder.record_audio()
        recorder.save_audio_recording(WAVE_OUTPUT_FILE)
    except Exception as e:
        print(f"Something went wrong with audio recording: {e}")

    # Generate Text Prompt from Audio
    prompt = chat.send_message([
        "Please transcribe the audio file exactly, do not make anything up. Provide only the words said as text.",
        {'mime_type': 'audio/wav', 'data': open(os.path.join(os.getcwd(), WAVE_OUTPUT_FILE), 'rb').read()}
    ])
    print(f"Transcribed Text Prompt: {prompt.text}")

    # Generate Text Response
    prompt = prompt.text
    print("Generating Gemini Response...")
    response = chat.send_message(prompt)
    print("Response:")
    print(response.text)

    # Convert Text Response to Speech
    print("Converting Response to Speech with OpenAI model...")
    audio_data = text_to_speech(response.text)
    print("Playing Audio...")
    player = AudioPlayer()
    player.add_audio(audio_data)

    # Wait for audio to finish
    player.wait_for_completion()
    print("Audio Finished.")
