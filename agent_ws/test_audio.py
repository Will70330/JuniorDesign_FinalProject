from openai import OpenAI
import google.generativeai as genai
import os
from io import BytesIO
import pyaudio
import queue
import threading

# Class for handling audio output
class AudioPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.playback_complete = threading.Event()
        self.audio_added = threading.Event()

        # Initializing PyAudio
        self.pAudio = pyaudio.PyAudio()
        self.stream = self.pAudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True    
        )

        # Start Audio playback thread
        self.audio_thread = threading.Thread(target=self.play_audio)
        self.audio_thread.start()

    def play_audio(self):
        while not self.playback_complete.is_set():
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

    # Generate Text Response
    prompt = "Write me a haiku about recursive programming."
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
