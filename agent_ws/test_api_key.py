import google.generativeai as genai

# NOTE: Need to use gRPC version 1.67.1 due to an ongoing issue with gRPC 1.68.0 resulting in unsaved model responses & warning/error message
# to fix, make sure you have gRPC version 1.67.1 installed and downgrade grpcio-status to version 1.67.0 to avoid dependency issues

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

    FILE_PATH = "C:/Users/mucke/Pitt/API_KEYS/gemini_api.txt"
    gemini_api_key = read_api_key(FILE_PATH)

    # Check that API key was loaded successfully
    if gemini_api_key:
        print(f"Successfully Loaded API Key: {gemini_api_key}")
    else:
        print("Failed to Load API Key.")

    # Test API key
    genai.configure(api_key=gemini_api_key)
    generation_conf = genai.types.GenerationConfig(
        candidate_count=1,
        temperature=1.0
    )
    model = genai.GenerativeModel("models/gemini-1.5-flash", safety_settings=None, generation_config=generation_conf)
    chat = model.start_chat(history=[])
    response = chat.send_message("Hello how are you today?")
    response = chat.send_message("What's your name?")
    print(chat.history)
