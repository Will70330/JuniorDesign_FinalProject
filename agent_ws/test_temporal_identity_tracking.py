import google.generativeai as genai
import cv2 as cv
import os
import json
import typing_extensions as typing

class Entity(typing.TypedDict):
    entity_name: str
    entity_type: str
    entity_location: list[int]

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
    
# Function to load images
def read_images(file_path, debug=False):
    files = os.listdir(file_path)
    images = []
    for file in files:
        if file.endswith(".png"):
            image = cv.imread(file_path + "/" + file, cv.IMREAD_COLOR)
            # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            images.append(image)

    if debug and images:
        cv.imshow("test", images[0])
        cv.waitKey(10000)

    return images


if __name__ == "__main__":
    IMAGE_PATH = "C:/Users/mucke/Pitt/Fall_2024/junior_design/JuniorDesign_FinalProject/examples"
    API_KEY_PATH = "C:/Users/mucke/Pitt/API_KEYS/gemini_api.txt"

    # Read in API Key and check for successful load
    gemini_api_key = read_api_key(API_KEY_PATH)
    if gemini_api_key:
        print(f"Successfully Loaded API Key: {gemini_api_key}")
    else:
        print("Failed to Load API Key.")
    
    # Read in Test Images
    imgs = read_images(IMAGE_PATH)

    # Set up Model API
    genai.configure(api_key=gemini_api_key)
    conf = genai.types.GenerationConfig(
        temperature=1.0,
        response_mime_type="application/json",
        response_schema=list[Entity]   
    )
    identifier_model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config=conf)

    # Start Interaction
    chat = identifier_model.start_chat(history=[])
    chat.send_message("""
        You are an identifier, your job is to identify people and objects within images.
        Provide answers that are concise and accurate, and do not hallucinate or make up any answers.
        
        For the entity name:
            if the entity is a person, fill with some random identifier ID unless provided with a specific name.
            if the entity is an object, fill with the name of the object (ie phone, pen, apple, etc).
                      Try to be specific as possible when identifying objects (iPhone vs. Phone, Red Shirt vs Shirt, Black Marker vs Pen)     
        For the entity location, use a list in [ymin, xmin, ymax, xmax] format.
        For the entity type, use 'person' for people, and 'item' for objects.
                      
        List all relevant entities (people and objects) within the frame that are relevant to the query.
    """)

    # Input Images 1 & 2 for Facial Identification
    _, test1 = cv.imencode(".png", imgs[0])
    _, test2 = cv.imencode(".png", imgs[1])
    test1 = test1.tobytes()
    test2 = test2.tobytes()
    response = chat.send_message([
        {'mime_type': 'image/png', 'data': test1},
        {'mime_type': 'image/png', 'data': test2},
        "The person in this image is William."
    ])
    print("_"*80)
    print("Prompt: The person in this image is William.")
    print(f"Model Response: \n {response.text}")

    # Input Image 3 for Object Detection
    _, test3 = cv.imencode(".png", imgs[2])
    test3 = test3.tobytes()
    response = chat.send_message([
        {'mime_type': 'image/png', 'data': test3},
        "Who is holding the phone?"
    ])
    print("_"*80)
    print("Prompt: Who is holding the phone?")
    print(f"Model Response: \n {response.text}")
    
    response = chat.send_message("What type of phone is he holding?")
    print("_"*80)
    print("Prompt: What type of phone is he holding")
    print(f"Model Response: \n {response.text}")
    
    # Test JSON Extraction from response
    try:
        data = json.loads(response.text)
        print(data)
    except Exception as e:
        print(f"JSON Extraction failed: {e}")
        quit(0)

    bbox = [coord / 1000 for coord in data[0]["entity_location"]]
    # Adjust BBOX Y coordinates back to original image coordinates
    bbox[0] = bbox[0] * imgs[2].shape[0]
    bbox[2] = bbox[2] * imgs[2].shape[0]
    # Adjust BBOX X coordinates back to original image coordinates
    bbox[1] = bbox[1] * imgs[2].shape[1]
    bbox[3] = bbox[3] * imgs[2].shape[1]
    bbox = [int(coord) for coord in bbox]

    framed_img = cv.rectangle(imgs[2], (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0,255,0), thickness=2)
    framed_img = cv.putText(framed_img, data[0]["entity_name"], (bbox[1], bbox[0]-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
    # print(framed_img.shape)
    # print(bbox)
    cv.imshow('Object Detection with Gemini', framed_img)
    cv.waitKey(10000)

