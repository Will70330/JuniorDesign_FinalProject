import cv2
import numpy as np
import os
import pyrealsense2 as rs
import urllib.request

def detectAndDisplay(frame, use_YN):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detect Faces
    if not use_YN:
        faces = face_cascade.detectMultiScale(frame_gray) # Using HaarCascades
    else:
        _, faces = face_YN_detector.detect(frame) # Using YuNet Model
    
    # Only modify frames if faces are detected
    if (not use_YN) or (not faces is None and faces.shape[0] >= 1):
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color=(0,0,255), thickness=2)

    cv2.imshow("Facial Detection Test", frame)

# Facial Classifiers
# -- Haar Cascades
face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier()

# -- YuNet
try:
    current_dir = os.getcwd()
    model_name = "face_detection_yunet_2023mar.onnx"
    model_path = os.path.join(current_dir, model_name)
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    # download model if it doesn't already exist
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Downloading Model...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model Downloaded to {model_path}")
    else:
        print(f"Found model at {model_path}")
    
    # Verify file is readable
    if os.path.exists(model_path):
        print(f"Model File Size: {os.path.getsize(model_path)} bytes")
    else:
        print("Model file still not found after downloading.")

    face_YN_detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),
        0.9,
    )

except Exception as e:
    print(f"Could not load YN face detector: {e}")
finally:
    print("YuNet Face Detector Model Created.")

# Load the Facial Detector
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Error Loading Face Cascade")

# Configure Streams from realsense
pipeline = rs.pipeline()
config = rs.config()

# Get device for setting supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Check to makesure that our camera is RGB
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The Demo requires Depth Camera with Color sensor")
    exit(0)

# Start streaming images
print("Press 'q' to exit...")
pipeline.start(config)
key = ord('0')

try:
    while not (key & 0xFF) == ord('q'):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        face_YN_detector.setInputSize((color_image.shape[1], color_image.shape[0]))

        # Run facial detection
        cv2.namedWindow("Facial Detection Test", cv2.WINDOW_AUTOSIZE)
        detectAndDisplay(color_image, True)
        key = cv2.waitKey(1)
    
finally:
    print("Exiting Program...")
    cv2.destroyAllWindows()
    pipeline.stop()