import cv2
import numpy as np
import os
import pyrealsense2 as rs
import urllib.request
import serial
import time

# Configure the serial port
arduino_port = "COM7"
baud_rate = 9600
timeout = 5.0

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detect Faces
    _, faces = face_YN_detector.detect(frame) # Using YuNet Model
    bbox = []
    
    # Only modify frames if faces are detected
    if not faces is None and faces.shape[0] >= 1:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color=(0,0,255), thickness=2)
            bbox.append((x, y, w, h))

    cv2.imshow("Facial Detection Test", frame)
    return bbox

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
    # Establish serial connection to Arduino
    arduino = serial.Serial(port=arduino_port, baudrate=baud_rate, timeout=timeout)
    time.sleep(2) # Wait for arduino to initialize
    print("Connected to Arduino. Enter servo positions (0-180). 'q' to quit.")
    current_angle = 0
    arduino.write(f"{current_angle}\n".encode())

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
        bboxes = detectAndDisplay(color_image)
        key = cv2.waitKey(1)

        if bboxes:
            bbox_center_x = bboxes[0][0] + (bboxes[0][2] // 2)
            image_center_x = color_image.shape[1] / 2
            error = bbox_center_x - image_center_x
            adjustment = 0.01 * error
            new_angle = current_angle - adjustment
            current_angle = max(0, min(180, int(new_angle)))
            print(f"bbox: {bbox_center_x}, image: {image_center_x}, error: {error}, adjustment: {adjustment}, new_angle: {new_angle}")
            if 0 <= current_angle <= 180:
                arduino.write(f"{current_angle}\n".encode())
                print(f"Sent Position: {current_angle}")
            else:
                print("Error: Invalid Position provided (range is 0 to 180).")
            # time.sleep(10)

except serial.SerialException as e:
    print(f"Error: Could not open serial port {arduino_port}: {e}")
finally:
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("Serial connection closed.")

    print("Exiting Program...")
    cv2.destroyAllWindows()
    pipeline.stop()

