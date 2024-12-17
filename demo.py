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
            self.arduino_port = "COM7"
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

if __name__ == "__main__":
    face_tracker = FacialTracker()
    sensor = Sensor()
    keypress = ord('0')
    
    try:
        print("Press 'q' in Window to exit...")
        sensor.start_pipeline()

        while not (keypress & 0xFF) == ord('q'):
            color_image = sensor.get_image_from_frames()
            cv.namedWindow("Facial Detection Test", cv.WINDOW_AUTOSIZE)
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
    except Exception as e:
        print(f"Error: Something went wrong: {e}")
    finally:
        print("Exiting Program...")
        sensor.end_session()
        face_tracker.close_port()


