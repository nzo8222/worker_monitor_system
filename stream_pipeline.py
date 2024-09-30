import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import TextIOWrapper
from typing import List

import cv2
import numpy as np

from stream_classifier import return_prediction  # Ensure this imports correctly

# Setup logging to write to a file
logging.basicConfig(
    filename="trigger_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Load configuration from config.json
with open("config.json", "r") as config_file:
    params = json.load(config_file)


class RTSPCapture:
    def __init__(self, rtsp_urls: List[str], max_size_mb: int = params["max_size_mb"]):
        self.rtsp_urls = rtsp_urls
        self.max_size_mb = max_size_mb
        self.caps = []
        for url in rtsp_urls:
            try:
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open RTSP stream: {url}")
                self.caps.append(cap)
            except Exception as e:
                print(f"Error with URL {url}: {e}")

    def __del__(self):
        for cap in self.caps:
            cap.release()

    def capture_frame(self, cap, retries=3, delay=1) -> cv2.VideoCapture:
        for attempt in range(retries):
            ret, frame = cap.read()
            if ret:
                # logging.info("Frame captured successfully")
                return frame
            else:
                logging.warning(f"Attempt {attempt + 1}: Failed to capture frame. Retrying in {delay} seconds...")
                time.sleep(delay)

        raise RuntimeError(f"Failed to capture image from RTSP stream url {self.rtsp_urls} after retries")

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        new_width = width // 2
        new_height = height // 2
        return cv2.resize(frame, (new_width, new_height))

    def encode_frame_to_base64(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        return jpg_as_text

    def is_within_size_limit(self, base64_str: str) -> bool:
        size_mb = (len(base64_str) * 3 / 4) / (1024 * 1024)
        return size_mb <= self.max_size_mb

    def get_encoded_frame(self, cap: cv2.VideoCapture) -> str:
        frame = self.capture_frame(cap)
        resized_frame = self.resize_frame(frame)
        encoded_frame = self.encode_frame_to_base64(resized_frame)
        if not self.is_within_size_limit(encoded_frame):
            raise ValueError("Encoded frame exceeds size limit")
        return encoded_frame


def process_frame(
    rtsp_capture: RTSPCapture,
    cap: cv2.VideoCapture,
    frame_count: int,
    results: list,
    write_lock,
    output_file_path: str = "rtsp_captures_result.jsonl",
):
    try:
        # Get the encoded frame
        # print(f"Thread {threading.current_thread().name} processing frame {frame_count}")

        encoded_frame = rtsp_capture.get_encoded_frame(cap)
        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        custom_id = f"frame_dt-{unique_id}"
        # print(f"Processing frame {custom_id}")

        # Send the frame for prediction
        try:
            prediction = return_prediction(encoded_frame, custom_id)
            image_id = prediction["image_id"]
            image = prediction["encoded_image"]
            detections = prediction["detections"]

            # Print the values
            # print("Image ID:", image_id)
            # print("Detections:", detections)

        except Exception as e:
            print(f"Error during prediction for {custom_id}: {e}")

        results.append({"image_id": prediction.custom_id, "predictions": prediction})
        with write_lock:
            try:
                with open(output_file_path, "a") as f:
                    # Save the response to a JSONL file
                    f.write(json.dumps({"image_id": image_id, "encoded_image": image, "detections": detections}) + "\n")
                    f.flush()
                # print(f"Successfully saved frame {custom_id} to {output_file_path}")
            except Exception as e:
                print(f"Error writing to file for {custom_id}: {e}")

        # Optionally display the prediction
        # print(f"Prediction for {prediction.custom_id}: { prediction.detection}")
    except RuntimeError as e:
        print(f"Error processing frame: {e}")





def run_realtime_capture(
    n: int = 20,
    video_capture_interval: int = 100,
    output_file_path="rtsp_captures_result.jsonl",
):
    # Save the response to a JSONL file
    write_lock = threading.Lock()

    rtsp_urls = params["rtsp_urls"]
    rtsp_capture = RTSPCapture(rtsp_urls)

    frame_count = 0  # Keep track of frame numbers
    results = []
    # Create a lock for synchronizing access to the file
    # Use ThreadPoolExecutor to handle the frame processing in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers as needed
        # print(f"no of frame captured={frame_count}")
        try:
            while True:
                # Capture a frame from each stream
                start_time = time.time()
                for idx, cap in enumerate(rtsp_capture.caps):
                    stream_frame_count = frame_count + idx
                    # Submit the frame processing to the thread pool
                    executor.submit(
                        process_frame,
                        rtsp_capture,
                        cap,
                        stream_frame_count,
                        results,
                        write_lock,
                    )
                    # Increment the frame counter
                    frame_count += 1
                # Check if n frames have been processed
                if frame_count % n == 0:
                    logging.info(f"frame count {frame_count}")
                if frame_count % video_capture_interval == 0:
                    logging.info(f"Capturing short video after {frame_count} frames")
                    # capture_short_video(rtsp_capture)
                elapsed_time = time.time() - start_time
                if elapsed_time < 1.0:
                    time.sleep(1.0 - elapsed_time)  # Wait for 1 second before capturing the next frame
        except KeyboardInterrupt:
            print("Stopping real-time capture...")
        finally:
            # Cleanup OpenCV resources
            rtsp_capture.__del__()
            cv2.destroyAllWindows()
