import json
import os
import threading
from datetime import datetime

import cv2
import requests
from requests.auth import HTTPDigestAuth

EXTRACTED_FRAMES_DIR = "extracted_frames"


def clean_old_images():
    today_str = datetime.now().strftime("%Y%m%d")
    for filename in os.listdir(EXTRACTED_FRAMES_DIR):
        if filename.endswith(".jpg"):
            file_date_str = filename.split("_")[0]
            if file_date_str != today_str:
                file_path = os.path.join(EXTRACTED_FRAMES_DIR, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted old image: {filename}")
                except Exception as e:
                    print(f"Failed to delete {filename}: {e}")


def clean_old_entries_in_jsonl(jsonl_file_path="rtsp_captures_result.jsonl"):
    today_str = datetime.now().strftime("%Y%m%d")
    temp_file_path = jsonl_file_path + ".tmp"

    try:
        with open(jsonl_file_path, "r") as infile, open(temp_file_path, "w") as outfile:
            for line in infile:
                line = line.strip()  # Strip any leading/trailing whitespace or newline characters
                if not line:
                    continue  # Skip empty lines

                try:
                    data = json.loads(line)
                    image_id = data.get("image_id", "")

                    # Extract the date from the image_id
                    if image_id and image_id.startswith("frame_dt-"):
                        image_date_str = image_id.split("_")[0].split("-")[1]

                        # Write the line to the temp file if it's today's date
                        if image_date_str == today_str:
                            outfile.write(line + "\n")  # Ensure the line is written with a newline character

                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line}")
                except Exception as e:
                    print(f"Error processing line in JSONL: {e}")

        # Replace the original file with the temp file
        os.replace(temp_file_path, jsonl_file_path)

    except FileNotFoundError:
        print(f"File {jsonl_file_path} not found.")
    except Exception as e:
        print(f"Error processing the JSONL file: {e}")


def detect_motion(camera_ips):
    # Load username and password from config.json
    with open("config.json", "r") as f:
        params = json.load(f)
        username = params["username"]
        password = params["password"]

    # Create a list to store whether motion was detected in each stream
    motion_detected = [False] * len(camera_ips)

    # Event to stop the threads once motion is detected
    motion_event = threading.Event()

    # Function to handle ISAPI motion detection for each camera
    def handle_motion_detection(index, ip):
        try:
            events_url = f"http://{ip}/ISAPI/Event/notification/alertStream"
            response = requests.get(events_url, auth=HTTPDigestAuth(username, password), stream=True)
            print(f"Listening for ISAPI motion detection events from camera {ip}...")

            text_data = ""

            # Process the incoming event stream
            for line in response.iter_lines():
                if line and not motion_event.is_set():
                    decoded_line = line.decode("utf-8").strip()

                    # Ignore boundary and content type lines
                    if decoded_line.startswith("--boundary") or decoded_line.startswith("Content-Type"):
                        continue

                    text_data += decoded_line

                    # Check if the closing XML tag is present
                    if "</EventNotificationAlert>" in decoded_line:
                        # Check for the motion alarm event
                        if "Motion alarm" in text_data:
                            motion_detected[index] = True  # Set motion detected flag
                            print(f"Motion detected via ISAPI from camera {ip}")
                            motion_event.set()  # Signal that motion has been detected
                        text_data = ""  # Reset for the next event

        except Exception as e:
            print(f"Error with ISAPI stream from camera {ip}: {e}")

    # Function to display the video stream and motion detection text
    def monitor_stream(index, ip):
        stream_url = f"rtsp://{username}:{password}@{ip}:554"

        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            print(f"Failed to open stream: {stream_url}")
            return

        while not motion_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from: {stream_url}")
                break

            # Display motion detection status on the frame
            if motion_detected[index]:
                cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No Motion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(f"Camera {index + 1}", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                motion_event.set()  # Allow early exit with 'q'
                break

        cap.release()
        cv2.destroyAllWindows()

    # Create and start threads for each stream
    for index, ip in enumerate(camera_ips):
        isapi_thread = threading.Thread(target=handle_motion_detection, args=(index, ip))
        stream_thread = threading.Thread(target=monitor_stream, args=(index, ip))

        isapi_thread.start()
        stream_thread.start()

        # Wait for both threads to finish
        isapi_thread.join()
        stream_thread.join()

    # Return True if motion was detected in any stream
    return any(motion_detected)
