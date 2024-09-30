# import threading
from collections import deque

import cv2
import imutils
import numpy as np

from stream_classifier import encode_frame_to_base64, return_prediction

# Video source
video_paths = [
    "rtsp://admin:Mayocabo1@192.168.1.8:554"#"dataset/output_cam2_20240823-110434.avi"
]  # rtsp://admin:Mayocabo1@192.168.1.8:554
cap = cv2.VideoCapture(video_paths[-1])

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width=640)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

# Add rectangle around the object
start_frame = cv2.rectangle(start_frame, (0, 10), (400, 40), (0, 0, 255), -1)

motion_mode = True
motion_counter = deque(maxlen=20)

count = 0
openai_response = False
detection_string = None
display_counter = 0


def fetch_prediction(image):
    global detection_string, openai_response, display_counter
    detection_string = return_prediction(image, "test")
    openai_response = True
    display_counter = 200  # Show detection for the next 50 frames
    print(detection_string)


first_time = True

# Desired FPS
fps = 30
frame_delay = int(1000 / fps)  # Delay per frame in milliseconds

while True:
    ret, frame = cap.read()
    if not ret:
        break
    original_frame = frame.copy()
    frame = imutils.resize(frame, width=640)
    frame_copy = frame.copy()
    cv2.rectangle(frame, (0, 10), (400, 40), (0, 0, 255), -1)

    if motion_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (7, 7), 0)

        difference = cv2.absdiff(frame_bw, start_frame)
        threshold = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY)[1]

        start_frame = frame_bw
        threshold_sum = threshold.sum()

        if threshold_sum > 2000:
        #     motion_counter.extend([True, True, True, True, True])
        # elif threshold_sum > 500:
            motion_counter.append(True)
        else:
            motion_counter.append(False)

        if sum(motion_counter) >= 20:
            cv2.putText(
                frame_copy,
                "Motion Detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                (0, 0, 255),
                2,
            )

            if (count == 0 or openai_response) and display_counter <= 0:
                if not openai_response:
                    image = encode_frame_to_base64(original_frame)
                    print("Calling OpenAI for prediction...")
                    fetch_prediction(image)
                    input("Press Enter to continue...")
                    openai_response = (
                        True  # Prevent additional calls until this one completes
                    )

            count += 1

        if openai_response and display_counter > 0:
            if detection_string is not None:
                words = str(detection_string["detections"]).split()
                lines = [" ".join(words[i : i + 4]) for i in range(0, len(words), 4)]
                text_to_display = "\n".join(lines)
                cv2.rectangle(frame_copy, (400, 10), (700, 80), (255, 255, 255), -1)
                y0, dy = 20, 10  # Start position and line height
                for i, line in enumerate(text_to_display.split("\n")):
                    y = y0 + i * dy
                    cv2.putText(
                        frame_copy,
                        line,
                        (400, y),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.35,  # font scale
                        (0, 92, 255),  # color
                        1,  # thickness
                    )
            display_counter -= 1

            if display_counter <= 0:
                # Reset conditions to check motion again after displaying results
                openai_response = False
                count = 0  # Reset count to trigger a new prediction after 50 frames

        # cv2.imshow("Cam", threshold)
        # cv2.imshow("Cam2", frame_copy)

        threshold_colored = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
        stacked_frame = np.hstack((threshold_colored, frame_copy))

        # Display the stacked frame in one window
        cv2.imshow("Output Cam", stacked_frame)
        frame_width = 640 * 2  # Since we're stacking two 640px frames horizontally
        frame_height = 480
        output_file = "output_stacked_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        out.write(stacked_frame)

    else:
        cv2.imshow("Output Cam", frame_copy)

    key_pressed = cv2.waitKey(frame_delay)
    if key_pressed == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
