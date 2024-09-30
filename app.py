import uuid
import asyncio
from collections import deque
from io import BytesIO
import cv2
import imutils
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from stream_classifier import encode_frame_to_base64, return_prediction

# Video source

cap = cv2.VideoCapture(video_paths[-1])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read initial frame and setup
_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width=640)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)
start_frame = cv2.rectangle(start_frame, (0, 10), (400, 40), (0, 0, 255), -1)

motion_mode = True
motion_counter = deque(maxlen=20)

# State variables
count = 0
openai_response = False
detection_string = None
display_counter = 0
extracted_frame = None

# Initialize the dataframe
df = pd.DataFrame()

# Streamlit Layout
st.title("Live Stream Monitoring")
st.markdown("---")
st.header("Live Stream")
livestream_placeholder = st.empty()
processing_status_placeholder = st.empty()

st.markdown("---")
left_column, right_column = st.columns(2)

with left_column:
    st.header("Extracted Frame")
    frame_placeholder = st.empty()

with right_column:
    st.header("Results")
    result_placeholder = st.empty()

st.markdown("---")
st.header("Excel Data")
df_placeholder = st.empty()

# Function to fetch prediction using threading
def fetch_prediction(image, original_frame):
    try:
        custom_id = str(uuid.uuid4())  # Generate a unique ID for each frame
        detection_response = return_prediction(image, custom_id)
        detection_string = detection_response["detections"]

        # Save the extracted frame image to disk
        image_path = f"extracted_frames/{custom_id}.jpg"
        cv2.imwrite(image_path, original_frame)

        # The detections data is a list of dictionaries
        detections_dicts = detection_response["raw_detections"]

        return detection_string, image_path, detections_dicts
    except Exception as e:
        print(f"Error in fetch_prediction: {e}")
        return None, None, None

def process_video_frame(frame, executor):
    global start_frame, count, openai_response, display_counter, extracted_frame, df

    original_frame = frame.copy()
    frame = imutils.resize(frame, width=640)
    frame_copy = frame.copy()

    if motion_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

        difference = cv2.absdiff(frame_bw, start_frame)
        threshold = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY)[1]

        start_frame = frame_bw

        if threshold.sum() > 2000:
            motion_counter.append(True)
        else:
            motion_counter.append(False)

        if sum(motion_counter) >= 20:
            cv2.putText(
                frame_copy,
                "Motion Detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Font scale
                (0, 0, 255),  # Red color
                2,
            )

            if not openai_response:
                processing_status_placeholder.write("Processing Video... Please Wait")
                livestream_placeholder.empty()  # Clear the live stream display

                image = encode_frame_to_base64(original_frame)

                # Offload the prediction to a separate thread
                future = executor.submit(fetch_prediction, image, original_frame)
                detection_string, image_path, detections_dicts = future.result()

                if detection_string and image_path:
                    # Load the saved frame to display it
                    extracted_frame = cv2.imread(image_path)

                    # Update the Streamlit placeholders with the results
                    result_placeholder.markdown(f"### Detection Results\n\n{detection_string}")
                    frame_placeholder.image(
                        extracted_frame, caption="Extracted Frame", use_column_width=True, channels="BGR"
                    )

                    # Update DataFrame
                    for worker_counter, detection in enumerate(detections_dicts):
                        data_to_append = {"Image Path": image_path, "Worker ID": worker_counter + 1}
                        data_to_append.update(detection)
                        df = pd.concat([df, pd.DataFrame([data_to_append])], ignore_index=True)

                    df_placeholder.dataframe(df, width=700, height=300)

                    # Reset the state to allow new API calls
                    openai_response = True  # Set to True after call
                    display_counter = 200  # Reset display counter
                    count = 0  # Reset count

                else:
                    openai_response = False  # Ensure we can retry if API call failed

        if openai_response and display_counter > 0:
            display_counter -= 1
            if display_counter <= 0:
                openai_response = False  # Allow new API calls after display_counter ends
                count = 0  # Reset count

        # Update Streamlit UI
        livestream_placeholder.image(frame_copy, channels="BGR")

async def main():
    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            process_video_frame(frame, executor)
            await asyncio.sleep(0.01)  # Yield control to the event loop

        cap.release()

# Run the asyncio event loop
asyncio.run(main())

# Option to save dataframe as xlsx
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Sheet1")
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# Option to save dataframe as csv
def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Export buttons
st.markdown("---")
st.header("Export Data")

st.download_button(
    label="Download data as CSV",
    data=to_csv(df),
    file_name="output_data.csv",
    mime="text/csv",
)
