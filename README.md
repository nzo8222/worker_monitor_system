# worker-monitoring-system

# RTSP Stream Object Detection Pipeline

This project consists of a pipeline for capturing frames from RTSP streams, encoding them, and sending them to an object detection model. The pipeline includes three main components:

1. **RTSP Stream Capture** (`stream_pipeline.py`)
2. **Image Classification** (`stream_classifier.py`)
3. **Response Parsing and Data Modeling** (`data_model.py`)

## Components

### 1. RTSP Stream Capture (`stream_pipeline.py`)

This script captures frames from RTSP streams, resizes them, encodes them to base64, and saves them to a `.jsonl` file.

#### Key Classes and Functions:
- `RTSPCapture`: Handles capturing, resizing, encoding, and saving frames.
- `capture_frame`: Captures a frame from the RTSP stream.
- `resize_frame`: Resizes the captured frame.
- `encode_frame_to_base64`: Encodes the frame to a base64 string.
- `get_batch_encoded_frames`: Captures and encodes a batch of frames.

### 2. Image Classification (`stream_classifier.py`)

This script reads the encoded frames from the `.jsonl` file, sends them to an object detection model via an API, and saves the responses to another `.jsonl` file.

#### Key Functions:
- `encode_image`: Encodes an image to base64.
- `requests.post`: Sends the encoded image to the object detection model.
- `responses.append`: Saves the response with the associated image ID.

### 3. Response Parsing and Data Modeling (`data_model.py`)

This script parses the responses from the object detection model and structures the data into Python objects.

#### Key Classes and Functions:
- `PersonDetection`: Represents the detection of a person with various attributes.
- `ImageResponse`: Represents the response for an image with detections.
- `parse_response`: Parses the response data into `PersonDetection` objects.

## Usage

### Prerequisites

- Python 3.x
- Required Python packages: `ast`, `json`, `requests`, `dotenv`, `IPython`, `cv2`, `numpy`
- OpenAI API key (stored in a `.env` file)

### Steps

1. **RTSP Stream Capture**:
   - Configure the RTSP URLs and other parameters in `config.json`.
   - Run `stream_pipeline.py` to capture and encode frames from the RTSP streams.

   ```bash
   python stream_pipeline.py
   ```

2. **Image Classification**:
   - Ensure the OpenAI API key is set in the `.env` file.
   - Run `stream_classifier.py` to classify the encoded frames.

   ```bash
   python stream_classifier.py
   ```

3. **Response Parsing and Data Modeling**:
   - Run `data_model.py` to parse the responses and structure the data.

   ```bash
   python data_model.py
   ```

## Configuration

### `config.json`
