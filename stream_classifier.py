import base64
import json
import os
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

from data_model import ImageResponse, parse_response
from trigger import trigger

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_frame_to_base64(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


def save_frame_as_jpeg(image_path, unique_filename, quality=85):
    # Open the image
    image = Image.open(image_path)

    # Define the path to save the image with a unique filename
    output_path = f"extracted_frames/{unique_filename}.jpg"

    # Save the compressed image as a JPEG file
    image.save(output_path, format="JPEG", quality=quality)

    return output_path


def get_readable_output(detections):
    output = ""
    required_items = [
        "White Rubber Boots",
        "Brown Pants",
        "Knee Protector",
        "Brown Long Sleeves Shirt",
        "Glasses",
        "Gloves",
        "Gas Mask",
    ]
    visitor_item = "croc"

    for idx, person_detection in enumerate(detections):
        wearing_items = []
        not_wearing_items = []
        detection_failed_items = []

        # Check if the person is a visitor (i.e., not wearing all required items)
        is_visitor = all(person_detection.get(item, -1) == 0 for item in required_items)

        if is_visitor:
            person_label = "Visitor"
            # Check only for "croc" if the person is a visitor
            if person_detection.get(visitor_item, 0) == 1:
                wearing_items.append(visitor_item)
            elif person_detection.get(visitor_item, 0) == 0:
                not_wearing_items.append(visitor_item)
            else:
                detection_failed_items.append(visitor_item)
        else:
            person_label = "Worker"
            # Categorize the items based on the detection values
            for clothing_item, detection_value in person_detection.items():
                if clothing_item == visitor_item:
                    continue  # Skip checking "croc" for workers
                if detection_value == 1:
                    wearing_items.append(clothing_item)
                elif detection_value == -1:
                    not_wearing_items.append(clothing_item)
                else:
                    detection_failed_items.append(clothing_item)

        # Create the final sentence for this person
        if wearing_items:
            output += (
                f"{person_label} {idx + 1} is wearing " + ", ".join(wearing_items) + "."
            )

        if not_wearing_items:
            if not wearing_items:
                # If no items are worn, don't print "is wearing", just print "is not wearing"
                output += (
                    f"{person_label} {idx + 1} is not wearing "
                    + ", ".join(not_wearing_items)
                    + "."
                )
            else:
                output += (
                    f" However, they are not wearing "
                    + ", ".join(not_wearing_items)
                    + "."
                )

        if detection_failed_items:
            output += (
                f" cannot find these items in frame: "
                + ", ".join(detection_failed_items)
                + "."
            )

        output += "\n"  # Add a new line for clarity between workers/visitors

    return output


def return_prediction(
    encoded_image: str,
    custom_id: str,
    output_file_path: str = "rtsp_captures_result.jsonl",
):
    # print(f"Encoded Image Length: {len(encoded_image)} for {custom_id}")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Prepare the payload with the encoded image and custom ID
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You are a biosecurity protocol monitoring expert at a vehicle washing workshop. You need to ensure following rules  you need to detect workers and visitors based on uniform/outfit wearing. Always start detecting people from left to right. 
                                If worker then classes will be (White Rubber Boots, Brown Pants, Knee Protector, Brown Long Sleeves Shirt, Glasses, Gloves, Gas Mask, ) 
                                For protective gear like knee protector, gloves and gas mask if you think that a worker is wearing but is hidden by any obstacle in picture then this particular class value will be in negative and if any class is not detected it will be zero probability. 
                                If outsider then class will only be Crocs. 
                                For each person you will only return output in form of python dictionary with each class as key and detected probability of that class,
                                
                                if class found: 1
                                if class not found: -1
                                if class cannot be seen in frame: 0

                                The probability will always be in int, return this as list of dictionaries if multiple people in frame. do not return any other text or information. Make sure that you accurate detect number of People and do not lose track of people. Each index of a list gives worker number

                                Important Note:
                                1. Always start detecting people from left to right.
                                2. If worker is fully visible then return appropriate class value.
                                3. If worker is not fully visible, then return -1 for classes that are not in the frame.
                                4. If Face is visible then return either 1 or -1 for Glasses and Gas Mask.
                                5. If Feet are visible then return either 1 or -1 for White Rubber Boots.
                                6. If Hands are visible then return either 1 or -1 for Gloves.
                                7. If Legs are visible then return either 1 or -1 for Brown Pants.
                                8. If Shirt is visible then return either 1 or -1 for Brown Long Sleeves Shirt.
                                9. If Knee is visible then return either 1 or -1 for Knee Protector.
                                
                                
                                If feets are not visible then return 0 for White Rubber Boots.
                                If hands are not visible then return 0 for Gloves.
                                If legs are not visible then return 0 for Brown Pants.
                                If shirt is not visible then return 0 for Brown Long Sleeves Shirt.
                                If knee is not visible then return 0 for Knee Protector.
                                If face is not visible then return 0 for Glasses and Gas Mask.
                    

                                Response Format:
                                [
                                "Class_Name": 0 (if item is not visible in frame)
                                "Class_Name": 1 (if worker is wearing item)
                                "Class_Name": -1 (if working is not wearing item)
                                ]

                                """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    # Send the request to the OpenAI API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    response_data = response.json()
    print("API Response:", response_data)

    # Parse the response to extract detections
    detections = parse_response(response_data)

    image_data = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(image_data))

    # Generate a unique filename based on the current datetime
    unique_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image_path = f"extracted_frames/{unique_filename}.jpg"

    # Save the image as JPEG in the extracted_frames folder
    image.save(output_image_path, format="JPEG")

    # Create an ImageResponse object with the image ID and detections
    image_response = ImageResponse(custom_id, detections)
    trigger(
        image_response.image_id,
        [detection.__dict__ for detection in image_response.detections],
    )

    # Print the prediction response
    print(f"Image ID: {custom_id}")
    print(
        get_readable_output(
            [detection.__dict__ for detection in image_response.detections]
        )
    )
    print("\n" + "=" * 50 + "\n")

    response_dict = {
        "image_id": image_response.image_id,
        "encoded_image": output_image_path,
        "detections": get_readable_output(
            [detection.__dict__ for detection in image_response.detections]
        ),
        "raw_detections": [detection.__dict__ for detection in image_response.detections],
    }

    with open(output_file_path, "a") as f:
        # Save the response to a JSONL file
        f.write(json.dumps(response_dict) + "\n")
        f.flush()

    return response_dict


# Example Usage
# if __name__ == "__main__":
#     # Assuming you have a base64-encoded image string and a custom_id
#     encoded_image = encode_image(
#         "path_to_your_image.jpg"
#     )  # Replace with your image path
#     custom_id = "example_image_id"

#     # Get and save the prediction
#     return_prediction(encoded_image, custom_id, output_file_path)
