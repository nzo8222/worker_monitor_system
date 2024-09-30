import ast
import json
from typing import Dict, List


class PersonDetection:
    def __init__(
        self,
        white_rubber_boots: float = 0.0,
        brown_pants: float = 0.0,
        knee_protector: float = 0.0,
        brown_long_sleeves_shirt: float = 0.0,
        glasses: float = 0.0,
        gloves: float = 0.0,
        gas_mask: float = 0.0,
        crocs: float = 0.0,
    ):
        self.white_rubber_boots = white_rubber_boots
        self.brown_pants = brown_pants
        self.knee_protector = knee_protector
        self.brown_long_sleeves_shirt = brown_long_sleeves_shirt
        self.glasses = glasses
        self.gloves = gloves
        self.gas_mask = gas_mask
        self.crocs = crocs

    def __repr__(self):
        return (
            f"PersonDetection(white_rubber_boots={self.white_rubber_boots}, brown_pants={self.brown_pants}, "
            f"knee_protector={self.knee_protector}, brown_long_sleeves_shirt={self.brown_long_sleeves_shirt}, "
            f"glasses={self.glasses}, gloves={self.gloves}, gas_mask={self.gas_mask}, crocs={self.crocs})"
        )


class ImageResponse:
    def __init__(self, image_id: str, detections: List[PersonDetection]):
        self.image_id = image_id
        self.detections = detections

    def __repr__(self):
        return f"ImageResponse(image_id={self.image_id}, detections={self.detections})"


def parse_response(response_data: Dict) -> List[PersonDetection]:
    detections = []
    content = response_data.get("choices", [])[0].get("message", {}).get("content", "")
    # Remove the code block markers and safely evaluate the string as a Python literal or JSON
    content = content.strip().strip("```python").strip("```json").strip("```").strip()

    # Check if the content starts with a valid JSON list or dictionary
    if not (content.startswith("[") or content.startswith("{")):
        print("Invalid content format, skipping...")
        print(f"Skipped content: {content}")
        return []

    try:
        person_list = ast.literal_eval(content)
    except Exception as e:
        print(f"Error evaluating content: {e}")
        print(f"Skipped content: {content}")
        return []

    for person in person_list:
        detections.append(
            PersonDetection(
                white_rubber_boots=person.get("White Rubber Boots", 0.0),
                brown_pants=person.get("Brown Pants", 0.0),
                knee_protector=person.get("Knee Protector", 0.0),
                brown_long_sleeves_shirt=person.get("Brown Long Sleeves Shirt", 0.0),
                glasses=person.get("Glasses", 0.0),
                gloves=person.get("Gloves", 0.0),
                gas_mask=person.get("Gas Mask", 0.0),
                crocs=person.get("crocs", 0.0),
            )
        )
    return detections


def main():
    # Path to the JSONL file
    input_file_path = "responses_updated.jsonl"

    # Read the JSONL file
    with open(input_file_path, "r") as f:
        lines = f.readlines()

    # Process each line
    for line in lines:
        response_data = json.loads(line)
        image_id = response_data["image_id"]
        response_content = response_data["response"]

        # Parse the response content
        detections = parse_response(response_content)
        image_response = ImageResponse(image_id=image_id, detections=detections)

        # Print the parsed data
        print(image_response)


if __name__ == "__main__":
    main()
