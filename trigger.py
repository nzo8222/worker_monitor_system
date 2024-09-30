from typing import Optional


class TriggerCounter:
    def __init__(self):
        self.consecutive_trigger_count = 0
        self.trigger_limit = 20
        self.last_image_id = None
        self.last_detections = None

    def reset(self):
        self.consecutive_trigger_count = 0
        self.last_image_id = None
        self.last_detections = None

    def increment(self, image_id: str, detections: list[dict]):
        self.consecutive_trigger_count += 1
        self.last_image_id = image_id
        self.last_detections = detections

    def check_trigger(self) -> bool:
        return self.consecutive_trigger_count >= self.trigger_limit

    def get_last_trigger_info(self) -> tuple[Optional[str], Optional[list[dict]]]:
        return self.last_image_id, self.last_detections


trigger_counters = {}


def trigger(image_id: str, detections: list[dict]):
    global trigger_counters

    # Since image_id is different for each frame, we only track the latest one in a single counter
    if "global" not in trigger_counters:
        trigger_counters["global"] = TriggerCounter()

    # Get the global trigger counter
    trigger_counter = trigger_counters["global"]

    trigger_occurred = False

    # Only proceed if detections list is not empty
    if detections:  # Ensure that the detections list is not empty or None
        for person_idx, person in enumerate(detections):
            # Iterate through each key-value pair in the person's detection dictionary
            all_zero = all(value == 0.0 for value in person.values())

            for item, value in person.items():
                # Check if the value is less than 0.1
                if item == "crocs" and all_zero:
                    continue
                if value < 0.1 and value >= 0.0:
                    trigger_occurred = True

            if all_zero:
                trigger_occurred = True

            # If the trigger condition occurred, increment the counter and store the current detection
            if trigger_occurred:
                trigger_counter.increment(image_id, detections)
                break

    # If the trigger has occurred for 20 consecutive times, print the result and reset the counter
    if trigger_counter.check_trigger():
        last_image_id, last_detections = trigger_counter.get_last_trigger_info()
        with open("trigger.txt", "a") as file:
            for person_idx, person in enumerate(last_detections):
                all_zero = all(value == 0.0 for value in person.values())
                for item, value in person.items():
                    if item == "crocs" and all_zero:
                        continue
                    if value < 0.1 and value >= 0.0:
                        output = f"Trigger for Image ID: {last_image_id}, worker {person_idx + 1}, is not wearing Item: {item}, Value: {value}"
                        print(output)
                        file.write(output + "\n")

                        break
                if all_zero:
                    output = f"Trigger for Image ID: {last_image_id}, worker {person_idx + 1} is not wearing crocs"
                    print(output)
                    file.write(output + "\n")

            file.write("\n" + "=" * 50 + "\n")
            print("\n" + "=" * 50 + "\n")

        # Reset the trigger counter after printing
        trigger_counter.reset()

    # Reset the counter if no trigger occurred in this round
    if not trigger_occurred:
        trigger_counter.reset()


if __name__ == "__main__":
    print("This file is a library and should not be executed directly.")
