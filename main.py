import json

from stream_pipeline import run_realtime_capture
from utils import clean_old_entries_in_jsonl, clean_old_images, detect_motion

if __name__ == "__main__":
    
    ips=['192.168.1.6','192.168.1.7']

    while True:
        if detect_motion(ips):
            print("Motion detected! Running classification pipeline.")
            run_realtime_capture()
            clean_old_images()
            clean_old_entries_in_jsonl()
        else:
            print("No motion detected, waiting for motion...")
