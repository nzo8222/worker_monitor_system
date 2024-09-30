import requests
from requests.auth import HTTPDigestAuth

# Camera details
camera_ip = '192.168.1.6'  # Replace with your camera's IP address
username = 'admin'         # Replace with your camera's username
password = 'Mayocabo1' # Replace with your camera's password

# ISAPI endpoint to receive event stream
events_url = f'http://{camera_ip}/ISAPI/Event/notification/alertStream'

# Function to monitor motion detection events
def monitor_motion_events():
    try:
        # Request the event stream from the camera with Digest Authentication
        response = requests.get(events_url, auth=HTTPDigestAuth(username, password), stream=True)

        print("Listening for motion detection events...")

        # Process the incoming event stream
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')

                # Check if the event contains motion detection information (VMD: Video Motion Detection)
                if "<eventType>VMD</eventType>" in decoded_line:
                    print(f"Motion detected: {decoded_line}")
                elif "<eventType>videoloss</eventType>" in decoded_line:
                    print(f"Video Loss Event: {decoded_line}")
                else:
                    print(f"Other Event: {decoded_line}")

    except Exception as e:
        print(f"Error occurred: {e}")

# Start monitoring
if __name__ == "__main__":
    monitor_motion_events()
