import os
import json
from guess import *

def scan_videos(directory):
    video_files = sorted([f for f in os.listdir(directory) if f.endswith('.mp4')])
    videos_info = []
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        door_states = guess_door_states(video_path)
        annotations = []
        for state_id, (frame, description) in enumerate(door_states, 1):
            annotations.append({
                "state_id": state_id,
                "description": description,
                "guessed_frame": frame
            })
        videos_info.append({
            "video_filename": video_file,
            "annotations": [
                {
                    "object": "Door",
                    "states": annotations
                }
            ]
        })
    return videos_info

def generate_json(output_filename, videos_info):
    with open(output_filename, 'w') as file:
        json.dump({"videos": videos_info}, file, indent=4)

def main():
    directory = "Evaluations"
    output_filename = "output.json"
    videos_info = scan_videos(directory)
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")

if __name__ == "__main__":
    main()