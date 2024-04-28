import imageio
import numpy as np
import os
import base64

def extract_frames(video_path, num_frames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames_paths = []
    for i, frame_index in enumerate(frame_indices):
        frame = reader.get_data(frame_index)
        frame_path = os.path.join(output_folder, f"frame_{i}.png")
        imageio.imwrite(frame_path, frame)
        frames_paths.append(frame_path)

    reader.close()
    return frames_paths

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')