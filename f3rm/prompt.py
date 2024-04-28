import requests
from f3rm.utils import extract_frames, encode_image
from typing import Dict
import os
import torch
import base64
from PIL import Image
from io import BytesIO
import json


API_KEY = os.getenv("OPENAI_API_KEY")

def get_data_json(image_tensor: torch.Tensor,image_path:str, prompt: str="", max_tokens: int=100):
    content = [{"type": "text", "text": prompt}]
    if image_path is None:
        base64_image = encode_tensor_to_base64(image_tensor)
    else:
        base64_image=encode_image(image_path)
    image_data = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }
    content.append(image_data)
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
            "role": "user",
            "content": content
            }
        ],
        "max_tokens": max_tokens
    }
    
    return payload

def get_response(payload: Dict[str, object], api_key: str=API_KEY):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def encode_tensor_to_base64(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)  # Ensure the tensor values are within [0, 1]
    image = Image.fromarray((tensor.numpy() * 255).astype('uint8'))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode('utf-8')
    return base64_image

select_cluster_prompt="""
    You are given a scene. We will give command to the robot and a robot will grasp an object. Help the robot to grasp the right object by answering the following questions.
    The user command is {}, the identified target_object is {}, the identified reference_object is {}
    The following coordinates [x,y], x is the 
    We have used clustering to find some centers of the target_objects: {}.
    We have used clustering to find some centers of the reference_object: {}.
    Based on the command, simply return a JSON string like below, make sure it can be sent to json.loads().
    Hint: For relationship like "near"/"beside"/"under"/"on", the target should be very very close to the targets (in both x and y coordinates). 
    If there are reference object:
    Due to the view perspective, for "under" relationship, you should choose the one that have both close enough x and y coordinates (if there is one coordinate gap > 50, it is not prioritized). To help you, I give you the relative coordinates gap:{}
    Otherwise reference object is None, you should remember that larger x is right, larger y is up or behind, smaller y is down or front.
    Don't be fooled by the coordinates.
    Keep in mind that only output the dictionary string, don't add '''json''' patterns.
    {{
        "reasoning":"If there are reference, reason about the relationship with the reference object, otherwise explicitly state the target objects coordinates and compare as reasoning. Briefly reason about selecting which cluster id in < 35 words"
        'target_object_cluster':'Your answer. Which cluster is the target object we want? Choose the cluster id.',
    }}
    """

semantic_parsing= """
    You are given a scene. We will give command to the robot and a robot will grasp an object. Help the robot to grasp the right object by answering the following questions.
    The user command is {}
    Based on the command, simply return a JSON string like below, make sure it can be sent to json.loads().
    Keep in mind that only output the dictionary string, don't add '''json''' patterns.
    {{
        'target_object':'Your answer. What is the target object that the user wants?',
        'reference_object': 'Your answer. What reference object does the user specify for locating the target object? It's likely that there are no reference object, just say None. Ensure that reference object is different from the target object if there is.',
    }}
    """

if __name__=="__main__":
    prompt = """
    You are given a scene. We will give command to the robot and a robot will grasp an object. Help the robot to grasp the right object by answering the following questions.
    The user command is {}, the identified target_object is {}, the identified reference_object is {}
    The following coordinates [x,y], x is the 
    We have used clustering to find some centers of the target_objects: {}.
    We have used clustering to find some centers of the reference_object: {}.
    Based on the command, simply return a JSON string like below, make sure it can be sent to json.loads(). Don't add '''json'''.
    {{
        'target_object_cluster':'Your answer. Which cluster is the target object we want? Choose the cluster id.',
    }}
    """
   
    max_tokens = 100
    # rgb=torch.load("./rgb_info.pth")
    payload = get_data_json(image_tensor=None, image_path="./rendered_image.png",prompt=prompt.format("Get the wooden block under the mug","wooden block","metal mug"), max_tokens=max_tokens)
    response = get_response(payload)['choices'][0]['message']['content']
    print(json.loads(response))