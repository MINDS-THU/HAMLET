import base64
import json
import mimetypes
import os
import uuid
from io import BytesIO

import PIL.Image
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from smolagents import Tool, tool


load_dotenv(override=True)

# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resize_image(image_path):
    img = PIL.Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path


@tool
def visualizer(image_path: str, question: str | None = None) -> str:
    """A tool that can answer questions about attached images.

    Args:
        image_path: The path to the image on which to answer the question. This should be a local path to downloaded image.
        question: The question to answer.
    """
    import mimetypes
    import os

    import requests

    # Use the encode_image function from this same module instead of relative import
    # from .visual_qa import encode_image

    add_note = False
    if not question:
        add_note = True
        question = "Please write a detailed caption for this image."
    if not isinstance(image_path, str):
        raise Exception("You should provide at least `image_path` string argument to this tool!")

    mime_type, _ = mimetypes.guess_type(image_path)
    base64_image = encode_image(image_path)  # Use the function defined in this module

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        "max_tokens": 1000,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        output = response.json()["choices"][0]["message"]["content"]
    except Exception:
        raise Exception(f"Response format unexpected: {response.json()}")

    if add_note:
        output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

    return output


if __name__ == "__main__":
    # Simple test of the visual QA functionality
    import os
    
    # Test image path
    test_image_path = r"C:\Users\li_ch\OneDrive\Pictures\MINDS_icon.png"
    
    # Check if the image exists
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        print("Please make sure the image exists or update the path.")
    else:
        print(f"Testing visual QA with image: {test_image_path}")
        
        # Test basic image loading first
        print("\n=== Testing basic image loading ===")
        try:
            import PIL.Image
            img = PIL.Image.open(test_image_path)
            print(f"Image loaded successfully: {img.size} pixels, mode: {img.mode}")
        except Exception as e:
            print(f"Failed to load image: {e}")
            exit(1)
        
        # Test the visualizer function
        print("\n=== Testing visualizer function ===")
        try:
            result = visualizer(test_image_path, "Describe this image in detail.")
            print(f"Visualizer result: {result}")
        except Exception as e:
            print(f"Visualizer error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
        
        print("\n=== Test completed ===")