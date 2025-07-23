import torch
from diffusers import StableDiffusionXLPipeline
import requests
import json

# Check if a GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pre-trained Stable Diffusion XL model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16"
)
pipe = pipe.to(device)

# Your text prompt
# prompt = "A high-quality photo of Los Angeles, California, at sunset, with a clear sky and the city lights starting to twinkle."
prompt = "Create a high quality image of a boy and her younger sibling playing in a park, with a bright blue sky and green grass, capturing the joy and innocence of childhood."

# Use Ollama (Gemma3:latest) to generate a high-quality, descriptive image name based on the prompt
ollama_url = "http://localhost:11434/api/generate"
ollama_payload = {
    "model": "gemma3:27b",
    "temprature": 0.9,
    "prompt": (
        "Your role is to create comprehensive, detailed filenames for images based on their descriptions. "
        "Follow these examples- description: A cat sitting on a windowsill during a rainy day.\n"
        "Filename: cat_on_windowsill_rainy_day_peaceful_scene\n"
        "Description: A futuristic city skyline at night with neon lights.\n"
        "Filename: futuristic_city_skyline_neon_night_lights\n"
        f"Description: {prompt}\n"
        "Filename: "
    )
}

response = requests.post(ollama_url, json=ollama_payload)

# Handle streaming/multi-line JSON response
lines = response.text.strip().splitlines()
filename = "generated_image"
for line in lines:
    try:
        data = json.loads(line)
        if "response" in data:
            filename = data["response"].strip().replace(" ", "_")
            break
    except json.JSONDecodeError:
        continue

filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
filename = filename[:50] or "generated_image"

# Generate the image
image = pipe(prompt=prompt).images[0]

# Save the image to a file with the generated name
image.save(f"{filename}.png")

print(f"Image saved as {filename}.png")