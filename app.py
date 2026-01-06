import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import os
from datetime import datetime

# Folder to save generated images
SAVE_DIR = "generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

def generate_image(prompt):
    image = pipe(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]

    # Create unique filename
    filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(SAVE_DIR, filename)

    # Save image
    image.save(filepath)

    return filepath, filepath


interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(
        lines=6,
        placeholder="Enter a long image prompt here..."
    ),
    outputs=[
        gr.Image(type="filepath", label="Generated Image"),
        gr.File(label="Download Image")
    ],
    title="AI Text to Image Generator",
    description="An AI/ML system using Stable Diffusion to generate realistic images from text prompts."
)

interface.launch()
