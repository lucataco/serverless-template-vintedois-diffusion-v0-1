import os
import time
import torch
import base64
from io import BytesIO
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    t1 = time.time()
    model_id = "22h/vintedois-diffusion-v0-1"
    model = StableDiffusionPipeline.from_pretrained(
        model_id
    ).to("cuda")
    t2 = time.time()
    print("Init took - ",t2-t1,"seconds")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative = model_inputs.get('negative', None)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    t1 = time.time()
    with autocast("cuda"):
        image = model(prompt, negative_prompt=negative, num_images_per_prompt=1, num_inference_steps=50, guidance_scale=7.5).images[0]
    t2 = time.time()
    print("Inference took - ",t2-t1,"seconds")
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}