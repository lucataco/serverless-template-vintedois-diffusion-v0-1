
# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import time
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    t1 = time.time()
    model_id = "22h/vintedois-diffusion-v0-1"
    model = StableDiffusionPipeline.from_pretrained(
        model_id
    )
    t2 = time.time()
    print("Download took - ",t2-t1,"seconds")

if __name__ == "__main__":
    download_model()