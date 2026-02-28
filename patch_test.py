import torch
from diffusers import StableDiffusionInpaintPipeline
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", weight_name="ip-adapter-faceid_sd15.bin", image_encoder_folder=None)
    print("Success loading without image encoder")
except Exception as e:
    print("Error:", e)
