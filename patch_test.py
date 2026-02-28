import torch
from diffusers import StableDiffusionInpaintPipeline
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", weight_name="ip-adapter-faceid_sd15.bin")
    
    import numpy as np
    from PIL import Image
    init_image = Image.new('RGB', (512, 512))
    mask_image = Image.new('RGB', (512, 512))
    
    embed = np.random.rand(512).astype(np.float32)
    faceid_embeds = torch.tensor(embed).unsqueeze(0).unsqueeze(0)
    
    try:
        pipe(prompt="", image=init_image, mask_image=mask_image, ip_adapter_image_embeds=[faceid_embeds], num_inference_steps=1)
        print("Success with [tensor]")
    except Exception as e:
        print("Failed [tensor]:", e)

    try:
        pipe(prompt="", image=init_image, mask_image=mask_image, ip_adapter_image_embeds=[[faceid_embeds]], num_inference_steps=1)
        print("Success with [[tensor]]")
    except Exception as e:
        print("Failed [[tensor]]:", e)
        
except Exception as e:
    print("Init Error:", e)
