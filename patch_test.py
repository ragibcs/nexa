import torch
from diffusers import StableDiffusionInpaintPipeline
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")
    # Don't even try load_ip_adapter, we are testing the pass structure
    import numpy as np
    from PIL import Image
    init_image = Image.new('RGB', (512, 512))
    mask_image = Image.new('RGB', (512, 512))
    
    embed = np.random.rand(512).astype(np.float32)
    faceid_embeds = torch.tensor(embed).unsqueeze(0).unsqueeze(0)
    
    pipe.unet.config.addition_embed_type = "image_proj"
    
    try:
        pipe(prompt="", image=init_image, mask_image=mask_image, cross_attention_kwargs={"ip_adapter_image_embeds": [faceid_embeds]}, num_inference_steps=1)
        print("Success")
    except Exception as e:
        print("Failed:", e)
        
except Exception as e:
    pass
