import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.utils import make_image_grid

def encode_image_to_latents(vae, image_path):
    """将图像编码为latents"""
    # 加载并预处理图像
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0  # 归一化到[-1,1]
    image_tensor = image_tensor.to(device)
    
    # 编码为latents
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # 执行缩放
    
    return latents


def decode_latents_to_image(vae, latents):
    """将latents解码回图像"""
    # 解码latents
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor  # 反向缩放
        image = vae.decode(latents).sample
    
    # 后处理图像
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image_pil = Image.fromarray((image * 255).round().astype("uint8"))
    
    return image_pil

# 第一部分，初始化vae
device = "cuda" if torch.cuda.is_available() else "cpu" # GPU优先
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
vae.eval() 

# 第二部分，对图像进行编码(从像素空间到潜在空间)
image_path = "Section1/data/astronaut.png"
latents = encode_image_to_latents(vae, image_path) # 编码图像到latents
print("潜在空间的tensor的形状:", latents.shape)
print("vae的缩放因子:", vae.config.scaling_factor)

# 第三部分，解码潜在空间到像素空间，拼接后保存
reconstructed_image = decode_latents_to_image(vae, latents)
final_image = make_image_grid([Image.open(image_path), reconstructed_image], rows=1, cols=2)
final_image.save("Section1/output/part2/demo3/vae_demo_.png")

# 运行时的控制台输出
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part2/demo3_vae.py       
# 潜在空间的tensor的形状: torch.Size([1, 4, 64, 64])
# vae的缩放因子: 0.18215
# PS D:\Project\Course\DiffusersCourse> 



