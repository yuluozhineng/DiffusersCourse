import time
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5" # 基座模型
controlnet_model_id = "lllyasviel/sd-controlnet-canny" # 边缘ControlNet对应的模型名称
controlnet_canny_image_path = "Section1/data/controlnet_canny_guess_image.png" # 边缘控制图

# 第二部分，生成边缘图像
original_image = load_image(controlnet_canny_image_path) # 加载输入图像
image = np.array(original_image)
low_threshold = 100  # 低阈值，低于此值的边缘被丢弃
high_threshold = 200 # 高阈值，高于此值的边缘被保留为强边缘
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)


# 第三部分，生成ControlNet模型
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True
)

# 第四部分，生成对应的Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, 
    controlnet=controlnet, 
    torch_dtype=torch.float16, 
    use_safetensors=True
).to(device)

# 第五部分，执行流水线生成图片
output = pipe(
    "", # 提示词
    image=canny_image, # 使用对应的边缘图
    num_inference_steps=25, # 推理步数
    guess_mode=True  # 确认使用猜测模式
).images[0]

# 第六部分，保存图片并打印监控信息
final_image = make_image_grid([original_image, canny_image, output], rows=1, cols=3)
final_image.save("Section1/output/part1/demo9/controlnet_canny_guess_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")




# 执行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> ^C
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo9_controlnet_canny_guess.py
# Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  3.48it/s] 
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:04<00:00,  5.32it/s] 
# 图片已生成，执行耗时: 10.724677085876465
# 占用显存峰值: 3987.49 MB
# PS D:\Project\Course\DiffusersCourse>

