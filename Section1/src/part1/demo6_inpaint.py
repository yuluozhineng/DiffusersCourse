import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import time

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "runwayml/stable-diffusion-inpainting" # 基座模型

# 第二部分，配置pipeline
pipeline = AutoPipelineForInpainting.from_pretrained(
    model_id,                       # 模型ID
    torch_dtype=torch.float16,      # 使用半精度
    variant="fp16"                  # 使用fp16变体
).to(device)

# 第三部分，加载原图和遮罩图片
init_image = load_image("Section1/data/road-inpaint.png")
mask_image = load_image("Section1/data/road-mask.png")

# 第四部分，执行局部重绘
image = pipeline(
    prompt="road",   # 提示词
    image=init_image,  # 待修改图片
    mask_image=mask_image, # 遮罩图片
    guidance_scale=7.5,    # 提示词相关性系数
    num_inference_steps=20  # 推理步数
).images[0]

# 第五部分，拼接图片并且保存，并输出监控信息(耗时和显存占用峰值)
image = make_image_grid([init_image, image], rows=1, cols=2)
image.save("Section1/output/part1/demo6/inpaint_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")



# 执行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo6_inpaint.py
# Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 23.29it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:02<00:00,  7.63it/s]
# 图片已生成，执行耗时: 7.452728748321533
# 占用显存峰值: 3256.72 MB
# PS D:\Project\Course\DiffusersCourse> 



