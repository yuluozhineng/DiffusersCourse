from diffusers import AutoPipelineForText2Image
import torch
from diffusers.utils import load_image
import time

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "runwayml/stable-diffusion-v1-5" # 基座模型

# 第二部分，定义Pipeline
pipeline = AutoPipelineForText2Image.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
).to(device)

# 第三部分，加载IP-Adapter模型文件
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

# 第四部分，加载提示图像并生成新图片
image = load_image("Section1/data/ip_adapter.png")
image = pipeline(
    prompt='best quality, high quality',
    ip_adapter_image=image, # 定义输入的图片提示
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=50
).images[0]

# 第五部分，保存图像并记录系统性能监控
image.save("Section1/output/part1/demo14/ip_adapter_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")


# 执行时控制台输出
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo14_ip_adapter.py
# Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████| 7/7 [00:04<00:00,  1.54it/s] 
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.25it/s] 
# 图片已生成，执行耗时: 19.32180881500244
# 占用显存峰值: 4505.25 MB
# PS D:\Project\Course\DiffusersCourse>
