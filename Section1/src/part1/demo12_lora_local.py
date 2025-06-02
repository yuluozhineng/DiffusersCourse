import time
import torch
from diffusers import AutoPipelineForText2Image

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "runwayml/stable-diffusion-v1-5" # 基座模型
lora_path = "Section1/data" # lora的地址
lora_file_name = "lora_local.safetensors" # 文件名

# 第二部分，配置pipeline
pipeline = AutoPipelineForText2Image.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
).to(device)

# 第三部分，加载本地的lora
pipeline.load_lora_weights(lora_path, weight_name=lora_file_name)

# 第四部分，开始生成图片
prompt = "kafuu chino,1 girl, light linen pink slightly curly hair, ponytail,dark black eyes,x hair ornament" 
image = pipeline(
    prompt, # 正向提示词
    negative_prompt = "blurry, low quality, deformed", # 负向提示词
    num_inference_steps=25  # 推理步数
).images[0]

# 第五部分，保存图片，并输出监控信息(耗时和显存占用峰值)
image.save("Section1/output/part1/demo12/lora_output_" + str(time.time())   + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")



# 执行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo12_lora_local.py
# Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  3.31it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:04<00:00,  5.75it/s]
# 图片已生成，执行耗时: 12.075511455535889
# 占用显存峰值: 3406.89 MB
# PS D:\Project\Course\DiffusersCourse> 