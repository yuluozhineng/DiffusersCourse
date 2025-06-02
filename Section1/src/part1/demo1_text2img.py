from diffusers import StableDiffusionPipeline
import torch
import time

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "runwayml/stable-diffusion-v1-5" # 基座模型


# 第二部分，配置pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,                    # 模型ID
    torch_dtype=torch.float16 ,  # 使用半精度
    use_safetensors=True         # 是否使用safetensors
).to(device)

# 第三部分，生成图片，默认大小是512*512
prompt = "A realistic photo of a astronaut riding a horse on Mars, 4k, high resolution" # 提示词
image = pipe(
    prompt,
    num_inference_steps=25,  # 推理步数（通常25-50）
    guidance_scale=7.5,       # 提示词相关性系数（通常7-12之间）
    negative_prompt="blurry, low quality"  # 负面提示词
).images[0]

# 第四部分，保存图片，并输出监控信息(耗时和显存占用峰值)
image.save("Section1/output/part1/demo1/text2img_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")



# 执行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo1_text2img.py
# Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  4.34it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:02<00:00,  8.84it/s]
# 图片已生成，执行耗时: 7.724191904067993
# 占用显存峰值: 3256.57 MB
# PS D:\Project\Course\DiffusersCourse> 