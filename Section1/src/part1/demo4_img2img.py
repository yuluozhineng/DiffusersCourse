import time
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "runwayml/stable-diffusion-v1-5" # 基座模型

# 第二部分，配置pipeline
pipe = AutoPipelineForImage2Image.from_pretrained(
    model_id,                    # 模型ID
    torch_dtype=torch.float16 ,  # 使用半精度
    use_safetensors=True         # 是否使用safetensors
).to(device)

# 第三部分，指定原始图片并加载
img_path = "Section1/data/1girl.png"
init_image = load_image(img_path)

# 第四部分，对原图进行调整后并生成新的图片
prompt = "raining, anime girl, sakura background, Studio Ghibli style, soft colors, cel-shading"
negative_prompt = "blurry, lowres, bad anatomy, extra fingers, deformed face"
image = pipe(prompt,  # 提示词，这里重点是加了下雨，也就是raining
            image=init_image,              # 指定初始图片
            strength=0.5,                  # 控制修改强度（0.3~0.7平衡输入与生成）
            guidance_scale=7.5,            # 提示词相关性（7-9为合理范围）
            num_inference_steps=20,        # 推理步数（20-40步性价比高）
            negative_prompt=negative_prompt  # 负面提示词
).images[0]

# 第五部分，对原图和新图拼接后保存，并输出监控信息(耗时和显存占用峰值)
image = make_image_grid([init_image, image], rows=1, cols=2)
image.save("Section1/output/part1/demo4/img2img_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")


# 执行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo4_img2img.py
# Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  4.55it/s] 
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.12it/s]
# 图片已生成，执行耗时: 5.970721483230591
# 占用显存峰值: 3256.57 MB
# PS D:\Project\Course\DiffusersCourse> 


