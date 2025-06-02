import time
import torch
import numpy as np
from PIL import Image
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

# 第四部分，设置会调函数，显示每一步的结果，观察是如何一步步生成图片的
def callback(pipe, step, timestep, callback_kwargs):
    print(f"Current step: {step} , Current timestep: {timestep}")
    # 将潜在空间解码为图像并且保存
    latents = callback_kwargs["latents"]

    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).astype(np.uint8)
    
    # 把图片按步数保存
    image = Image.fromarray(image)
    image.save(f"Section1/output/part1/demo5/step_{step}.png")
    return callback_kwargs


# 第四部分，对原图进行调整后并生成新的图片
prompt = "raining, anime girl, sakura background, Studio Ghibli style, soft colors, cel-shading"
negative_prompt = "blurry, lowres, bad anatomy, extra fingers, deformed face"
image = pipe(prompt,  # 提示词，这里重点是加了下雨，也就是raining
            image=init_image,              # 指定初始图片
            strength=0.5,                  # 控制修改强度
            guidance_scale=7.5,            # 提示词相关性
            num_inference_steps=20,        # 推理步数
            negative_prompt=negative_prompt,  # 负面提示词
            callback_on_step_end=callback, # 回调函数
            callback_on_step_end_tensor_inputs=["latents"] # 只需要放入潜在空间的tensor即可
).images[0]

# 第五部分，对原图和新图拼接后保存，并输出监控信息(耗时和显存占用峰值)
image = make_image_grid([init_image, image], rows=1, cols=2)
image.save("Section1/output/part1/demo5/img2img_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")




# 执行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo5_img2img_step.py
# Loading pipeline components...: 100%|██████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  3.60it/s]
#   0%|                                                                                                 | 0/10 [00:00<?, ?it/s]Current step: 0 , Current timestep: 501
# Current step: 1 , Current timestep: 451
#  10%|████████▉                                                                                | 1/10 [00:00<00:06,  1.29it/s]Current step: 2 , Current timestep: 401
#  20%|█████████████████▊                                                                       | 2/10 [00:01<00:04,  1.94it/s]Current step: 3 , Current timestep: 351
#  30%|██████████████████████████▋                                                              | 3/10 [00:01<00:03,  2.32it/s]Current step: 4 , Current timestep: 301
#  40%|███████████████████████████████████▌                                                     | 4/10 [00:01<00:02,  2.31it/s]Current step: 5 , Current timestep: 251
#  50%|████████████████████████████████████████████▌                                            | 5/10 [00:02<00:01,  2.52it/s]Current step: 6 , Current timestep: 201
#  60%|█████████████████████████████████████████████████████▍                                   | 6/10 [00:02<00:01,  2.65it/s]Current step: 7 , Current timestep: 151
#  80%|███████████████████████████████████████████████████████████████████████▏                 | 8/10 [00:03<00:00,  2.46it/s]Current step: 9 , Current timestep: 51
#  90%|████████████████████████████████████████████████████████████████████████████████         | 9/10 [00:03<00:00,  2.44it/s]Current step: 10 , Current timestep: 1
# 100%|████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:04<00:00,  2.35it/s]
# 图片已生成，执行耗时: 9.55759310722351
# 占用显存峰值: 3256.57 MB
# PS D:\Project\Course\DiffusersCourse>