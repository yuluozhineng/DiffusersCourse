import time
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

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
    image.save(f"Section1/output/part1/demo7/step_{step}.png")
    return callback_kwargs


# 第四部分，执行局部重绘
image = pipeline(
    prompt="road", # 提示词
    image=init_image, # 待修改图片
    mask_image=mask_image, # 遮罩图片
    guidance_scale=7.5,    # 提示词相关性系数
    num_inference_steps=20,  # 推理步数
    callback_on_step_end=callback, # 回调函数
    callback_on_step_end_tensor_inputs=["latents"] # 只需要放入潜在空间的tensor即可
).images[0]

# 第五部分，拼接图片并且保存，并输出监控信息(耗时和显存占用峰值)
image = make_image_grid([init_image, image], rows=1, cols=2)
image.save("Section1/output/part1/demo7/inpaint_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")







# 执行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo7_inpaint_step.py        
# Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 20.75it/s]
#   0%|                                                                                                         | 0/20 [00:00<?, ?it/s]Current step: 0 , Current timestep: 951
#   5%|████▊                                                                                            | 1/20 [00:00<00:06,  2.72it/s]Current step: 1 , Current timestep: 901
#  10%|█████████▋                                                                                       | 2/20 [00:00<00:06,  2.74it/s]Current step: 2 , Current timestep: 851
#  15%|██████████████▌                                                                                  | 3/20 [00:01<00:06,  2.79it/s]Current step: 3 , Current timestep: 801
#  20%|███████████████████▍                                                                             | 4/20 [00:01<00:05,  2.79it/s]Current step: 4 , Current timestep: 751
#  25%|████████████████████████▎                                                                        | 5/20 [00:01<00:05,  2.78it/s]Current step: 5 , Current timestep: 701
#  30%|█████████████████████████████                                                                    | 6/20 [00:02<00:05,  2.66it/s]Current step: 6 , Current timestep: 651
#  35%|█████████████████████████████████▉                                                               | 7/20 [00:02<00:04,  2.70it/s]Current step: 7 , Current timestep: 601
#  40%|██████████████████████████████████████▊                                                          | 8/20 [00:02<00:04,  2.63it/s]Current step: 8 , Current timestep: 551
#  45%|███████████████████████████████████████████▋                                                     | 9/20 [00:03<00:04,  2.71it/s]Current step: 9 , Current timestep: 501
#  50%|████████████████████████████████████████████████                                                | 10/20 [00:03<00:03,  2.80it/s]Current step: 10 , Current timestep: 451
#  55%|████████████████████████████████████████████████████▊                                           | 11/20 [00:03<00:03,  2.86it/s]Current step: 11 , Current timestep: 401
#  60%|█████████████████████████████████████████████████████████▌                                      | 12/20 [00:04<00:02,  2.83it/s]Current step: 12 , Current timestep: 351
#  65%|██████████████████████████████████████████████████████████████▍                                 | 13/20 [00:04<00:02,  2.80it/s]Current step: 13 , Current timestep: 301
#  70%|███████████████████████████████████████████████████████████████████▏                            | 14/20 [00:05<00:02,  2.80it/s]Current step: 14 , Current timestep: 251
#  75%|████████████████████████████████████████████████████████████████████████                        | 15/20 [00:05<00:01,  2.85it/s]Current step: 15 , Current timestep: 201
#  80%|████████████████████████████████████████████████████████████████████████████▊                   | 16/20 [00:05<00:01,  2.86it/s]Current step: 16 , Current timestep: 151
#  85%|█████████████████████████████████████████████████████████████████████████████████▌              | 17/20 [00:06<00:01,  2.77it/s]Current step: 17 , Current timestep: 101
#  90%|██████████████████████████████████████████████████████████████████████████████████████▍         | 18/20 [00:06<00:00,  2.69it/s]Current step: 18 , Current timestep: 51
#  95%|███████████████████████████████████████████████████████████████████████████████████████████▏    | 19/20 [00:06<00:00,  2.62it/s]Current step: 19 , Current timestep: 1
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:07<00:00,  2.72it/s]
# 图片已生成，执行耗时: 11.876234769821167
# 占用显存峰值: 3256.72 MB
# PS D:\Project\Course\DiffusersCourse> 

