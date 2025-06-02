from diffusers import StableDiffusionPipeline
import torch
import time

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
##################################################################################################
#
# 注意: 这里的model_path是本地的基座模型，需要替换为你自己的
# 我这里使用的是麦橘写实的V7版本，如果读者朋友想要复现，需要替换为自己本地的模型
#
##################################################################################################
model_path = "C:/Users/yuluo/Downloads/majicmix7.safetensors" # 本地的基座模型，需要替换成你自己的

# 第二部分，配置pipeline
pipe = StableDiffusionPipeline.from_single_file(
    model_path,                    # 模型路径
    torch_dtype=torch.float16  # 使用半精度
).to(device)

# 第三部分，生成图片，默认大小是512*512
prompt = "1girl" # 使用经典提示词
image = pipe(
    prompt, # 正向提示词
    num_inference_steps=25,  # 推理步数
    guidance_scale=7.5,       # 提示词相关性系数
    negative_prompt="blurry, low quality"  # 负面提示词
).images[0]

# 第四部分，保存图片，并输出监控信息(耗时和显存占用峰值)
image.save("Section1/output/part2/demo7/text2img_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")


# 运行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part2/demo7_local.py
# Fetching 11 files: 100%|██████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<?, ?it/s]
# Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 28.75it/s]
# You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:02<00:00,  8.75it/s]
# 图片已生成，执行耗时: 4.955540418624878
# 占用显存峰值: 2675.41 MB
# PS D:\Project\Course\DiffusersCourse> 



