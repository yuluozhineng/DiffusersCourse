import time
import torch
from diffusers import AutoPipelineForText2Image

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "runwayml/stable-diffusion-v1-5" # 基座模型

# 第二部分，构建pipeline
pipeline = AutoPipelineForText2Image.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
).to("cuda")

# 第三部分，加载Textual Inversion
pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")

# 第四部分，生成新图片
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, <gta5-artwork> style"
image = pipeline(prompt).images[0]

# 第五部分，保存图片，并输出监控信息(耗时和显存占用峰值)
image.save("Section1/output/part1/demo15/textual_inversion_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")



# 执行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo15_textual_inversion.py
# Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  3.21it/s]
# The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:07<00:00,  7.10it/s]
# 图片已生成，执行耗时: 12.322498083114624
# 占用显存峰值: 3257.19 MB
# PS D:\Project\Course\DiffusersCourse> 