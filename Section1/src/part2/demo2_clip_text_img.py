import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# 第一部分，配置参数
device = "cuda" if torch.cuda.is_available() else "cpu" # 选择运行设备
model_id = "openai/clip-vit-base-patch16" # 模型ID
image_path = "Section1/data/astronaut.png" # 图片路径
# 待判断与该图片相似的文本列表
text_descriptions = [
    "A astronaut riding a horse on Mars",
    "A fish",
    "A astronaut",
    "A horse"
]

# 第二部分，加载模型
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)

# 第三部分，计算图片的特征数据
image = Image.open(image_path).convert("RGB") # 加载图像
inputs = processor(images=image, return_tensors="pt").to(device) # 处理图片
with torch.no_grad():
    image_features = model.get_image_features(**inputs) # 获取图像特征
image_features = image_features / image_features.norm(dim=-1, keepdim=True) # 归一化特征向量

# 第四部分，依次计算文本的特征数据并求余弦值
similarities = []
for text in text_descriptions:
    # 使用处理器处理文本
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化特征向量
    # 计算相似度
    similarity = torch.matmul(image_features, text_features.T).cpu().numpy()[0][0]
    similarities.append(similarity)
    print(f"与文本 '{text}': 相似度 {similarity:.4f}")

# 第五部分，结果可视化
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Origin Image")
plt.axis("off")

plt.subplot(1, 2, 2)
bars = plt.bar(text_descriptions, similarities, color='skyblue')
plt.ylim(0, 1)
plt.title("similarity")
plt.ylabel("cosine similarity")
plt.xticks(rotation=45, ha='right')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig("Section1/output/part2/demo1/clip_similarity_results.png")
plt.show()

# 运行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part2/demo2_clip_text_img.py
# 与文本 'A astronaut riding a horse on Mars': 相似度 0.3555
# 与文本 'A fish': 相似度 0.1577
# 与文本 'A astronaut': 相似度 0.2773
# 与文本 'A horse': 相似度 0.2536
# PS D:\Project\Course\DiffusersCourse> 
