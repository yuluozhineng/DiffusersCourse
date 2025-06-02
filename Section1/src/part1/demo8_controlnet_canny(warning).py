import time
import torch
from controlnet_aux import CannyDetector
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# 备注: 这个例子中引入了controlnet_aux，可能会和timm库有些冲突，从而产生警告
# 在最后会把这些警告信息打出来(通过给出控制台信息的方式)

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5" # 基座模型
controlnet_model_id = "lllyasviel/sd-controlnet-canny" # 边缘ControlNet对应的模型名称
controlnet_canny_image_path = "Section1/data/controlnet_canny_image.png" # 边缘控制图

# 第二部分，生成边缘图像
original_image = load_image(controlnet_canny_image_path) # 加载输入图像
canny_detector = CannyDetector() # 创建 Canny 检测器
canny_image = canny_detector(
    original_image, 
    low_threshold=100, # 低阈值，低于此值的边缘被丢弃
    high_threshold=200 # 高阈值，高于此值的边缘被保留为强边缘
)

# 第三部分，生成ControlNet模型
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True
)

# 第四部分，生成对应的Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, 
    controlnet=controlnet, 
    torch_dtype=torch.float16, 
    use_safetensors=True
).to(device)

# 第五部分，执行流水线生成图片
output = pipe(
    "the mona lisa", # 提示词
    image=canny_image, # 使用对应的边缘图
    num_inference_steps=25 # 推理步数
).images[0]

# 第六部分，保存图片并打印监控信息
final_image = make_image_grid([original_image, canny_image, output], rows=1, cols=3)
final_image.save("Section1/output/part1/demo8/controlnet_canny_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")










# 执行时控制台输出(包含警告信息):
# D:\Software\Anaconda\envs\cv1\Lib\site-packages\timm\models\layers\__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
#   warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
# D:\Software\Anaconda\envs\cv1\Lib\site-packages\timm\models\registry.py:4: FutureWarning: Importing from timm.models.registry is deprecated, please import via timm.models
#   warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
# D:\Software\Anaconda\envs\cv1\Lib\site-packages\controlnet_aux\segment_anything\modeling\tiny_vit_sam.py:654: UserWarning: Overwriting tiny_vit_5m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_5m_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
#   return register_model(fn_wrapper)
# D:\Software\Anaconda\envs\cv1\Lib\site-packages\controlnet_aux\segment_anything\modeling\tiny_vit_sam.py:654: UserWarning: Overwriting tiny_vit_11m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_11m_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
#   return register_model(fn_wrapper)
# D:\Software\Anaconda\envs\cv1\Lib\site-packages\controlnet_aux\segment_anything\modeling\tiny_vit_sam.py:654: UserWarning: Overwriting tiny_vit_21m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
#   return register_model(fn_wrapper)
# D:\Software\Anaconda\envs\cv1\Lib\site-packages\controlnet_aux\segment_anything\modeling\tiny_vit_sam.py:654: UserWarning: Overwriting tiny_vit_21m_384 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_384. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
#   return register_model(fn_wrapper)
# D:\Software\Anaconda\envs\cv1\Lib\site-packages\controlnet_aux\segment_anything\modeling\tiny_vit_sam.py:654: UserWarning: Overwriting tiny_vit_21m_512 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_512. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
#   return register_model(fn_wrapper)
# Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  3.16it/s]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:09<00:00,  5.30it/s]
# 图片已生成，执行耗时: 15.565855979919434
# 占用显存峰值: 3988.28 MB
# PS D:\Project\Course\DiffusersCourse> 



