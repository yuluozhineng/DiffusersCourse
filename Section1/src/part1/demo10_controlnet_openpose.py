import time
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# 第一部分，参数准备
start_time = time.time() # 开始计时
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置可用的设备，cuda优先
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5" # 基座模型
controlnet_model_id = "lllyasviel/control_v11p_sd15_openpose" # 姿势ControlNet对应的模型名称
controlnet_openpose_path = "Section1/data/controlnet_openpose_image.png" # 姿势控制图

# 第二部分，生成姿势控制图
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet") # 加载 OpenPose 检测器
original_image = load_image(controlnet_openpose_path)   # 加载姿势参考图
openpose_image = openpose(original_image) # 提取姿势关键点

# 第三部分，生成ControlNet模型
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id, 
    torch_dtype=torch.float16
)

# 第四部分，生成对应的Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet, # 指定使用的Controlnet
    torch_dtype=torch.float16
).to(device)

# 第五部分，执行流水线生成图片
prompt = "a stylish model posing in a studio, high fashion photoshoot, 4k, highly detailed"
output_image = pipe(
    prompt, # 提示词
    negative_prompt="low quality, blurry, ugly, deformed", # 负向提示词
    image=openpose_image, # 姿态图片
    num_inference_steps=20,
    controlnet_conditioning_scale=0.8  # 控制ControlNet影响强度(建议0.8到1.2)
).images[0]


# 第六部分，保存图片并打印监控信息
final_image = make_image_grid([original_image, openpose_image, output_image], rows=1, cols=3)
final_image.save("Section1/output/part1/demo10/controlnet_openpose_" + str(start_time) + "_.png")
print("图片已生成，执行耗时:", time.time() - start_time)
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"占用显存峰值: {peak_allocated:.2f} MB")



# 运行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part1/demo10_controlnet_openpose.py
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
# Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  3.45it/s] 
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:04<00:00,  4.91it/s] 
# 图片已生成，执行耗时: 14.528980493545532
# 占用显存峰值: 3988.28 MB
# PS D:\Project\Course\DiffusersCourse>










