import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL

def encode_prompt(tokenizer, text_encoder, prompt):
    """将文本提示编码为嵌入向量"""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    
    # 为无条件输入创建空文本嵌入（用于分类器自由引导）
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    # 连接条件和无条件嵌入
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings

def generate_noise_latents(scheduler, height, width, num_channels=4, batch_size=1, seed=None):
    """生成随机噪声latents作为初始输入"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # 创建随机噪声
    latents_shape = (batch_size, num_channels, height // 8, width // 8)
    latents = torch.randn(latents_shape, device=device)
    
    # 根据调度器的配置缩放噪声
    latents = latents * scheduler.init_noise_sigma
    return latents

def denoise_step(scheduler, unet, latents, t, text_embeddings, guidance_scale=7.5):
    """执行一步去噪过程"""
    # 复制latents用于条件和无条件预测
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # 预测噪声
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    
    # 执行分类器自由引导
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # 拆分为无条件和有条件两部分
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(noise_pred, t, latents).prev_sample     # 更新latents
    return latents

def decode_latents(latents, vae):
    """将latents解码为图像像素"""
    # 缩放和解码latents
    latents = 1 / vae.config.scaling_factor * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    # 转换为PIL图像
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# 第一部分，参数配置
device = "cuda" if torch.cuda.is_available() else "cpu" # 优先使用GPU
model_id = "runwayml/stable-diffusion-v1-5" # 模型ID

# 第二部分，配置所需的各种组件
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
unet.eval()

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer") # 分词器
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device) # 文本编码器
text_encoder.eval()

scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler") # 调度器
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device) # vae

# 第三部分, 配置生成图片的参数
prompt = "A realistic photo of a astronaut riding a horse on Mars, 4k, high resolution" # 提示词
height, width = 512, 512 # 高度和宽度
num_inference_steps = 30 # 推理步数
guidance_scale = 7.5 # 提示词相关性系数
seed = 77 # 随机种子

# 第四部分，去噪之前的准备
text_embeddings = encode_prompt(tokenizer, text_encoder, prompt)  # 编码提示词
latents = generate_noise_latents(scheduler, height, width, seed=seed) # 生成初始噪声
scheduler.set_timesteps(num_inference_steps)     # 设置调度器的时间步

# 第五部分，执行去噪过程
for i, t in enumerate(scheduler.timesteps):
    latents = denoise_step(scheduler, unet, latents, t, text_embeddings, guidance_scale)
    image = decode_latents(latents, vae)
    image[0].save("Section1/output/part2/demo6/step_" + str(i+1) + "_.png")
    print(f"完成去噪步数: {i+1}/{num_inference_steps}")

# 第六部分，把数据从潜在空间解码到像素空间，然后保存
images = decode_latents(latents, vae)
images[0].save("Section1/output/part2/demo6/generated_image.png")



# 运行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part2/demo6_process.py
# 完成去噪步数: 1/30
# 完成去噪步数: 2/30
# 完成去噪步数: 3/30
# 完成去噪步数: 4/30
# 完成去噪步数: 5/30
# 完成去噪步数: 6/30
# 完成去噪步数: 7/30
# 完成去噪步数: 8/30
# 完成去噪步数: 9/30
# 完成去噪步数: 10/30
# 完成去噪步数: 11/30
# 完成去噪步数: 12/30
# 完成去噪步数: 13/30
# 完成去噪步数: 14/30
# 完成去噪步数: 15/30
# 完成去噪步数: 16/30
# 完成去噪步数: 17/30
# 完成去噪步数: 18/30
# 完成去噪步数: 19/30
# 完成去噪步数: 20/30
# 完成去噪步数: 21/30
# 完成去噪步数: 22/30
# 完成去噪步数: 23/30
# 完成去噪步数: 24/30
# 完成去噪步数: 25/30
# 完成去噪步数: 26/30
# 完成去噪步数: 27/30
# 完成去噪步数: 28/30
# 完成去噪步数: 29/30
# 完成去噪步数: 30/30
# PS D:\Project\Course\DiffusersCourse> 