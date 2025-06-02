import torch
from diffusers import UNet2DConditionModel

# 第一部分，参数配置
device = "cuda" if torch.cuda.is_available() else "cpu" # 优先使用GPU
model_id = "runwayml/stable-diffusion-v1-5" # 模型ID

# 第二部分，配置所需的各种组件
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
unet.eval()

# 第三部分，模拟unet的模型预测
latents_shape = (1, 4, 64, 64) # 潜在空间的维度 (batch_size, channels, height, width)
noisy_latents = torch.randn(latents_shape, dtype=torch.float16).to(device) # 模拟上一次得到的噪声
timestep = torch.tensor([10], device=device) # 随机时间步
text_embeddings = torch.randn(1, 77, 768, dtype=torch.float16).to(device)  # 文本embedding之后的数据

with torch.no_grad():
    noise_pred = unet(
        noisy_latents,
        timestep,
        encoder_hidden_states=text_embeddings
    ).sample  # 注意要访问.sample属性

print("noise_pred值的形状:", noise_pred.shape)  # 输出: torch.Size([1, 4, 64, 64])
print("noise_pred值的类型", noise_pred.dtype)  # 输出: torch.float16
print("noise_pred部分数据查看:", noise_pred[0, 0, 0, :5])  # 打印部分数值

print("unet结构:", unet)


# 运行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part2/demo4_unet.py
# noise_pred值的形状: torch.Size([1, 4, 64, 64])
# noise_pred值的类型 torch.float16
# noise_pred部分数据查看: tensor([ 0.1525, -0.8413,  1.3018, -0.3459, -0.5659], device='cuda:0',
#        dtype=torch.float16)
# unet结构: UNet2DConditionModel(
#   (conv_in): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (time_proj): Timesteps()
#   (time_embedding): TimestepEmbedding(
#     (linear_1): Linear(in_features=320, out_features=1280, bias=True)
#     (act): SiLU()
#     (linear_2): Linear(in_features=1280, out_features=1280, bias=True)
#   )
#   (down_blocks): ModuleList(
#     (0): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (2): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (3): DownBlock2D(
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#     )
#   )
#   (up_blocks): ModuleList(
#     (0): UpBlock2D(
#       (resnets): ModuleList(
#         (0-2): 3 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
#           (conv1): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (2): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
#           (conv1): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
#           (conv1): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (3): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
#           (conv1): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1-2): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#     )
#   )
#   (mid_block): UNetMidBlock2DCrossAttn(
#     (attentions): ModuleList(
#       (0): Transformer2DModel(
#         (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#         (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         (transformer_blocks): ModuleList(
#           (0): BasicTransformerBlock(
#             (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (attn1): Attention(
#               (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=1280, out_features=1280, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (attn2): Attention(
#               (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_k): Linear(in_features=768, out_features=1280, bias=False)
#               (to_v): Linear(in_features=768, out_features=1280, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=1280, out_features=1280, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (ff): FeedForward(
#               (net): ModuleList(
#                 (0): GEGLU(
#                   (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                 )
#                 (1): Dropout(p=0.0, inplace=False)
#                 (2): Linear(in_features=5120, out_features=1280, bias=True)
#               )
#             )
#           )
#         )
#         (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#       )
#     )
#     (resnets): ModuleList(
#       (0-1): 2 x ResnetBlock2D(
#         (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#         (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#         (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (nonlinearity): SiLU()
#       )
#     )
#   )
#   (conv_norm_out): GroupNorm(32, 320, eps=1e-05, affine=True)
#   (conv_act): SiLU()
#   (conv_out): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# )
# PS D:\Project\Course\DiffusersCourse> 




































