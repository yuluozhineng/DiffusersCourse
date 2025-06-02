import torch
from transformers import CLIPTextModel, CLIPTokenizer

# 第一部分，参数配置
device = "cuda" if torch.cuda.is_available() else "cpu" # 优先使用GPU
model_id = "runwayml/stable-diffusion-v1-5" # 模型ID

# 第二部分，获取tokenizer与encoder
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer") # 分词器
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device) # 文本编码器
text_encoder.eval()

# 第三部分，输出tokenizer后的数据
print("tokenizer支持的最大长度:", tokenizer.model_max_length)
text_inputs = tokenizer(
    "A astronaut riding a horse on Mars",
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
print("text_inputs的key:", text_inputs.keys())
print("text_inputs:", text_inputs)

# 第四部分，输出解码后的数据，并观察
decoded_text = tokenizer.decode(text_inputs.input_ids[0])
print("解码后的数据:", decoded_text)

# 第五部分，输出词嵌入之后的形状
text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
print("text_embeddings的形状:", text_embeddings.shape)



# 运行时控制台输出:
# PS D:\Project\Course\DiffusersCourse> & D:/Software/Anaconda/envs/cv1/python.exe d:/Project/Course/DiffusersCourse/Section1/src/part2/demo1_clip_text.py   
# tokenizer支持的最大长度: 77
# text_inputs的key: dict_keys(['input_ids', 'attention_mask'])
# text_inputs: {'input_ids': tensor([[49406,   320, 18376,  6765,   320,  4558,   525,  7496, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#          49407, 49407, 49407, 49407, 49407, 49407, 49407]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0]])}
# 解码后的数据: <|startoftext|>a astronaut riding a horse on mars <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
# text_embeddings的形状: torch.Size([1, 77, 768])
# PS D:\Project\Course\DiffusersCourse>







