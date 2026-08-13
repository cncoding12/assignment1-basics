import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.generation import generate

# 1. 加载现有的 Tokenizer
output_dir = "results"
tokenizer = Tokenizer.from_files(
    f"{output_dir}/tinystories_vocab.json",
    f"{output_dir}/tinystories_merges.json",
    special_tokens=["<|endoftext|>"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 实例化模型并加载权重
model = TransformerLM(
    vocab_size=10000,
    context_length=256,
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024,
    device=device,
)
model.load_state_dict(torch.load("results/model_weights.pt", map_location=device))
model.to(device)

# 3. 输入 Prompt
prompt_text = "Once upon a time, there was a little girl named Lily."
print(f"📝 Prompt: {prompt_text}\n")

prompt_ids = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long, device=device)

# 获取 <|endoftext|> 的 Token ID (修改了这一行)
eos_id = tokenizer.encode("<|endoftext|>")[0]

# 4. 自回归生成故事
output_ids = generate(
    model=model,
    prompt_tokens=prompt_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    eos_token_id=eos_id,
)

# 5. 解码打印
story = tokenizer.decode(output_ids[0].tolist())
print("📖 Generated Story:\n" + "-"*40)
print(story)
print("-" * 40)