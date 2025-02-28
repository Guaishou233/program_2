# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "deepseek-ai/Janus-Pro-7B"  # 选择模型大小，例如 opt-125m、opt-350m、opt-1.3b、opt-6.7b 等

# # 加载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # 保存到本地
# save_directory = "/data/hangyu/models/Janus-Pro-7B"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)

# print(f"Model and tokenizer saved to {save_directory}")  


from transformers import AutoTokenizer, AutoModelForCausalLM

model_names = ["facebook/opt-13b", "facebook/opt-30b"]

for model_name in model_names:
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 保存到本地（可选）
    save_path = f"/data/models/{model_name.split('/')[-1]}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model {model_name} 已保存到 {save_path}")
