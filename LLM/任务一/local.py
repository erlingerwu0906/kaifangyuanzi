from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 设置模型路径（使用已下载的模型）
model_path = "D:/huggingface_models/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"

try:
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )

    # 准备提示词并进行推理
    prompt = "请简单介绍一下牛顿第三定律。"
    print(f"问题: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"回答: {response}")

except Exception as e:
    print(f"错误: {e}")