from dashscope import Generation

def call_qwen3_model():
    response = Generation.call(
        model='qwen3-235b-a22b-instruct-2507',
        prompt='请简单介绍一下牛顿第三定律',
        # 可选参数
        top_p=0.8,
        top_k=50,
        temperature=0.7,
        max_tokens=500
    )

    if response.status_code == 200:
        print("回答内容:")
        print(response.output.text)
    else:
        print(f"请求失败: {response.code} - {response.message}")


if __name__ == '__main__':
    call_qwen3_model()