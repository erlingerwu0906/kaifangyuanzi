import dashscope
from dashscope import Generation
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import os

# 设置DashScope API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

class DashScopeQwen:
    def __init__(self, model_name='qwen3-235b-a22b-instruct-2507'):
        self.model_name = model_name

    def __call__(self, prompt: str) -> str:
        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            top_p=0.8,
            temperature=0.3,
            max_tokens=800
        )

        if response.status_code == 200:
            return response.output.text
        else:
            return f"请求失败: {response.code} - {response.message}"

# 1. 加载和分割文档
loader = TextLoader("physics_lecture.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
texts = text_splitter.split_documents(documents)

# 2. 嵌入和向量存储 - 使用本地缓存模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_folder="C:/Users/鲍心妍/.cache/huggingface/hub",
    model_kwargs={'local_files_only': True}
)

vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 创建Prompt Template
prompt_template = """请根据以下上下文信息回答问题。
上下文：{context}
问题：{question}
请基于上下文提供准确、详细的回答："""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 4. 初始化模型
llm = DashScopeQwen()

# 5. 创建RAG流程
def rag_qa(question: str) -> str:
    # 检索相关文档
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 构建prompt
    final_prompt = PROMPT.format(context=context, question=question)

    # 调用模型
    answer = llm(final_prompt)
    return answer

# 6. 测试系统
if __name__ == '__main__':
    question = "狭义相对论和广义相对论有什么区别？"
    print(f"问题: {question}")
    print("\n回答:")
    result = rag_qa(question)
    print(result)