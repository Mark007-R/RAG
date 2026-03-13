from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------
# 1) Setup Flask + Environment
# ---------------------------
app = Flask(__name__)
load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------------------------
# 2) Scrape + Save Content
# ---------------------------
# url = "https://en.wikipedia.org/wiki/Machine_learning"
# response = requests.get(url)
# soup = BeautifulSoup(response.text, "html.parser")
# content = "\n".join([p.get_text() for p in soup.select("p")])

# with open("b.txt", "w", encoding="utf-8") as f:
#     f.write(content)

url = "https://en.wikipedia.org/wiki/Machine_learning"
headers = {
    "User-Agent": (
        "Chrome/140.0.0.0 Safari/537.36"
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
    )
}

response = requests.get(url, headers=headers)

if response.status_code != 200:
    raise ValueError(f"Failed to fetch page: {response.status_code}")

soup = BeautifulSoup(response.text, "html.parser")
content = "\n".join([p.get_text() for p in soup.select("p")])

if not content.strip():
    raise ValueError("⚠️ Wikipedia content scrape failed — no <p> tags found!")

with open("b.txt", "w", encoding="utf-8") as f:
    f.write(content)
# ---------------------------
# 3) Load, Split, Embed
# ---------------------------
loader = TextLoader("b.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=200,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ---------------------------
# 4) RAG Chain Setup
# ---------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

template = """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know, say you don't know. Keep the answer concise (<= 10 sentences).

Question: {question}
Context:
{context}

Answer:"""

# prompt = ChatPromptTemplate.from_template(template)
# llm = ChatOllama(model="gemma3:4b")   # Ensure Ollama server is running & model pulled

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Create a local Hugging Face pipeline
generator = pipeline("text-generation", model="distilgpt2")  
llm = HuggingFacePipeline(pipeline=generator)


rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------------
# 5) Flask Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            answer = rag_chain.invoke(question)
    return render_template("index.html", answer=answer)

# ---------------------------
# 6) Run Flask
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
