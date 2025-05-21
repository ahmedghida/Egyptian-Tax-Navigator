# arabic_qa_chatbot.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime

def initialize_vectorstore(persist_dir: str = os.path.join('data', "chroma_db")) -> Chroma:
    """
    Initializes and returns a Chroma vector store using multilingual-e5-base embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
    return vectorstore

vectorstore = initialize_vectorstore()
load_dotenv()

# Prompt definition
prompt = ChatPromptTemplate.from_messages([
    ("system", "\n".join([
        "You are an intelligent assistant that answers questions strictly in Arabic, including Egyptian Arabic when appropriate.",
        "Your answers must be based only on the provided retrieved context.",
        "If the context does not contain sufficient information, respond with 'لا أعلم' (I don't know).",
        "Do not make up information. Never respond in English, even if the question is in English."
    ])),
    ("user", "\n".join([
        "Question: {question}",
        "Retrieved context:\n{context}",
        "Based on the context above, answer the question in Arabic only."
    ]))
])

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.3)

# Data retrieval function
def retrive_data(question: str) -> dict:
    docs = vectorstore.similarity_search_with_score(question, k=5)
    return {
        'question': question,
        'context': "\n".join([doc[0].page_content for doc in docs])
    }

# Create the full chain
main_chain = RunnableLambda(retrive_data) | prompt | llm | StrOutputParser()

