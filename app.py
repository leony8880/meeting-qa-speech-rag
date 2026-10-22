from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma
loader=DirectoryLoader("/home/tracxn-lp-752/Downloads/projects/Speech-analysis/transcripts",loader_cls=TextLoader)
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=300)
chunks=text_splitter.split_documents(docs)
embeddingg=OllamaEmbeddings(model="embeddinggemma-2")
vector=Chroma.from_documents(embedding=embeddingg,documents=chunks)
llm=ChatOllama(model="gemma:7b",temperature=0)
qa_chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=vector.as_retriever(search_kwargs={"k":4}))

app=FastAPI()

class Question (BaseModel):
    question:str
class Answer (BaseModel):
    answer:str

@app.post("/ask",response_model=Answer)
def ask(q:Question):
    response=qa_chain.invoke({"query":q.question})
    return Answer (answer=response["result"])
