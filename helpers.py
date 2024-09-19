import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class MyModel(BaseModel):
    name: str
    age: int

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

LLM_CONFIG = {
    'model': 'gpt-4o-mini',
    'api_key': OPENAI_API_KEY
}

# Pinecone
vectorstore = PineconeVectorStore(
    index_name='langchain-fastapi', embedding=embeddings
)

index_name = 'langchain-fastapi'

namespace = 'Search Wikipedia'

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to your pinecone index
pinecone_index = pc.Index('langchain-fastapi')

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 2}
)

retriever.invoke("What is the city named after trees")

retriever

llm = ChatOpenAI(**LLM_CONFIG)

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([('human', message)])


from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=lambda x: x['num'] * 123121,
)

runnable.invoke({'num': 31})


from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()


# chain = {'context': retriever, 'question': RunnablePassthrough()} | prompt | llm | parser


# chain.invoke('what type of food is best known in Austin Texas')


def get_chain():
    return {'context': retriever, 'question': RunnablePassthrough()} | prompt | llm | parser