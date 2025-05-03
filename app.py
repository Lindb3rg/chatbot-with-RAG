import os
import glob
from dotenv import load_dotenv
import gradio as gr
from utils.tools import logger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vector_store import VectorStore
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from config.config import SYSTEM_PROMPT


load_dotenv()
MODEL = os.getenv("OPENAI_MODEL")
VECTOR_DB_NAME = os.getenv("VECTOR_DB_NAME", "faiss_index")
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
folders = glob.glob("knowledge-base/*")

embeddings = OpenAIEmbeddings()
vector_store = None

if os.path.exists(VECTOR_DB_NAME) and os.path.isdir(VECTOR_DB_NAME):
    try:
        logger.info(f"Loading existing vector store from {VECTOR_DB_NAME}...")
        vector_store = FAISS.load_local(VECTOR_DB_NAME, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Vector store loaded successfully with {len(vector_store.index_to_docstore_id)} documents")
    except Exception as e:
        logger.warning(f"Error loading existing vector store: {e}")
        logger.info("Creating new vector store...")
        vector_store = None


if vector_store is None:
    logger.info("Creating new vector store...")
    store = VectorStore(folders)
    store.prepare_documents()
    store.create_vector_store()
    vector_store = store.vector_store
    

    vector_store.save_local(VECTOR_DB_NAME)
    logger.info(f"Vector store saved to {VECTOR_DB_NAME}")

system_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}")
])


print(system_prompt)

llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vector_store.as_retriever()
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=memory,

)

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]


if __name__ == "__main__":
    view = gr.ChatInterface(chat).launch()