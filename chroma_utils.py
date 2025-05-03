from dotenv import load_dotenv
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from utils.tools import logger


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

DB_NAME = os.getenv("VECTOR_DB_NAME")




class PrepareDocumentsFolder:
    def __init__(self,folders):
        self.folders = folders
        self.text_loader_kwargs = {'encoding': 'utf-8'}
        self.documents = []
        self.doc_type = ""
        self.doc_types = []
        self.chunks = []
        self.folder_docs = []
        

    def create_chunks(self):
        
        if not self.folders:
            raise ValueError("No folders specified: please set path to folder with directories to be prepared.")
        
        for folder in self.folders:
            if not os.path.isdir(folder):
                raise ValueError(f"Folder path '{folder}' does not exist or is not a directory")
        
        self.documents = []
        for folder in self.folders:
            self.doc_type = os.path.basename(folder)
            
            try:
                loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=self.text_loader_kwargs)
                self.folder_docs = loader.load()
            except Exception as e:
                raise RuntimeError(f"Error loading documents from '{folder}': {e}") from e
            
            for doc in self.folder_docs:
                doc.metadata["doc_type"] = self.doc_type
                self.documents.append(doc)    

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = text_splitter.split_documents(self.documents)
        self.doc_types = set(chunk.metadata['doc_type'] for chunk in self.chunks)
        logger.info(f"Document types found: {', '.join(self.doc_types)}") 

        return self.chunks



class VectorStore:
    def __init__(self,chunks,db_name):
        self.db_name = db_name
        self.chunks = chunks
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._initialize_vectorstore()
        self.collection = self.vectorstore._collection
        self.dimensions = self._vectorstore_dimensions()
        self.collection_result = None
        self.vectors = []
        self.documents = []
        self.doc_types = []
        self._collect_embedding_result()
        self.colors = []
        self._set_document_colours()
        

    def _initialize_vectorstore(self):
        if not self.db_name:
            raise ValueError("Chroma DB not found: Please set name for vector db name.")
        
        if os.path.exists(self.db_name):
            temp_db = Chroma(persist_directory=self.db_name, embedding_function=self.embeddings)
            temp_db.delete_collection()
            logger.info(f"Deleted existing collection at {self.db_name}")
        
        vectorstore = Chroma.from_documents(
            documents=self.chunks, 
            embedding=self.embeddings, 
            persist_directory=self.db_name
        )
        
        logger.info(f"Vectorstore created with {vectorstore._collection.count()} documents")
        return vectorstore


    def _vectorstore_dimensions(self):    
        self.collection = self.vectorstore._collection
        sample_embedding = self.collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)
        logger.info(f"Vectorstore '{self.db_name}' has {dimensions:,} dimensions")
        return dimensions

    

    def _collect_embedding_result(self):
        self.collection_result = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        self.vectors = np.array(self.collection_result['embeddings'])
        self.documents = self.collection_result['documents']
        self.doc_types = [metadata['doc_type'] for metadata in self.collection_result['metadatas']]


        
    def _set_document_colours(self):
        available_colors = ['blue', 'green', 'red', 'purple', 'orange', 'teal','yellow','brown']
        color_map = {doc_type: available_colors[i % len(available_colors)] 
                    for i, doc_type in enumerate(set(self.doc_types))}
        
        self.colors = [color_map.get(t, 'gray') for t in self.doc_types]
    
    
    def show_vectors_2D(self,n_components=2,random_state=42,perplexity=None):
        if len(self.vectors) <= 1:
            logger.warning("Not enough vectors to perform dimensionality reduction")
            return
        
        if perplexity is None:
            perplexity = min(30, len(self.vectors) - 1)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            
        reduced_vectors = tsne.fit_transform(self.vectors)
        
        fig = go.Figure(data=[go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers',
            marker=dict(size=5, color=self.colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(self.doc_types, self.documents)],
            hoverinfo='text'
        )])

        fig.update_layout(
            title='2D Chroma Vector Store Visualization',
            scene=dict(xaxis_title='x',yaxis_title='y'),
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )

        fig.show()
    
    
        
    

