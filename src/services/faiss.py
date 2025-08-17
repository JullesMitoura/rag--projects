import os
import uuid
from typing import List, Dict, Tuple, Union
import numpy as np
import faiss
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils import setup_logger

logger = setup_logger(__name__)


class FaissService:
    def __init__(self, 
                 embeddings, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """
            embeddings: Embedding
            chunk_size: Size of chunks for document splitting
            chunk_overlap: Overlap between chunks
        """
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self._temp_databases = {}  # Store temporary databases
        logger.info(f"FaissService initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def create_local_database(self, 
                              documents: List[Document], 
                              index_path: str) -> FAISS:
        try:
            logger.info(f"Starting local database creation at: {index_path}")
            
            # Process documents into chunks
            chunks = self._process_documents(documents)
            
            if not chunks:
                raise ValueError("No chunks were generated from the provided documents")
            
            # Create FAISS database
            vdb = FAISS.from_documents(chunks, self.embeddings)
            
            # Save locally
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            vdb.save_local(index_path)
            
            logger.info(f"Database created with {len(chunks)} chunks and saved at: {index_path}")
            return vdb
            
        except Exception as e:
            logger.error(f"Error creating local database: {str(e)}")
            raise
    
    def create_temporary_database(self, 
                                  documents: List[Document]) -> Tuple[str, FAISS]:
        try:
            logger.info("Starting temporary database creation")
            
            # Process documents into chunks
            chunks = self._process_documents(documents)
            
            if not chunks:
                raise ValueError("No chunks were generated from the provided documents")
            
            # Create FAISS database in memory
            vdb = FAISS.from_documents(chunks, self.embeddings)
            
            # Generate unique ID for temporary database
            temp_id = str(uuid.uuid4())
            self._temp_databases[temp_id] = vdb
            
            logger.info(f"Temporary database created with {len(chunks)} chunks. ID: {temp_id}")
            return temp_id, vdb
            
        except Exception as e:
            logger.error(f"Error creating temporary database: {str(e)}")
            raise
    
    def load_local_database(self, index_path: str) -> FAISS:
        try:
            logger.info(f"Loading database from: {index_path}")
            
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index not found at: {index_path}")
            
            vdb = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"Database successfully loaded from: {index_path}")
            return vdb
            
        except Exception as e:
            logger.error(f"Error loading local database: {str(e)}")
            raise
    
    def add_documents_to_database(self, 
                                  vdb: FAISS, 
                                  documents: List[Document]) -> FAISS:
        try:
            logger.info(f"Adding {len(documents)} documents to existing database")
            
            # Process new documents
            chunks = self._process_documents(documents)
            
            if not chunks:
                logger.warning("No chunks were generated from the new documents")
                return vdb
            
            # Add to existing database
            vdb.add_documents(chunks)
            
            logger.info(f"{len(chunks)} new chunks added to database")
            return vdb
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        vdb_or_id: Union[FAISS, str], 
        query: str, 
        k: int = 5,
        return_scores: bool = False
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        
        try:
            logger.debug(f"Performing similarity search: '{query[:100]}...' (k={k})")
            
            # Determine which database to use
            if isinstance(vdb_or_id, str):
                if vdb_or_id not in self._temp_databases:
                    raise ValueError(f"Temporary database not found: {vdb_or_id}")
                vdb = self._temp_databases[vdb_or_id]
                logger.debug(f"Using temporary database: {vdb_or_id}")
            else:
                vdb = vdb_or_id
                logger.debug("Using local/loaded database")
            
            # Perform search
            if return_scores:
                results = vdb.similarity_search_with_score(query, k=k)
                logger.info(f"Search completed: {len(results)} results with scores")
                return results
            else:
                results = vdb.similarity_search(query, k=k)
                logger.info(f"Search completed: {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def get_database_info(self, 
                          vdb_or_id: Union[FAISS, str]) -> dict:
        
        try:
            logger.debug("Getting database information")
            
            # Determine which database to use
            if isinstance(vdb_or_id, str):
                if vdb_or_id not in self._temp_databases:
                    raise ValueError(f"Temporary database not found: {vdb_or_id}")
                vdb = self._temp_databases[vdb_or_id]
                db_type = "Temporary"
                db_id = vdb_or_id
            else:
                vdb = vdb_or_id
                db_type = "Local/Loaded"
                db_id = "N/A"
            
            # Get index information
            total_vectors = vdb.index.ntotal
            dimension = vdb.index.d
            
            info = {
                "type": db_type,
                "id": db_id,
                "total_vectors": total_vectors,
                "dimension": dimension,
                "embedding_model": type(self.embeddings).__name__
            }
            
            logger.debug(f"Information retrieved: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting database information: {str(e)}")
            raise
    
    def _process_documents(self, 
                           documents: List[Document]) -> List[Document]:
        
        logger.debug(f"Processing {len(documents)} documents into chunks")
        
        chunks = []
        
        for i, doc in enumerate(documents):
            # Split document into chunks
            doc_chunks = self.text_splitter.split_documents([doc])
            
            # Add metadata to chunks
            for j, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "doc_id": i,
                    "chunk_id": j,
                    "source": doc.metadata.get("source", f"document_{i}"),
                    "total_chunks": len(doc_chunks)
                })
                chunks.append(chunk)
        
        logger.debug(f"Processing completed: {len(chunks)} chunks generated")
        return chunks

    @staticmethod
    def text_to_documents(text_data):
        documents = [
                Document(
                    page_content=doc["texts"],
                    metadata={"source": doc["document"]}
                )
                for doc in text_data.values()
            ]
        
        return documents