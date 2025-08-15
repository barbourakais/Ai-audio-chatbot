"""
Vector Database Manager for semantic search
"""

import json
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from config import get_config
from utils import colored_print, setup_logging

logger = setup_logging()

class VectorDBManager:
    """Manages vector database operations for semantic search"""
    
    def __init__(self, db_path: str = "chroma_db"):
        self.db_path = db_path
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.chunk_size = 512  # Maximum tokens per chunk
        self.chunk_overlap = 50  # Overlap between chunks
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_database()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        try:
            colored_print("Loading embedding model...", "cyan")
            # Use a lightweight but effective model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            colored_print("✓ Embedding model loaded", "green")
        except Exception as e:
            colored_print(f"✗ Error loading embedding model: {e}", "red")
            raise
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection"""
        try:
            colored_print("Initializing vector database...", "cyan")
            
            # Create database directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            collection_name = "ox4labs_knowledge"
            try:
                self.collection = self.client.get_collection(collection_name)
                colored_print(f"✓ Loaded existing collection: {collection_name}", "green")
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Ox4labs company knowledge base"}
                )
                colored_print(f"✓ Created new collection: {collection_name}", "green")
            
        except Exception as e:
            colored_print(f"✗ Error initializing database: {e}", "red")
            raise
    
    def chunk_text(self, text: str, max_chunk_size: int = None) -> List[str]:
        """Split text into semantically meaningful chunks"""
        if max_chunk_size is None:
            max_chunk_size = self.chunk_size
        
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Clean the sentence
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add period if missing
            if not sentence.endswith('.'):
                sentence += '.'
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def parse_json_content(self, json_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse JSON content into structured chunks with metadata"""
        chunks = []
        
        def add_chunk(text: str, metadata: Dict[str, Any]):
            """Helper function to add a chunk with metadata"""
            if text.strip():
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id": chunk_id,
                    "text": text.strip(),
                    "metadata": metadata
                })
        
        # Company information
        if "company" in json_content:
            company = json_content["company"]
            company_text = f"Company name: {company.get('name', '')}. Tagline: {company.get('tagline', '')}. Location: {company.get('location', '')}. Description: {company.get('description', '')}."
            add_chunk(company_text, {"section": "company", "type": "company_info"})
        
        # Services
        if "services" in json_content:
            services = json_content["services"]
            for service_key, service_data in services.items():
                service_text = f"Service: {service_data.get('title', '')}. Description: {service_data.get('description', '')}."
                add_chunk(service_text, {"section": "services", "type": "service", "service_key": service_key})
                
                # Add offerings as separate chunks
                offerings = service_data.get('offerings', [])
                if offerings:
                    offerings_text = f"Offerings for {service_data.get('title', '')}: {', '.join(offerings)}."
                    add_chunk(offerings_text, {"section": "services", "type": "offerings", "service_key": service_key})
        
        # Process
        if "process" in json_content:
            process = json_content["process"]
            for step_key, step_data in process.items():
                step_text = f"Process step: {step_data.get('title', '')}. Description: {step_data.get('description', '')}."
                add_chunk(step_text, {"section": "process", "type": "process_step", "step_key": step_key})
        
        # Contact information
        if "contact" in json_content:
            contact = json_content["contact"]
            contact_text = f"Contact information: Website: {contact.get('website', '')}, Email: {contact.get('email', '')}, Phone: {contact.get('phone', '')}, Location: {contact.get('location', '')}."
            add_chunk(contact_text, {"section": "contact", "type": "contact_info"})
        
        return chunks
    
    def embed_and_store(self, json_file_path: str = "Ox4labs.json") -> int:
        """Load JSON content, chunk it, embed it, and store in vector database"""
        try:
            colored_print("Loading JSON content...", "cyan")
            
            # Load JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
            
            # Parse content into chunks
            colored_print("Parsing content into chunks...", "cyan")
            chunks = self.parse_json_content(json_content)
            
            if not chunks:
                colored_print("No content chunks found", "yellow")
                return 0
            
            colored_print(f"Created {len(chunks)} content chunks", "green")
            
            # Clear existing collection
            try:
                # Get all existing IDs and delete them
                existing_data = self.collection.get()
                if existing_data and existing_data['ids']:
                    self.collection.delete(ids=existing_data['ids'])
            except Exception as e:
                # If collection is empty or doesn't exist, continue
                pass
            
            # Prepare data for embedding
            texts = [chunk["text"] for chunk in chunks]
            ids = [chunk["id"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Generate embeddings
            colored_print("Generating embeddings...", "cyan")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Store in database
            colored_print("Storing in vector database...", "cyan")
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            colored_print(f"✓ Successfully stored {len(chunks)} chunks in vector database", "green")
            return len(chunks)
            
        except Exception as e:
            colored_print(f"✗ Error embedding and storing content: {e}", "red")
            raise
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant content using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in database
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    formatted_results.append({
                        "text": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """Get relevant context for a query"""
        results = self.search(query, top_k)
        
        if not results:
            return ""
        
        # Combine relevant chunks into context
        context_parts = []
        for result in results:
            context_parts.append(result["text"])
        
        return " ".join(context_parts)
    
    def update_knowledge_base(self, json_file_path: str = "Ox4labs.json") -> bool:
        """Update the knowledge base with new content"""
        try:
            colored_print("Updating knowledge base...", "cyan")
            num_chunks = self.embed_and_store(json_file_path)
            colored_print(f"✓ Knowledge base updated with {num_chunks} chunks", "green")
            return True
        except Exception as e:
            colored_print(f"✗ Error updating knowledge base: {e}", "red")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name,
                "embedding_model": self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else None
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    def reset_database(self) -> bool:
        """Reset the vector database"""
        try:
            colored_print("Resetting vector database...", "cyan")
            self.collection.delete(where={})
            colored_print("✓ Vector database reset", "green")
            return True
        except Exception as e:
            colored_print(f"✗ Error resetting database: {e}", "red")
            return False 