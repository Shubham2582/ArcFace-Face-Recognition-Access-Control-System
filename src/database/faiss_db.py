"""
FAISS-based face embeddings database for fast similarity search.
This is optimized for large-scale databases.
"""

import logging
import os
import numpy as np
import pickle
import faiss
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)

class FaissDatabase:
    """FAISS-based database for efficient face embedding search."""
    
    def __init__(self, 
                 db_path=settings.FAISS_DB_PATH,
                 labels_path=settings.FAISS_LABELS_PATH):
        """
        Initialize FAISS database.
        
        Args:
            db_path: Path to store/load the FAISS index
            labels_path: Path to store/load the labels
        """
        self.db_path = db_path
        self.labels_path = labels_path
        self.dimension = 512  # ArcFace embedding dimension
        self.index = None
        self.identities = []
        self.metadata = {}
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize index
        self._initialize_index()
        
        # Load database if it exists
        if os.path.exists(db_path) and os.path.exists(labels_path):
            self.load_database()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        try:
            # Create an inner product index (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Initialized FAISS index")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise
    
    def load_database(self):
        """Load FAISS index and labels from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.db_path)
            
            # Load labels and metadata
            with open(self.labels_path, 'rb') as f:
                data = pickle.load(f)
                self.identities = data['identities']
                self.metadata = data['metadata']
            
            logger.info(f"Loaded FAISS database with {len(self.identities)} identities")
        except Exception as e:
            logger.error(f"Error loading FAISS database: {e}")
            self._initialize_index()
            self.identities = []
            self.metadata = {}
    
    def save_database(self):
        """Save FAISS index and labels to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.db_path)
            
            # Save labels and metadata
            with open(self.labels_path, 'wb') as f:
                data = {
                    'identities': self.identities,
                    'metadata': self.metadata
                }
                pickle.dump(data, f)
            
            logger.info(f"Saved FAISS database with {len(self.identities)} identities")
        except Exception as e:
            logger.error(f"Error saving FAISS database: {e}")
    
    def add_identity(self, identity_name, embedding, num_images=1):
        """
        Add identity to database.
        
        Args:
            identity_name: Name of the person
            embedding: Face embedding vector
            num_images: Number of images used to create the embedding
        """
        if identity_name is None or embedding is None:
            logger.warning("Cannot add identity with empty name or embedding")
            return
        
        try:
            # Add embedding to FAISS index
            self.index.add(np.array([embedding]).astype('float32'))
            
            # Add identity to list
            self.identities.append(identity_name)
            
            # Add metadata
            self.metadata[identity_name] = {
                'num_images': num_images,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'index': len(self.identities) - 1
            }
            
            logger.info(f"Added identity to FAISS: {identity_name} with {num_images} images")
        except Exception as e:
            logger.error(f"Error adding identity to FAISS: {e}")
    
    def update_identity(self, identity_name, embedding, num_images=1):
        """
        Update identity in database.
        
        Note: FAISS doesn't support direct updates, so we need to remove and re-add.
        For simplicity in this implementation, we just add a new entry.
        A more efficient implementation would rebuild the index.
        
        Args:
            identity_name: Name of the person
            embedding: Face embedding vector
            num_images: Number of images used to create the embedding
        """
        # For simplicity, just add a new entry
        self.add_identity(identity_name, embedding, num_images)
        
        # Mark existing entries as outdated in metadata
        for name, data in self.metadata.items():
            if name == identity_name and data['index'] != len(self.identities) - 1:
                data['outdated'] = True
    
    def get_identity_metadata(self, identity_name):
        """
        Get identity metadata.
        
        Args:
            identity_name: Name of the person
        
        Returns:
            Metadata or None if not found
        """
        if identity_name not in self.metadata:
            return None
        
        return self.metadata[identity_name]
    
    def list_identities(self):
        """
        List all unique identities in database.
        
        Returns:
            List of identity names
        """
        return list(set(self.identities))
    
    def get_database_info(self):
        """
        Get database information.
        
        Returns:
            Dictionary with database information
        """
        return {
            'num_identities': len(set(self.identities)),
            'total_embeddings': len(self.identities),
            'identities': self.list_identities(),
            'db_path': self.db_path
        }
    
    def find_match(self, embedding, threshold=settings.RECOGNITION_THRESHOLD, k=5):
        """
        Find matching identity for embedding.
        
        Args:
            embedding: Face embedding vector
            threshold: Similarity threshold for recognition
            k: Number of nearest neighbors to search
        
        Returns:
            (identity_name, similarity) if match found, (None, 0) otherwise
        """
        if embedding is None or self.index.ntotal == 0:
            return None, 0
        
        try:
            # Search k nearest neighbors
            embedding_array = np.array([embedding]).astype('float32')
            scores, indices = self.index.search(embedding_array, k)
            
            # Get best matching identity
            best_score = scores[0][0]
            best_index = indices[0][0]
            
            # Check if score is above threshold
            if best_score >= threshold and 0 <= best_index < len(self.identities):
                best_match = self.identities[best_index]
                return best_match, float(best_score)
            else:
                return None, 0
        except Exception as e:
            logger.error(f"Error finding match in FAISS: {e}")
            return None, 0
    
    def get_all_similarity_scores(self, embedding, max_results=10):
        """
        Get similarity scores for top matches.
        
        Args:
            embedding: Face embedding vector
            max_results: Maximum number of results to return
        
        Returns:
            Dictionary of identity names to similarity scores
        """
        if embedding is None or self.index.ntotal == 0:
            return {}
        
        try:
            # Search max_results nearest neighbors
            embedding_array = np.array([embedding]).astype('float32')
            k = min(max_results, self.index.ntotal)
            scores, indices = self.index.search(embedding_array, k)
            
            # Create dictionary of scores
            results = {}
            for i in range(k):
                score = scores[0][i]
                idx = indices[0][i]
                
                if idx >= 0 and idx < len(self.identities):
                    identity = self.identities[idx]
                    
                    # Keep highest score for each identity
                    if identity not in results or score > results[identity]:
                        results[identity] = float(score)
            
            return results
        except Exception as e:
            logger.error(f"Error getting similarity scores from FAISS: {e}")
            return {}