"""
Face embeddings database operations.
"""

import logging
import os
import pickle
from datetime import datetime
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

class EmbeddingsDatabase:
    """Database for storing and retrieving face embeddings."""
    
    def __init__(self, db_path=settings.DB_PATH):
        """
        Initialize embeddings database.
        
        Args:
            db_path: Path to store/load the face embeddings database
        """
        self.db_path = db_path
        self.database = {}
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Load database if it exists
        if os.path.exists(db_path):
            self.load_database()
    
    def load_database(self):
        """Load face embeddings database from disk."""
        try:
            with open(self.db_path, 'rb') as f:
                self.database = pickle.load(f)
            logger.info(f"Loaded database with {len(self.database)} identities")
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.database = {}
    
    def save_database(self):
        """Save face embeddings database to disk."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.database, f)
            logger.info(f"Saved database with {len(self.database)} identities")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
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
        
        self.database[identity_name] = {
            'embedding': embedding,
            'num_images': num_images,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Added identity: {identity_name} with {num_images} images")
    
    def update_identity(self, identity_name, embedding, num_images=1):
        """
        Update identity in database.
        
        Args:
            identity_name: Name of the person
            embedding: Face embedding vector
            num_images: Number of images used to create the embedding
        """
        if identity_name not in self.database:
            self.add_identity(identity_name, embedding, num_images)
            return
        
        self.database[identity_name]['embedding'] = embedding
        self.database[identity_name]['num_images'] = num_images
        self.database[identity_name]['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Updated identity: {identity_name} with {num_images} images")
    
    def get_identity(self, identity_name):
        """
        Get identity data from database.
        
        Args:
            identity_name: Name of the person
        
        Returns:
            Identity data or None if not found
        """
        if identity_name not in self.database:
            return None
        
        return self.database[identity_name]
    
    def get_embedding(self, identity_name):
        """
        Get embedding for identity.
        
        Args:
            identity_name: Name of the person
        
        Returns:
            Embedding vector or None if not found
        """
        identity = self.get_identity(identity_name)
        if identity is None:
            return None
        
        return identity['embedding']
    
    def remove_identity(self, identity_name):
        """
        Remove identity from database.
        
        Args:
            identity_name: Name of the person
        
        Returns:
            True if removed, False otherwise
        """
        if identity_name not in self.database:
            logger.warning(f"Identity not found: {identity_name}")
            return False
        
        del self.database[identity_name]
        logger.info(f"Removed identity: {identity_name}")
        return True
    
    def list_identities(self):
        """
        List all identities in database.
        
        Returns:
            List of identity names
        """
        return list(self.database.keys())
    
    def get_database_info(self):
        """
        Get database information.
        
        Returns:
            Dictionary with database information
        """
        return {
            'num_identities': len(self.database),
            'identities': self.list_identities(),
            'db_path': self.db_path
        }
    
    def find_match(self, embedding, threshold=settings.RECOGNITION_THRESHOLD):
        """
        Find matching identity for embedding.
        
        Args:
            embedding: Face embedding vector
            threshold: Similarity threshold for recognition
        
        Returns:
            (identity_name, similarity) if match found, (None, 0) otherwise
        """
        if embedding is None:
            return None, 0
        
        best_match = None
        best_score = 0
        
        for identity_name, data in self.database.items():
            db_embedding = data['embedding']
            
            # Calculate cosine similarity
            similarity = np.dot(embedding, db_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_match = identity_name
        
        # Check if score is above threshold
        if best_score >= threshold:
            return best_match, float(best_score)
        else:
            return None, 0
    
    def get_all_similarity_scores(self, embedding):
        """
        Get similarity scores for all identities.
        
        Args:
            embedding: Face embedding vector
        
        Returns:
            Dictionary of identity names to similarity scores
        """
        if embedding is None:
            return {}
        
        scores = {}
        
        for identity_name, data in self.database.items():
            db_embedding = data['embedding']
            
            # Calculate cosine similarity
            similarity = np.dot(embedding, db_embedding)
            scores[identity_name] = float(similarity)
        
        return scores