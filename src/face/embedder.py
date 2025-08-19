"""
Face embedding generation module.
"""

import logging
import numpy as np
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class FaceEmbedder:
    """Face embedder class for generating and processing embeddings."""
    
    def __init__(self):
        """Initialize face embedder."""
        logger.info("Initialized face embedder")
    
    def get_embedding(self, face):
        """
        Get embedding from face.
        
        Args:
            face: Face object from InsightFace
        
        Returns:
            Normalized embedding vector
        """
        if face is None:
            logger.warning("Empty face provided to get_embedding")
            return None
        
        try:
            # Get embedding
            embedding = face.embedding
            
            # Normalize embedding
            normalized_embedding = self.normalize_embedding(embedding)
            
            return normalized_embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def normalize_embedding(self, embedding):
        """
        Normalize embedding vector.
        
        Args:
            embedding: Face embedding vector
        
        Returns:
            Normalized embedding vector
        """
        if embedding is None:
            return None
        
        try:
            # Reshape and normalize
            embedding_reshaped = embedding.reshape(1, -1)
            normalized = normalize(embedding_reshaped)[0]
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing embedding: {e}")
            return None
    
    def average_embeddings(self, embeddings):
        """
        Calculate average embedding from multiple embeddings.
        
        Args:
            embeddings: List of embedding vectors
        
        Returns:
            Average normalized embedding
        """
        if not embeddings:
            logger.warning("Empty embeddings list provided to average_embeddings")
            return None
        
        try:
            # Stack embeddings
            embeddings_array = np.vstack(embeddings)
            
            # Calculate average
            avg_embedding = np.mean(embeddings_array, axis=0)
            
            # Normalize
            return self.normalize_embedding(avg_embedding)
        except Exception as e:
            logger.error(f"Error averaging embeddings: {e}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0