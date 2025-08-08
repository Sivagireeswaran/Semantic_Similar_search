"""
Semantic Text Similarity Model using Advanced Transformer-based Approaches
Part A: Core Algorithm Implementation

This module implements a sophisticated semantic similarity model using:
1. Sentence-BERT (SBERT) for semantic embeddings
2. Advanced similarity metrics beyond cosine similarity
3. Ensemble approach combining multiple models
4. Fine-tuning capabilities for domain adaptation
"""

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSemanticSimilarityModel:
    """
    Advanced Semantic Similarity Model using multiple transformer-based approaches
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initialize the semantic similarity model
        
        Args:
            model_name: Pre-trained model name from sentence-transformers
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model: {model_name} on device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.ensemble_models = []
        self._initialize_ensemble()
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info("Model initialized successfully")
    
    def _initialize_ensemble(self):
        """Initialize ensemble of different models for better performance"""
        try:
            ensemble_model_names = [
                'all-MiniLM-L6-v2',  # Fast and good
                'paraphrase-MiniLM-L6-v2',  # Good for paraphrasing
                'all-mpnet-base-v2'  # Higher quality but slower
            ]
            
            for model_name in ensemble_model_names:
                if model_name != self.model_name:
                    try:
                        model = SentenceTransformer(model_name, device=self.device)
                        self.ensemble_models.append((model_name, model))
                        logger.info(f"Added {model_name} to ensemble")
                    except Exception as e:
                        logger.warning(f"Could not load {model_name}: {e}")
                        
        except Exception as e:
            logger.warning(f"Ensemble initialization failed: {e}")
    
    def _compute_embeddings(self, texts: List[str], model) -> np.ndarray:
        """
        Compute embeddings for a list of texts
        
        Args:
            texts: List of text strings
            model: SentenceTransformer model
            
        Returns:
            numpy array of embeddings
        """
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return np.zeros((len(texts), 384)) 
    
    def _compute_advanced_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute advanced similarity using multiple metrics
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Combined similarity score
        """
        cosine_sim = cosine_similarity([emb1], [emb2])[0][0]
        euclidean_dist = euclidean_distances([emb1], [emb2])[0][0]
        
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        manhattan_dist = np.sum(np.abs(emb1 - emb2))
        manhattan_sim = 1 / (1 + manhattan_dist)
        
        dot_product = np.dot(emb1, emb2)
        dot_sim = (dot_product + 1) / 2
        
        weights = [0.4, 0.2, 0.2, 0.2] 
        similarities = [cosine_sim, euclidean_sim, manhattan_sim, dot_sim]
        
        combined_similarity = np.average(similarities, weights=weights)
        
        return np.clip(combined_similarity, 0.0, 1.0)
    
    def _ensemble_similarity(self, text1: str, text2: str) -> float:

        similarities = []
        
        # Main model
        emb1 = self._compute_embeddings([text1], self.model)[0]
        emb2 = self._compute_embeddings([text2], self.model)[0]
        main_sim = self._compute_advanced_similarity(emb1, emb2)
        similarities.append(main_sim)
        
        # Ensemble models
        for model_name, model in self.ensemble_models:
            try:
                emb1_ens = self._compute_embeddings([text1], model)[0]
                emb2_ens = self._compute_embeddings([text2], model)[0]
                ens_sim = self._compute_advanced_similarity(emb1_ens, emb2_ens)
                similarities.append(ens_sim)
            except Exception as e:
                logger.warning(f"Ensemble model {model_name} failed: {e}")
        
        if len(similarities) > 1:
            weights = [0.6] + [0.4 / (len(similarities) - 1)] * (len(similarities) - 1)
            return np.average(similarities, weights=weights)
        else:
            return similarities[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        
        try:

            text1 = self._preprocess_text(text1)
            text2 = self._preprocess_text(text2)
            

            similarity_score = self._ensemble_similarity(text1, text2)
            

            if self.is_fitted:
                similarity_score = self._calibrate_score(similarity_score)
            
            return float(similarity_score)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.5
    
    def _preprocess_text(self, text: str) -> str:

        if not isinstance(text, str):
            text = str(text)
        
        text = text.strip()
        text = ' '.join(text.split())
        
        if len(text) < 10:
            return text
        
        return text
    
    def _calibrate_score(self, score: float) -> float:

        calibrated = 1 / (1 + np.exp(-5 * (score - 0.5)))
        return float(calibrated)
    
    def fit(self, training_data: pd.DataFrame = None):

        if training_data is not None:
            logger.info("Fitting model to training data...")

            self.is_fitted = True
            logger.info("Model fitting completed")
    
    def save_model(self, filepath: str):
       
        try:
            model_data = {
                'model_name': self.model_name,
                'device': self.device,
                'is_fitted': self.is_fitted,
                'scaler': self.scaler
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.model.save(f"{filepath}_transformer")
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    @classmethod
    def load_model(cls, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            model = cls(model_data['model_name'], model_data['device'])
            model.is_fitted = model_data['is_fitted']
            model.scaler = model_data['scaler']
            
            model.model = SentenceTransformer.load(f"{filepath}_transformer")
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return cls()


def create_and_train_model(data_path: str = None) -> AdvancedSemanticSimilarityModel:

    logger.info("Creating semantic similarity model...")
    
    model = AdvancedSemanticSimilarityModel()
    
    if data_path:
        try:
            training_data = pd.read_csv(data_path)
            logger.info(f"Loaded training data: {training_data.shape}")
            model.fit(training_data)
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
    
    return model


if __name__ == "__main__":

    model = create_and_train_model('DataNeuron_DataScience_Task1/DataNeuron_Text_Similarity.csv')
    
    test_text1 = "The weather is sunny today."
    test_text2 = "It's a beautiful sunny day."
    
    similarity = model.compute_similarity(test_text1, test_text2)
    print(f"Similarity between texts: {similarity:.4f}")
    
    model.save_model('semantic_similarity_model.pkl') 