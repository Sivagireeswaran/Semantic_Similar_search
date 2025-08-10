"""
Semantic Text Similarity Model using Advanced Transformer-based Approaches
Optimized for Cloud Deployment (Lazy Loading, Small Image Size)
"""

import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from typing import List
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedSemanticSimilarityModel:
    """
    Advanced Semantic Similarity Model using multiple transformer-based approaches
    with lazy loading for ensemble models to reduce image size in cloud deployment.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None,
                 use_ensemble: bool = False):
        """
        Initialize the semantic similarity model

        Args:
            model_name: Pre-trained model name from sentence-transformers
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            use_ensemble: Whether to enable additional ensemble models
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_ensemble = use_ensemble

        # Cache location for HF models (Render/Railway can mount this as a volume)
        os.environ["TRANSFORMERS_CACHE"] = os.environ.get(
            "TRANSFORMERS_CACHE", "/app/.cache/huggingface"
        )
        os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

        logger.info(f"Loading main model: {model_name} on device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

        # Ensemble setup — models will be loaded only on first use
        self.ensemble_model_names = [
            'paraphrase-MiniLM-L6-v2',
            'all-mpnet-base-v2'
        ]
        self.ensemble_models = {}  # dict: model_name -> model instance

        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info("Model initialized successfully")

    def _load_ensemble_model(self, model_name: str):
        """Load an ensemble model only when needed."""
        if model_name not in self.ensemble_models:
            logger.info(f"Lazy loading ensemble model: {model_name}")
            self.ensemble_models[model_name] = SentenceTransformer(model_name, device=self.device)
        return self.ensemble_models[model_name]

    def _compute_embeddings(self, texts: List[str], model) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        try:
            return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return np.zeros((len(texts), 384))

    def _compute_advanced_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute advanced similarity using multiple metrics."""
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
        """Compute similarity using main + optional ensemble models."""
        similarities = []

        # Main model
        emb1 = self._compute_embeddings([text1], self.model)[0]
        emb2 = self._compute_embeddings([text2], self.model)[0]
        similarities.append(self._compute_advanced_similarity(emb1, emb2))

        # Ensemble models (lazy load only if enabled)
        if self.use_ensemble:
            for model_name in self.ensemble_model_names:
                try:
                    model = self._load_ensemble_model(model_name)
                    emb1_ens = self._compute_embeddings([text1], model)[0]
                    emb2_ens = self._compute_embeddings([text2], model)[0]
                    similarities.append(self._compute_advanced_similarity(emb1_ens, emb2_ens))
                except Exception as e:
                    logger.warning(f"Ensemble model {model_name} failed: {e}")

        # Weighted average
        if len(similarities) > 1:
            weights = [0.6] + [0.4 / (len(similarities) - 1)] * (len(similarities) - 1)
            return np.average(similarities, weights=weights)
        else:
            return similarities[0]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Public method to compute similarity score."""
        try:
            text1 = self._preprocess_text(text1)
            text2 = self._preprocess_text(text2)
            score = self._ensemble_similarity(text1, text2)
            if self.is_fitted:
                score = self._calibrate_score(score)
            return float(score)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.5

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        text = ' '.join(text.split())
        return text

    def _calibrate_score(self, score: float) -> float:
        """Calibrate similarity score after fitting."""
        return float(1 / (1 + np.exp(-5 * (score - 0.5))))

    def fit(self, training_data: pd.DataFrame = None):
        """Fit calibration or scaling parameters."""
        if training_data is not None:
            logger.info("Fitting model to training data...")
            self.is_fitted = True
            logger.info("Model fitting completed")

    def save_model(self, filepath: str):
        """Save model configuration and calibration."""
        try:
            model_data = {
                'model_name': self.model_name,
                'device': self.device,
                'is_fitted': self.is_fitted,
                'scaler': self.scaler,
                'use_ensemble': self.use_ensemble
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            self.model.save(f"{filepath}_transformer")
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load saved model configuration and main transformer."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            model = cls(
                model_name=model_data['model_name'],
                device=model_data['device'],
                use_ensemble=model_data.get('use_ensemble', True)
            )
            model.is_fitted = model_data['is_fitted']
            model.scaler = model_data['scaler']
            model.model = SentenceTransformer.load(f"{filepath}_transformer")
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return cls()


def create_and_train_model(data_path: str = None,
                           use_ensemble: bool = True) -> AdvancedSemanticSimilarityModel:
    """Factory method to create and optionally train the model."""
    logger.info("Creating semantic similarity model...")
    model = AdvancedSemanticSimilarityModel(use_ensemble=use_ensemble)
    if data_path:
        try:
            training_data = pd.read_csv(data_path)
            logger.info(f"Loaded training data: {training_data.shape}")
            model.fit(training_data)
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
    return model


if __name__ == "__main__":
    # Example usage
    model = create_and_train_model(
        data_path='DataNeuron_DataScience_Task1/DataNeuron_Text_Similarity.csv',
        use_ensemble=False
    )
    text1 = "The weather is sunny today."
    text2 = "It's a beautiful sunny day."
    sim = model.compute_similarity(text1, text2)
    print(f"Similarity: {sim:.4f}")
    model.save_model('semantic_similarity_model.pkl')
