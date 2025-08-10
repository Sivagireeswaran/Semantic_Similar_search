"""
Semantic Text Similarity API Server
Part B: API Deployment

This module provides a Flask-based REST API for semantic text similarity
that can be deployed to cloud services like Render, Heroku, or AWS.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import time
from datetime import datetime

# Import the lazy-loading model class
from semantic_similarity_model import AdvancedSemanticSimilarityModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model reference
model = None


MODEL_PATH = 'semantic_similarity_model.pkl'
TRAIN_DATA_PATH = 'DataNeuron_DataScience_Task1/DataNeuron_Text_Similarity.csv'
load_model()

def load_model():
    """Load model from disk or create a new one without eager transformer downloads."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}...")
            model = AdvancedSemanticSimilarityModel.load_model(MODEL_PATH)
        else:
            logger.warning(f"No pre-trained model found at {MODEL_PATH}. Creating new instance (lazy loading).")
            model = AdvancedSemanticSimilarityModel()
            # Optionally: call model.train(TRAIN_DATA_PATH) here if you want runtime training
            model.save_model(MODEL_PATH)

        logger.info("Model initialized successfully (transformers will load lazily).")
        return True
    except Exception as e:
        logger.exception(f"Error initializing model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/similarity', methods=['POST'])
def compute_similarity():
    try:
        data = request.get_json(force=True)
        text1 = data.get('text1')
        text2 = data.get('text2')

        if not text1 or not text2:
            return jsonify({'error': 'Both text1 and text2 are required'}), 400

        start_time = time.time()
        similarity_score = model.compute_similarity(text1, text2)
        elapsed = time.time() - start_time

        logger.info(f"Computed similarity={similarity_score:.4f} in {elapsed:.3f}s")

        return jsonify({'similarity_score': round(similarity_score, 4)}), 200

    except Exception as e:
        logger.exception("Error computing similarity")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_similarity', methods=['POST'])
def batch_similarity():
    try:
        data = request.get_json(force=True)
        pairs = data.get('pairs', [])

        if not isinstance(pairs, list):
            return jsonify({'error': 'pairs must be a list'}), 400

        results = []
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                return jsonify({'error': f'Invalid pair at index {i}'}), 400

            text1 = pair.get('text1')
            text2 = pair.get('text2')

            if not isinstance(text1, str) or not isinstance(text2, str):
                return jsonify({'error': f'Texts must be strings at index {i}'}), 400

            score = model.compute_similarity(text1, text2)
            results.append({'text1': text1, 'text2': text2, 'similarity_score': round(score, 4)})

        return jsonify({'results': results}), 200

    except Exception as e:
        logger.exception("Error in batch similarity")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'model_name': model.model_name,
        'device': model.device,
        'is_fitted': getattr(model, 'is_fitted', False),
        'ensemble_models': len(model.ensemble_models),
        'loaded_at': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Semantic Text Similarity API',
        'version': '1.0.0',
        'endpoints': {
            'POST /similarity': 'Compute similarity between two texts',
            'POST /batch_similarity': 'Compute similarity for multiple text pairs',
            'GET /health': 'Health check',
            'GET /model_info': 'Model information'
        },
        'usage': {
            'request_format': {
                'text1': 'First text paragraph',
                'text2': 'Second text paragraph'
            },
            'response_format': {
                'similarity_score': 'Score between 0 and 1'
            }
        }
    }), 200
@app.errorhandler(404)
def not_found(_):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(_):
    return jsonify({'error': 'Internal server error'}), 500

def create_app():
    if not load_model():
        logger.error("Failed to load model")
        return None
    return app

if __name__ == '__main__':
    if load_model():
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port)
    else:
        logger.error("Model failed to load; server not started.")
