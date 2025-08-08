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
import json
from semantic_similarity_model import AdvancedSemanticSimilarityModel, create_and_train_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model instance
model = None

def load_model():
    """Load the semantic similarity model"""
    global model
    try:
        model_path = 'semantic_similarity_model.pkl'
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            model = AdvancedSemanticSimilarityModel.load_model(model_path)
        else:
            logger.info("Creating new model...")
            model = create_and_train_model('DataNeuron_DataScience_Task1/DataNeuron_Text_Similarity.csv')
            model.save_model(model_path)
        
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/similarity', methods=['POST'])
def compute_similarity():
    """
    Compute semantic similarity between two texts
    
    Expected request body:
    {
        "text1": "first text paragraph",
        "text2": "second text paragraph"
    }
    
    Returns:
    {
        "similarity_score": 0.75
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text1 = data.get('text1')
        text2 = data.get('text2')
        
        if not text1 or not text2:
            return jsonify({'error': 'Both text1 and text2 are required'}), 400
        
        if not isinstance(text1, str) or not isinstance(text2, str):
            return jsonify({'error': 'Both text1 and text2 must be strings'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        start_time = time.time()
        similarity_score = model.compute_similarity(text1, text2)
        computation_time = time.time() - start_time
        
        logger.info(f"Similarity computed: {similarity_score:.4f} in {computation_time:.3f}s")
        
        response = {
            'similarity_score': round(similarity_score, 4)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_similarity', methods=['POST'])
def batch_similarity():
    """
    Compute similarity for multiple text pairs
    
    Expected request body:
    {
        "pairs": [
            {"text1": "first text", "text2": "second text"},
            {"text1": "third text", "text2": "fourth text"}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'pairs' not in data:
            return jsonify({'error': 'No pairs data provided'}), 400
        
        pairs = data['pairs']
        
        if not isinstance(pairs, list):
            return jsonify({'error': 'pairs must be a list'}), 400
        
        results = []
        
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict) or 'text1' not in pair or 'text2' not in pair:
                return jsonify({'error': f'Invalid pair format at index {i}'}), 400
            
            text1 = pair['text1']
            text2 = pair['text2']
            
            if not isinstance(text1, str) or not isinstance(text2, str):
                return jsonify({'error': f'Texts must be strings at index {i}'}), 400
            
            similarity_score = model.compute_similarity(text1, text2)
            results.append({
                'text1': text1,
                'text2': text2,
                'similarity_score': round(similarity_score, 4)
            })
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logger.error(f"Error in batch similarity: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_name': model.model_name,
        'device': model.device,
        'is_fitted': model.is_fitted,
        'ensemble_models': len(model.ensemble_models),
        'loaded_at': datetime.now().isoformat()
    }
    
    return jsonify(info), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
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
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def create_app():
    """Application factory for deployment"""
    if not load_model():
        logger.error("Failed to load model")
        return None
    
    return app

if __name__ == '__main__':
    if load_model():
        port = int(os.environ.get('PORT', 5000))
        
        logger.info(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Failed to start server - model not loaded") 