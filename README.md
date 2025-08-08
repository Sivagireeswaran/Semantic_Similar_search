# Semantic Text Similarity API

A sophisticated semantic text similarity model and REST API built using advanced transformer-based approaches. This project implements Part A (algorithm/model) and Part B (API deployment) of the DataNeuron assessment.

## Features

### Part A: Advanced Semantic Similarity Model
- **Sentence-BERT (SBERT)** for semantic embeddings
- **Ensemble approach** combining multiple transformer models
- **Advanced similarity metrics** beyond basic cosine similarity
- **Multiple distance measures**: Cosine, Euclidean, Manhattan, Dot Product
- **Text preprocessing** and normalization
- **Score calibration** for better distribution
- **Model persistence** and loading capabilities

### Part B: REST API Deployment
- **Flask-based REST API** with comprehensive endpoints
- **Docker containerization** for easy deployment
- **Cloud deployment ready** (Render, Heroku, AWS)
- **Health checks** and monitoring
- **Error handling** and validation
- **Batch processing** capabilities
- **CORS support** for web applications

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│  Flask API       │───▶│  SBERT Model    │
│                 │    │  (api_server.py) │    │  (Ensemble)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Similarity      │
                       │  Computation     │
                       └──────────────────┘
```

## API Endpoints

### Main Endpoint
- **POST** `/similarity` - Compute similarity between two texts

**Request:**
```json
{
    "text1": "nuclear body seeks new tech .......",
    "text2": "terror suspects face arrest ......"
}
```

**Response:**
```json
{
    "similarity_score": 0.2
}
```

### Additional Endpoints
- **GET** `/health` - Health check
- **GET** `/model_info` - Model information
- **GET** `/` - API documentation
- **POST** `/batch_similarity` - Process multiple text pairs

## Installation & Setup

### Local Development

1. **Clone the repository:**
```bash
git clone <repository-url>
cd semantic-similarity-api
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
and seperately install 
pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

3. **Run the API server:**
```bash
python api_server.py
```

4. **Test the API:**
```bash
python test_api.py
```

### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t semantic-similarity-api .
```

2. **Run the container:**
```bash
docker run -p 5000:5000 semantic-similarity-api
```

### Cloud Deployment (Render)

1. **Connect your GitHub repository to Render**
2. **Create a new Web Service**
3. **Configure the service:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python api_server.py`
   - **Environment:** Python 3.9

## Model Details

### Core Approach
The model uses **Sentence-BERT (SBERT)** with an ensemble approach:

1. **Primary Model**: `all-MiniLM-L6-v2` (fast and accurate)
2. **Ensemble Models**: 
   - `paraphrase-MiniLM-L6-v2` (paraphrase detection)
   - `all-mpnet-base-v2` (higher quality)

### Similarity Computation
Instead of basic cosine similarity, the model uses:

1. **Cosine Similarity** (40% weight)
2. **Euclidean Distance** converted to similarity (20% weight)
3. **Manhattan Distance** converted to similarity (20% weight)
4. **Dot Product** similarity (20% weight)

### Advanced Features
- **Text preprocessing** for better embedding quality
- **Score calibration** using sigmoid function
- **Error handling** with fallback scores
- **Model persistence** for faster loading

## Performance

- **Response Time**: ~200-500ms per request
- **Accuracy**: High semantic understanding
- **Scalability**: Stateless API design
- **Memory Usage**: ~500MB (model loading)

## Testing

### Manual Testing
```bash
curl -X POST http://localhost:5000/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "The weather is sunny today.",
    "text2": "It is a beautiful sunny day."
  }'
```

### Automated Testing
```bash
python test_api.py
```

## Deployment Options

### 1. Render (Recommended)
- Free tier available
- Automatic deployments from Git
- Easy configuration with `render.yaml`

### 2. Heroku
- Add `Procfile`:
```
web: python api_server.py
```

### 3. AWS/GCP
- Use Docker container
- Deploy to ECS/GKE

## Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Environment mode (production/development)

## Monitoring

The API includes:
- **Health checks** at `/health`
- **Request logging** with timestamps
- **Error tracking** and reporting
- **Model information** endpoint

## Security

- **Input validation** for all endpoints
- **Error handling** without exposing internals
- **CORS support** for web applications
- **Non-root user** in Docker container

## Future Enhancements

1. **Fine-tuning** on domain-specific data
2. **Caching** for repeated requests
3. **Rate limiting** for API protection
4. **Authentication** for production use
5. **Model versioning** and A/B testing

## Troubleshooting

### Common Issues

1. **Model loading fails:**
   - Check internet connection (downloads models)
   - Ensure sufficient disk space

2. **Memory issues:**
   - Reduce ensemble models in `semantic_similarity_model.py`
   - Use smaller model variants

3. **Deployment fails:**
   - Check `requirements.txt` compatibility
   - Verify Python version (3.9+)

## License

This project is created for the DataNeuron assessment.

## Contact

For questions or issues, please refer to the assessment submission guidelines. 