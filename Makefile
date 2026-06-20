.PHONY: setup data train serve test docker-build docker-run clean

# Setup environment
setup:
	pip install -r requirements.txt
	pip install -e .

# Download and prepare data
data:
	python -c "from src.data.ingestion import download_data; download_data()"

# Train model
train:
	python train.py

# Start MLflow UI
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlruns.db

# Start API server
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test:
	pytest tests/ -v

# Build Docker image
docker-build:
	docker build -t churn-mlops:latest -f deployment/Dockerfile .

# Run with Docker Compose
docker-run:
	docker-compose -f deployment/docker-compose.yml up --build

# Clean artifacts
clean:
	rm -rf mlruns/
	rm -rf data/processed/*
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete