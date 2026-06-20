from setuptools import setup, find_packages

setup(
    name="churn-mlops",
    version="1.0.0",
    description="End-to-End MLOps Pipeline for Customer Churn Prediction",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "mlflow>=2.10",
        "fastapi>=0.109",
        "uvicorn>=0.27",
        "pydantic>=1.7.4,<2.0",
        "pyyaml>=6.0",
        "joblib>=1.3",
    ],
)