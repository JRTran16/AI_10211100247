# ML/AI Explorer

A comprehensive machine learning and AI exploration tool built with Streamlit, featuring regression analysis, clustering, neural networks, and LLM-powered Q&A capabilities.

## Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jrtran16-ml-ai-explorer-app-n0gcfs.streamlit.app/)

## Features

### 1. Regression Analysis
**Usage:**
1. Upload a CSV file containing your dataset
2. Select the target column (numeric only)
3. Choose feature columns for regression
4. Apply optional preprocessing steps:
   - Remove outliers
   - Normalize features
   - Log transform
5. View model performance metrics (MAE, R²)
6. Make custom predictions using the trained model

### 2. K-Means Clustering
**Usage:**
1. Upload a CSV file with your data
2. Select 2-3 features for clustering
3. Choose the number of clusters (2-10)
4. View the cluster visualization (2D or 3D)
5. Download the clustered data with labels

### 3. Neural Network Training
**Usage:**
1. Upload a CSV file
2. Select target and feature columns
3. Configure hyperparameters:
   - Number of epochs
   - Learning rate
4. Train the model
5. Make predictions using the trained model
6. View training loss curve

### 4. LLM Q&A System
**Usage:**
1. Upload a PDF or CSV document
2. Enter your question about the document
3. Receive AI-generated answers with relevant context

## Technical Architecture

### Models Used
1. **Regression Analysis**
   - Linear Regression from scikit-learn
   - Supports multiple features
   - Provides performance metrics

2. **Clustering**
   - K-Means from scikit-learn
   - 2D and 3D visualization
   - Automatic categorical encoding

3. **Neural Network**
   - MLPClassifier from scikit-learn
   - Architecture: (64, 32) hidden layers
   - ReLU activation, Adam optimizer

4. **LLM System**
   - Gemini 1.5 Pro
   - Sentence Transformers for embeddings
   - FAISS for vector storage

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JRTran16/ML-AI-explorer.git
   cd ML-AI-explorer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python -m streamlit run app.py
   ```

## Requirements
See `requirements.txt` for detailed package versions and dependencies.
