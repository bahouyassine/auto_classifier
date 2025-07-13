# LLM Classification Evaluator

This project provides a backend powered by FastAPI together with a minimal React front‑end. It allows you to generate and evaluate classification prompts using OpenAI models.

## Features

- **Automated Prompt Generation**: Creates initial classification prompts based on your data
- **Iterative Optimization**: Improves prompts based on misclassification patterns
- **Parallel Processing**: Evaluates multiple rows simultaneously for efficiency
- **Real-time Metrics**: Tracks accuracy, precision, recall, and F1-score
- **Test Set Evaluation**: Applies optimized prompts to unseen data
- **Export Results**: Download predictions as CSV files

## How It Works

1. Start the FastAPI server:
   ```bash
   pip install -r requirements.txt
   python backend/api.py
   ```
2. In the `frontend` directory install dependencies and run the React app:
   ```bash
   cd frontend
   npm install
   npm start
   ```
3. Use the web interface to upload data, configure parameters and trigger evaluation.

## Sample Data

The `data/` folder includes a small sample dataset to try the app.
