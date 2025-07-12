# LLM Classification Evaluator

A Streamlit application that uses GPT-4o-mini to automatically generate and optimize classification prompts through iterative evaluation.

## Features

- **Automated Prompt Generation**: Creates initial classification prompts based on your data
- **Iterative Optimization**: Improves prompts based on misclassification patterns
- **Parallel Processing**: Evaluates multiple rows simultaneously for efficiency
- **Real-time Metrics**: Tracks accuracy, precision, recall, and F1-score
- **Test Set Evaluation**: Applies optimized prompts to unseen data
- **Export Results**: Download predictions as CSV files

## How It Works

1. **Upload Data**: Provide annotated training data and test datasets
2. **Configure**: Set target columns, accuracy threshold, and optimization parameters
3. **Auto-Optimize**: The app generates prompts and iteratively improves them
4. **Evaluate**: Once accuracy threshold is met, evaluate the test set
5. **Export**: Download results for further analysis

## Usage

1. Enter your OpenAI API key in the sidebar
2. Upload your datasets:
   - **Training Data**: CSV/Excel file with ground truth labels
   - **Test Data**: CSV/Excel file to be classified
3. Configure parameters:
   - Select target column (ground truth)
   - Choose feature columns for classification
   - Set accuracy threshold (default: 80%)
   - Adjust max iterations and parallel workers
4. Click "Begin Training & Optimization" to start the process

## Data Format

### Training Data
- Must contain a target column with ground truth classifications
- Should include feature columns relevant to classification
- Supports CSV and Excel formats

### Test Data
- Should have the same feature columns as training data
- No target column required
- Results will include predicted classifications

## Requirements

- OpenAI API key with GPT-4o-mini access
- Internet connection for API calls

## Sample Data

The application includes sample datasets for testing:
- Challenging 5-class sentiment classification problem
- 260 training samples, 150 test samples
- Categories: frustrated, disappointed, concerned, satisfied, enthusiastic
