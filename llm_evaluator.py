import logging
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import openai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEvaluator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def extract_classes_from_data(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Extract unique classes from the annotated dataset"""
        return sorted(df[target_column].unique().tolist())

    def generate_initial_prompt(self, classes: List[str], target_columns: List[str], sample_data: pd.DataFrame) -> str:
        """Generate initial evaluation prompt based on classes and sample data"""
        sample_rows = sample_data.head(3).to_dict('records')

        prompt = f"""You are a classification expert. Your task is to classify data into one of these categories: {', '.join(classes)}.

Target columns to consider: {', '.join(target_columns)}

Here are some example data points:
{json.dumps(sample_rows, indent=2)}

Instructions:
1. Analyze the provided data carefully
2. Consider the target columns: {', '.join(target_columns)}
3. Classify into exactly one of these categories: {', '.join(classes)}
4. Respond with ONLY the classification label, no explanation

Classification:"""

        return prompt

    def evaluate_single_row(self, row_data: Dict, prompt: str, classes: List[str]) -> str:
        """Evaluate a single row using the LLM"""
        try:
            # Format the row data for the prompt
            row_text = json.dumps(row_data, indent=2)
            full_prompt = f"{prompt}\n\nData to classify:\n{row_text}\n\nClassification:"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise classification assistant. Respond only with the exact classification label."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )

            prediction = response.choices[0].message.content.strip()

            # Ensure prediction is in valid classes
            if prediction not in classes:
                # Try to find closest match
                prediction_lower = prediction.lower()
                for cls in classes:
                    if cls.lower() in prediction_lower or prediction_lower in cls.lower():
                        return cls
                # If no match found, return first class as fallback
                return classes[0]

            return prediction

        except Exception as e:
            logger.error(f"Error evaluating row: {e}")
            return classes[0]  # Return first class as fallback

    def evaluate_batch_parallel(self, df: pd.DataFrame, prompt: str, classes: List[str],
                               target_columns: List[str], max_workers: int = 5) -> List[str]:
        """Evaluate multiple rows in parallel"""
        predictions = []

        # Prepare data for evaluation (only target columns)
        eval_data = df[target_columns].to_dict('records')

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.evaluate_single_row, row_data, prompt, classes): i
                for i, row_data in enumerate(eval_data)
            }

            # Initialize predictions list with correct size
            predictions = [None] * len(eval_data)

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    prediction = future.result()
                    predictions[index] = prediction
                except Exception as e:
                    logger.error(f"Error in parallel evaluation: {e}")
                    predictions[index] = classes[0]  # Fallback

        return predictions

    def calculate_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Calculate classification metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def optimize_prompt(self, current_prompt: str, misclassified_samples: pd.DataFrame,
                       classes: List[str], target_columns: List[str]) -> str:
        """Generate an improved prompt based on misclassified samples"""
        if misclassified_samples.empty:
            return current_prompt

        # Analyze common misclassification patterns
        error_analysis = misclassified_samples.head(5).to_dict('records')

        optimization_prompt = f"""
        Current classification prompt needs improvement. Here are some misclassified examples:

        {json.dumps(error_analysis, indent=2)}

        Current prompt:
        {current_prompt}

        Please generate an improved classification prompt that:
        1. Better distinguishes between classes: {', '.join(classes)}
        2. Focuses on the key features in: {', '.join(target_columns)}
        3. Addresses the misclassification patterns shown above
        4. Maintains the same format and structure

        Improved prompt:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a prompt engineering expert. Generate improved classification prompts."},
                    {"role": "user", "content": optimization_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )

            improved_prompt = response.choices[0].message.content.strip()
            return improved_prompt

        except Exception as e:
            logger.error(f"Error optimizing prompt: {e}")
            return current_prompt
