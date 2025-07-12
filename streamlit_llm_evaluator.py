import streamlit as st
import pandas as pd
import numpy as np
import openai
from typing import List, Dict, Tuple, Any
import asyncio
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LLM Classification Evaluator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LLMEvaluator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        
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

def main():
    st.title("🤖 LLM Classification Evaluator")
    st.markdown("Upload your datasets, set parameters, and let GPT-4o-mini optimize classification prompts!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()
        
        # Initialize evaluator
        evaluator = LLMEvaluator(api_key)
        
        st.header("Parameters")
        accuracy_threshold = st.slider("Accuracy Threshold", 0.5, 1.0, 0.8, 0.05)
        max_iterations = st.slider("Max Optimization Iterations", 1, 10, 5)
        max_workers = st.slider("Parallel Workers", 1, 10, 5)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Training Data (Annotated)")
        training_file = st.file_uploader(
            "Upload training dataset", 
            type=['csv', 'xlsx'], 
            key="training"
        )
        
        if training_file:
            try:
                if training_file.name.endswith('.csv'):
                    training_df = pd.read_csv(training_file)
                else:
                    training_df = pd.read_excel(training_file)
                
                st.success(f"Loaded {len(training_df)} training samples")
                st.dataframe(training_df.head())
                
                # Column selection
                st.subheader("Select Columns")
                all_columns = training_df.columns.tolist()
                
                target_column = st.selectbox(
                    "Target Column (Ground Truth)", 
                    all_columns,
                    help="Column containing the correct classifications"
                )
                
                feature_columns = st.multiselect(
                    "Feature Columns for Evaluation",
                    [col for col in all_columns if col != target_column],
                    default=[col for col in all_columns if col != target_column][:3],
                    help="Columns to use for classification"
                )
                
                if target_column and feature_columns:
                    # Extract classes
                    classes = evaluator.extract_classes_from_data(training_df, target_column)
                    st.info(f"Detected classes: {', '.join(classes)}")
                    
                    # Store in session state
                    st.session_state.training_df = training_df
                    st.session_state.target_column = target_column
                    st.session_state.feature_columns = feature_columns
                    st.session_state.classes = classes
                    
            except Exception as e:
                st.error(f"Error loading training data: {e}")
    
    with col2:
        st.header("🧪 Test Data")
        test_file = st.file_uploader(
            "Upload test dataset", 
            type=['csv', 'xlsx'], 
            key="test"
        )
        
        if test_file:
            try:
                if test_file.name.endswith('.csv'):
                    test_df = pd.read_csv(test_file)
                else:
                    test_df = pd.read_excel(test_file)
                
                st.success(f"Loaded {len(test_df)} test samples")
                st.dataframe(test_df.head())
                
                # Store in session state
                st.session_state.test_df = test_df
                
            except Exception as e:
                st.error(f"Error loading test data: {e}")
    
    # Evaluation section
    if (hasattr(st.session_state, 'training_df') and 
        hasattr(st.session_state, 'target_column') and 
        hasattr(st.session_state, 'feature_columns')):
        
        st.header("🚀 Training & Optimization")
        
        # Training button
        if not hasattr(st.session_state, 'training_completed'):
            if st.button("Begin Training & Optimization", type="primary"):
                st.session_state.training_in_progress = True
        
        # Training process
        if hasattr(st.session_state, 'training_in_progress') and st.session_state.training_in_progress:
            with st.spinner("Initializing evaluation..."):
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.container()
                
                # Generate initial prompt
                status_text.text("Generating initial prompt...")
                initial_prompt = evaluator.generate_initial_prompt(
                    st.session_state.classes,
                    st.session_state.feature_columns,
                    st.session_state.training_df
                )
                
                current_prompt = initial_prompt
                best_accuracy = 0
                iteration = 0
                all_iterations_results = []
                
                # Optimization loop
                while iteration < max_iterations and best_accuracy < accuracy_threshold:
                    iteration += 1
                    status_text.text(f"Iteration {iteration}: Evaluating training data...")
                    
                    # Evaluate training data
                    predictions = evaluator.evaluate_batch_parallel(
                        st.session_state.training_df,
                        current_prompt,
                        st.session_state.classes,
                        st.session_state.feature_columns,
                        max_workers
                    )
                    
                    # Calculate metrics
                    true_labels = st.session_state.training_df[st.session_state.target_column].tolist()
                    metrics = evaluator.calculate_metrics(true_labels, predictions)
                    
                    # Store iteration results
                    all_iterations_results.append({
                        'iteration': iteration,
                        'metrics': metrics,
                        'prompt': current_prompt
                    })
                    
                    current_accuracy = metrics['accuracy']
                    
                    if current_accuracy >= accuracy_threshold:
                        status_text.text(f"🎉 Target accuracy {accuracy_threshold:.1%} achieved!")
                        break
                    
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                    
                    # Optimize prompt if not at target accuracy
                    if iteration < max_iterations and current_accuracy < accuracy_threshold:
                        status_text.text(f"Optimizing prompt for iteration {iteration + 1}...")
                        
                        # Find misclassified samples
                        training_with_preds = st.session_state.training_df.copy()
                        training_with_preds['predicted'] = predictions
                        misclassified = training_with_preds[
                            training_with_preds[st.session_state.target_column] != training_with_preds['predicted']
                        ]
                        
                        # Optimize prompt
                        current_prompt = evaluator.optimize_prompt(
                            current_prompt,
                            misclassified,
                            st.session_state.classes,
                            st.session_state.feature_columns
                        )
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                # Store results in session state
                st.session_state.final_prompt = current_prompt
                st.session_state.final_metrics = metrics
                st.session_state.all_iterations = all_iterations_results
                st.session_state.training_completed = True
                st.session_state.training_in_progress = False
                
                # Force rerun to show results
                st.rerun()
    
    # Display training results if completed
    if hasattr(st.session_state, 'training_completed') and st.session_state.training_completed:
        st.header("📊 Training Results")
        
        # Display final metrics
        st.subheader("Final Performance")
        col1, col2, col3, col4 = st.columns(4)
        final_metrics = st.session_state.final_metrics
        col1.metric("Accuracy", f"{final_metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{final_metrics['precision']:.3f}")
        col3.metric("Recall", f"{final_metrics['recall']:.3f}")
        col4.metric("F1 Score", f"{final_metrics['f1_score']:.3f}")
        
        # Show iteration history
        if len(st.session_state.all_iterations) > 1:
            st.subheader("Training Progress")
            iterations_df = pd.DataFrame([
                {
                    'Iteration': result['iteration'],
                    'Accuracy': result['metrics']['accuracy'],
                    'Precision': result['metrics']['precision'],
                    'Recall': result['metrics']['recall'],
                    'F1 Score': result['metrics']['f1_score']
                }
                for result in st.session_state.all_iterations
            ])
            
            # Plot training progress
            fig = px.line(iterations_df, x='Iteration', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                         title='Training Progress Over Iterations')
            st.plotly_chart(fig, use_container_width=True)
        
        # Display final prompt
        st.subheader("Final Optimized Prompt")
        with st.expander("View Final Prompt", expanded=False):
            st.code(st.session_state.final_prompt, language="text")
        
        # Reset training button
        if st.button("🔄 Restart Training", type="secondary"):
            # Clear training session state
            for key in ['training_completed', 'training_in_progress', 'final_prompt', 'final_metrics', 'all_iterations']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Test set evaluation section (separate from training)
    if (hasattr(st.session_state, 'training_completed') and 
        st.session_state.training_completed and 
        hasattr(st.session_state, 'test_df')):
        
        st.header("🧪 Test Set Evaluation")
        
        if not hasattr(st.session_state, 'test_results'):
            if st.button("Evaluate Test Set", type="primary"):
                with st.spinner("Evaluating test set..."):
                    test_predictions = evaluator.evaluate_batch_parallel(
                        st.session_state.test_df,
                        st.session_state.final_prompt,
                        st.session_state.classes,
                        st.session_state.feature_columns,
                        max_workers
                    )
                    
                    # Add predictions to test dataframe
                    test_results = st.session_state.test_df.copy()
                    test_results['predicted_class'] = test_predictions
                    
                    # Store in session state
                    st.session_state.test_results = test_results
                    st.session_state.test_predictions = test_predictions
                    
                    st.rerun()
        
        # Display test results if available
        if hasattr(st.session_state, 'test_results'):
            st.subheader("Test Results")
            st.dataframe(st.session_state.test_results)
            
            # Class distribution
            st.subheader("Prediction Distribution")
            pred_counts = pd.Series(st.session_state.test_predictions).value_counts()
            fig = px.bar(x=pred_counts.index, y=pred_counts.values, 
                        title="Distribution of Predicted Classes",
                        labels={'x': 'Class', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = st.session_state.test_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Test Results",
                data=csv,
                file_name="test_results.csv",
                mime="text/csv"
            )
            
            # Reset test evaluation
            if st.button("🔄 Re-evaluate Test Set"):
                if 'test_results' in st.session_state:
                    del st.session_state['test_results']
                if 'test_predictions' in st.session_state:
                    del st.session_state['test_predictions']
                st.rerun()

if __name__ == "__main__":
    main()
