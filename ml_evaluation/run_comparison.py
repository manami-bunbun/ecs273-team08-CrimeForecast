import os
import pandas as pd
import logging
from baseline_rf import RandomForestPipeline
from correlation_enhanced_rf import CorrelationEnhancedRandomForest
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_comparison():
    """Run both models and compare their performance"""
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Define target columns (reduced for faster comparison)
    label_columns = [
        'crime_type_super',  # Main categories
        'crime_type_violent' # Binary classification
    ]
    
    results = []
    
    # Run baseline model
    logger.info("\nRunning baseline Random Forest model...")
    baseline = RandomForestPipeline()
    start_time = time.time()
    
    for label_col in label_columns:
        try:
            X_train, X_test, y_train, y_test, label_encoder = baseline.preprocess(label_col)
            model, metrics = baseline.train_and_evaluate(
                X_train, X_test, y_train, y_test,
                label_encoder, suffix=f"baseline_{label_col}"
            )
            metrics['model'] = 'baseline'
            metrics['label'] = label_col
            metrics['runtime'] = time.time() - start_time
            results.append(metrics)
        except Exception as e:
            logger.error(f"Error in baseline model for {label_col}: {e}")
    
    # Run correlation-enhanced model
    logger.info("\nRunning correlation-enhanced Random Forest model...")
    enhanced_model = CorrelationEnhancedRandomForest()
    start_time = time.time()
    
    try:
        enhanced_results = enhanced_model.run_experiment(label_columns)
        for _, row in enhanced_results.iterrows():
            metrics = {
                'model': 'correlation_enhanced',
                'label': row['label'],
                'accuracy': row['accuracy'],
                'f1_score': row['f1_score'],
                'precision': row['precision'],
                'runtime': time.time() - start_time
            }
            results.append(metrics)
    except Exception as e:
        logger.error(f"Error in correlation-enhanced model: {e}")
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    df_results.to_csv("outputs/model_comparison_results.csv", index=False)
    
    # Display results
    logger.info("\n=== Model Comparison Results ===")
    for label in label_columns:
        logger.info(f"\nResults for {label}:")
        label_results = df_results[df_results['label'] == label]
        logger.info("\n" + str(label_results.to_string()))
    
    return df_results

if __name__ == "__main__":
    run_comparison() 
