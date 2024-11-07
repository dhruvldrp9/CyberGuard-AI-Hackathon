import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ['TF_DISABLE_CUDNN_RNN'] = '1'

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_saved_model(model_path='model_output1'):
    """
    Load the saved model and its components
    """
    try:
        # Load the model
        model = tf.keras.models.load_model(f'{model_path}/model.h5')
        
        # Load tokenizer and encoders
        with open(f'{model_path}/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(f'{model_path}/category_encoder.pickle', 'rb') as handle:
            category_encoder = pickle.load(handle)
        with open(f'{model_path}/subcategory_encoder.pickle', 'rb') as handle:
            subcategory_encoder = pickle.load(handle)
            
        return model, tokenizer, category_encoder, subcategory_encoder
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        raise

def preprocess_text(text):
    """
    Preprocess text with complete NLP pipeline
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def handle_unknown_labels(test_df, category_encoder, subcategory_encoder):
    """
    Handle unknown labels in test data by filtering them out
    """
    known_categories = set(category_encoder.classes_)
    known_subcategories = set(subcategory_encoder.classes_)
    
    # Create mask for known categories and subcategories
    category_mask = test_df['category'].isin(known_categories)
    subcategory_mask = test_df['sub_category'].isin(known_subcategories)
    
    # Filter dataset
    valid_df = test_df[category_mask & subcategory_mask].copy()
    
    # Print statistics about filtered data
    print("\nLabel Statistics:")
    print(f"Original test set size: {len(test_df)}")
    print(f"Filtered test set size: {len(valid_df)}")
    print(f"Removed {len(test_df) - len(valid_df)} samples with unknown labels")
    
    if len(test_df) - len(valid_df) > 0:
        print("\nUnknown categories found:")
        unknown_categories = set(test_df['category']) - known_categories
        if unknown_categories:
            print("Categories:", unknown_categories)
        
        unknown_subcategories = set(test_df['sub_category']) - known_subcategories
        if unknown_subcategories:
            print("Subcategories:", unknown_subcategories)
    
    return valid_df

def evaluate_model(model, tokenizer, category_encoder, subcategory_encoder, test_df, max_len=200):
    """
    Evaluate the model on test data and return detailed metrics
    """
    # Handle unknown labels
    valid_df = handle_unknown_labels(test_df, category_encoder, subcategory_encoder)
    
    if len(valid_df) == 0:
        raise ValueError("No valid samples remaining after filtering unknown labels")
    
    # Prepare test data
    test_texts = valid_df['crimeaditionalinfo'].apply(preprocess_text)
    sequences = tokenizer.texts_to_sequences(test_texts)
    X_test = pad_sequences(sequences, maxlen=max_len)
    
    # Get true labels
    y_true_cat = category_encoder.transform(valid_df['category'])
    y_true_subcat = subcategory_encoder.transform(valid_df['sub_category'])
    
    # Get predictions and convert to numpy array if needed
    predictions = model.predict(X_test, verbose=0)
    if isinstance(predictions, list):
        predictions = [np.array(pred) for pred in predictions]
        category_preds = np.argmax(predictions[0], axis=1)
        subcategory_preds = np.argmax(predictions[1], axis=1)
        
        # Also convert confidence scores
        category_confidence = np.max(predictions[0], axis=1)
        subcategory_confidence = np.max(predictions[1], axis=1)
    else:
        num_categories = len(category_encoder.classes_)
        category_preds = np.argmax(predictions[:, :num_categories], axis=1)
        subcategory_preds = np.argmax(predictions[:, num_categories:], axis=1)
        
        # Get confidence scores
        category_confidence = np.max(predictions[:, :num_categories], axis=1)
        subcategory_confidence = np.max(predictions[:, num_categories:], axis=1)
    
    # Get actual classes present in the data
    present_category_classes = sorted(list(set(y_true_cat)))
    present_subcategory_classes = sorted(list(set(y_true_subcat)))
    
    # Calculate metrics with explicit label lists
    metrics = {
        'category': {
            'accuracy': accuracy_score(y_true_cat, category_preds),
            'report': classification_report(
                y_true_cat, 
                category_preds,
                labels=present_category_classes,
                target_names=[category_encoder.classes_[i] for i in present_category_classes],
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_true_cat, category_preds)
        },
        'subcategory': {
            'accuracy': accuracy_score(y_true_subcat, subcategory_preds),
            'report': classification_report(
                y_true_subcat, 
                subcategory_preds,
                labels=present_subcategory_classes,
                target_names=[subcategory_encoder.classes_[i] for i in present_subcategory_classes],
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_true_subcat, subcategory_preds)
        }
    }
    
    # Add confidence scores
    metrics['confidence_scores'] = {
        'category': {
            'mean': float(np.mean(category_confidence)),
            'std': float(np.std(category_confidence))
        },
        'subcategory': {
            'mean': float(np.mean(subcategory_confidence)),
            'std': float(np.std(subcategory_confidence))
        }
    }
    
    return metrics, valid_df

def plot_confusion_matrix(conf_matrix, classes, title):
    """
    Plot confusion matrix using seaborn with improved visibility
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    try:
        # Load test data
        test_df = pd.read_csv('test.csv')
        
        print("Loading model and components...")
        model, tokenizer, category_encoder, subcategory_encoder = load_saved_model()
        
        print("Evaluating model...")
        metrics, valid_df = evaluate_model(model, tokenizer, category_encoder, 
                                         subcategory_encoder, test_df)
        
        print("\n=== Model Evaluation Results ===")
        print(f"\nCategory Classification Accuracy: {metrics['category']['accuracy']:.4f}")
        print(f"Subcategory Classification Accuracy: {metrics['subcategory']['accuracy']:.4f}")
        
        print("\nConfidence Scores:")
        print(f"Category - Mean: {metrics['confidence_scores']['category']['mean']:.4f}, "
              f"Std: {metrics['confidence_scores']['category']['std']:.4f}")
        print(f"Subcategory - Mean: {metrics['confidence_scores']['subcategory']['mean']:.4f}, "
              f"Std: {metrics['confidence_scores']['subcategory']['std']:.4f}")
        
        # Plot confusion matrices
        plot_confusion_matrix(metrics['category']['confusion_matrix'],
                            category_encoder.classes_,
                            'Category Classification Confusion Matrix (Normalized)')
        
        plot_confusion_matrix(metrics['subcategory']['confusion_matrix'],
                            subcategory_encoder.classes_,
                            'Subcategory Classification Confusion Matrix (Normalized)')
        
        # Save metrics to file
        output_dict = {
            'metrics': {
                'category_accuracy': float(metrics['category']['accuracy']),
                'subcategory_accuracy': float(metrics['subcategory']['accuracy']),
                'category_report': metrics['category']['report'],
                'subcategory_report': metrics['subcategory']['report'],
                'confidence_scores': metrics['confidence_scores']
            },
            'data_stats': {
                'original_size': len(test_df),
                'evaluated_size': len(valid_df)
            }
        }
        
        with open('evaluation_metrics.json', 'w') as f:
            json.dump(output_dict, f, indent=4)
        
        print("\nEvaluation metrics have been saved to 'evaluation_metrics.json'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
        
