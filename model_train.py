import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings
warnings.filterwarnings('ignore')


class CyberIncidentClassifier:
    def __init__(self, max_words=50000, max_len=200, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.category_encoder = LabelEncoder()
        self.subcategory_encoder = LabelEncoder()
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing for cybersecurity domain"""
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text, flags=re.MULTILINE)
        
        # Replace email addresses with 'email'
        text = re.sub(r'\S+@\S+', ' email ', text)
        
        # Replace IP addresses with 'ip_address'
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ip_address ', text)
        
        # Replace numbers with 'num'
        text = re.sub(r'\d+', ' num ', text)
        
        # Remove special characters but keep important security-related symbols
        text = re.sub(r'[^a-zA-Z0-9\s@#$%&*/_-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords but keep security-relevant ones
        stop_words = set(stopwords.words('english')) - {'no', 'not', 'false', 'true'}
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)

    def build_model(self, num_categories, num_subcategories):
        """Build hierarchical model with attention mechanism"""
        # Input layer
        input_text = Input(shape=(self.max_len,))
        
        # Embedding layer
        embedding = Embedding(self.max_words, self.embedding_dim, input_length=self.max_len)(input_text)
        
        # Main LSTM branch
        lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(embedding)
        lstm_1 = BatchNormalization()(lstm_1)
        lstm_1 = Dropout(0.3)(lstm_1)
        
        lstm_2 = Bidirectional(LSTM(64, return_sequences=True))(lstm_1)
        lstm_2 = BatchNormalization()(lstm_2)
        lstm_2 = Dropout(0.3)(lstm_2)
        
        # Global pooling
        pooled = GlobalMaxPooling1D()(lstm_2)
        
        # Category branch
        category_dense = Dense(256, activation='relu')(pooled)
        category_dense = BatchNormalization()(category_dense)
        category_dense = Dropout(0.3)(category_dense)
        category_output = Dense(num_categories, activation='softmax', name='category_output')(category_dense)
        
        # Subcategory branch (with category information)
        subcategory_input = Concatenate()([pooled, category_dense])
        subcategory_dense = Dense(512, activation='relu')(subcategory_input)
        subcategory_dense = BatchNormalization()(subcategory_dense)
        subcategory_dense = Dropout(0.3)(subcategory_dense)
        subcategory_output = Dense(num_subcategories, activation='softmax', name='subcategory_output')(subcategory_dense)
        
        model = Model(inputs=input_text, outputs=[category_output, subcategory_output])
        
        # Custom learning rates for different parts of the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss={
                'category_output': 'categorical_crossentropy',
                'subcategory_output': 'categorical_crossentropy'
            },
            loss_weights={
                'category_output': 1.0,
                'subcategory_output': 2.0  # Give more weight to subcategory classification
            },
            metrics={
                'category_output': ['accuracy'],
                'subcategory_output': ['accuracy']
            }  # Fixed: Specify metrics for each output
        )
        
        return model


    def prepare_data(self, texts, categories, subcategories):
        """Prepare data for training"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Fit tokenizer
        self.tokenizer.fit_on_texts(processed_texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Encode categories and subcategories
        y_cat = self.category_encoder.fit_transform(categories)
        y_subcat = self.subcategory_encoder.fit_transform(subcategories)
        
        # Convert to one-hot encoding
        y_cat = to_categorical(y_cat)
        y_subcat = to_categorical(y_subcat)
        
        return X, y_cat, y_subcat

    def train(self, X, y_cat, y_subcat, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model with early stopping and checkpointing"""
        model = self.build_model(y_cat.shape[1], y_subcat.shape[1])
        
        # Fixed: Updated callbacks with proper configuration
        early_stopping = EarlyStopping(
            monitor='val_subcategory_output_accuracy',
            mode='max',  # Explicitly specify we want to maximize accuracy
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.keras',
            monitor='val_subcategory_output_accuracy',
            mode='max',  # Also specify mode for checkpoint
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        history = model.fit(
            X,
            {'category_output': y_cat, 'subcategory_output': y_subcat},
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        self.model = model
        return history

    def predict(self, texts):
        """Make predictions on new texts"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Make predictions
        cat_preds, subcat_preds = self.model.predict(X)
        
        # Convert to labels
        cat_labels = self.category_encoder.inverse_transform(np.argmax(cat_preds, axis=1))
        subcat_labels = self.subcategory_encoder.inverse_transform(np.argmax(subcat_preds, axis=1))
        
        return cat_labels, subcat_labels
    
    
    
# Usage example
if __name__ == "__main__":
    # Load your data
    train_df = pd.read_csv('/kaggle/input/indiaai/train.csv')
    
    # Initialize classifier
    classifier = CyberIncidentClassifier()
    
    # Prepare data
    X, y_cat, y_subcat = classifier.prepare_data(
        train_df['crimeaditionalinfo'],
        train_df['category'],
        train_df['sub_category']
    )
    
    # Train model
    history = classifier.train(X, y_cat, y_subcat)
    
    # Save the model and encoders
    classifier.model.save('/kaggle/working/model_output/model.h5')
    
    import pickle
    with open('/kaggle/working/model_output/tokenizer.pickle', 'wb') as handle:
        pickle.dump(classifier.tokenizer, handle)
    with open('/kaggle/working/model_output/category_encoder.pickle', 'wb') as handle:
        pickle.dump(classifier.category_encoder, handle)
    with open('/kaggle/working/model_output/subcategory_encoder.pickle', 'wb') as handle:
        pickle.dump(classifier.subcategory_encoder, handle)