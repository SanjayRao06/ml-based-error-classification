import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# Download required NLTK data
def download_nltk_data():
    """Download necessary NLTK datasets"""
    required_downloads = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    for resource_path, download_name in required_downloads:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except:
                pass

download_nltk_data()

class CppErrorClassifier:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.data = None
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def load_dataset(self, csv_file_path):
        """Load dataset from CSV file"""
        try:
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file '{csv_file_path}' not found!")
            
            self.data = pd.read_csv(csv_file_path)
            
            # Validate columns
            required_cols = ['error_message', 'error_type']
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            # Remove duplicates and empty rows
            self.data = self.data.drop_duplicates()
            self.data = self.data[self.data['error_message'].notna()]
            self.data = self.data[self.data['error_type'].notna()]
            
            print(f"✓ Dataset loaded: {len(self.data)} examples")
            print(f"\nClass distribution:")
            print(self.data['error_type'].value_counts())
            
            return self.data
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None
    
    def preprocess_text(self, text):
        """Enhanced preprocessing for C++ error messages"""
        try:
            text = str(text).lower()
            
            # Important C++ keywords to preserve
            cpp_keywords = {'int', 'char', 'float', 'double', 'void', 'const', 'static', 
                           'class', 'struct', 'template', 'namespace', 'using', 'std',
                           'cout', 'cin', 'endl', 'vector', 'string', 'nullptr', 'auto'}
            
            # Normalize file paths and line numbers
            text = re.sub(r'\b\w+\.(cpp|h|hpp|c)\b', 'FILENAME', text)
            text = re.sub(r':\d+:\d+', ' LINECOLNUM ', text)
            text = re.sub(r'line\s+\d+', 'line NUMBER', text)
            text = re.sub(r'column\s+\d+', 'column NUMBER', text)
            
            # Normalize identifiers and literals
            text = re.sub(r"'[^']*'", ' QUOTED_ID ', text)
            text = re.sub(r'"[^"]*"', ' STRING_LIT ', text)
            
            # Remove special characters but keep some punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\(\)\{\}\[\]\;\:\.\,\_\-\<\>]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenization
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Keep important programming terms
            programming_terms = {'error', 'expected', 'missing', 'undefined', 'undeclared',
                               'identifier', 'variable', 'function', 'type', 'cannot',
                               'invalid', 'before', 'after', 'declaration', 'statement',
                               'expression', 'token', 'warning', 'fatal', 'syntax',
                               'semantic', 'lexical', 'declared', 'convert', 'mismatch'}
            
            # Filter tokens intelligently
            filtered_tokens = []
            for token in tokens:
                if (token not in self.stop_words or 
                    token in programming_terms or 
                    token in cpp_keywords or
                    len(token) <= 3):
                    filtered_tokens.append(token)
            
            # Lemmatization
            try:
                processed_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            except:
                processed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
            
            return ' '.join(processed_tokens)
        except Exception as e:
            print(f"Warning: Error preprocessing text: {e}")
            return str(text).lower()
    
    def prepare_features(self):
        """Prepare features and labels"""
        if self.data is None:
            raise ValueError("No data loaded! Call load_dataset() first.")
        
        print("\nPreprocessing text...")
        self.data['processed_text'] = self.data['error_message'].apply(self.preprocess_text)
        
        # Remove empty processed rows
        self.data = self.data[self.data['processed_text'].str.strip() != '']
        
        if len(self.data) == 0:
            raise ValueError("No valid data after preprocessing!")
        
        X = self.data['processed_text']
        y = self.data['error_type']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Classes found: {list(self.label_encoder.classes_)}")
        print(f"Dataset size after preprocessing: {len(X)}")
        
        return X, y_encoded
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models and select the best one"""
        print(f"\n{'='*70}")
        print(f"TRAINING MODELS")
        print(f"{'='*70}")
        print(f"Training with {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            norm='l2'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature dimensions: {X_train_tfidf.shape[1]}")
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=25,
                min_samples_split=5,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                random_state=random_state,
                class_weight='balanced'
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(
                C=10.0,
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced',
                solver='lbfgs'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        }
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train_tfidf, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    model, X_train_tfidf, y_train, 
                    cv=skf, scoring='accuracy', n_jobs=-1
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = accuracy
                cv_std = 0.0
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred
            }
            
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (± {cv_std*2:.4f})")
        
        # Select best model
        self.best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"CV Score: {results[self.best_model_name]['cv_mean']:.4f}")
        print(f"{'='*70}")
        
        # Detailed evaluation
        self.evaluate_detailed(y_test, results[self.best_model_name]['predictions'])
        
        # Store results
        self.models = results
        self.X_test = X_test_tfidf
        self.y_test = y_test
        
        return results
    
    def evaluate_detailed(self, y_true, y_pred):
        """Print detailed evaluation metrics"""
        target_names = self.label_encoder.classes_
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    def predict_error_type(self, error_message):
        """Predict error type for a new error message"""
        if not self.best_model or not self.vectorizer:
            raise ValueError("Model not trained! Call train_models() first.")
        
        processed = self.preprocess_text(error_message)
        if not processed.strip():
            return None, None
        
        # Vectorize
        message_tfidf = self.vectorizer.transform([processed])
        
        # Predict
        prediction = self.best_model.predict(message_tfidf)[0]
        prediction_proba = self.best_model.predict_proba(message_tfidf)[0]
        
        # Decode
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence_dict = dict(zip(self.label_encoder.classes_, prediction_proba))
        
        return predicted_class, confidence_dict
    
    def save_model(self, filepath='cpp_error_classifier.pkl'):
        """Save trained model to file"""
        if not self.best_model:
            raise ValueError("No model to save! Train a model first.")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'models': self.models
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to '{filepath}'")
    
    def load_model(self, filepath='cpp_error_classifier.pkl'):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found!")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data.get('best_model_name', 'Unknown')
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.models = model_data.get('models', {})
        
        print(f"✓ Model loaded successfully: {self.best_model_name}")


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("C++ ERROR CLASSIFIER - TRAINING")
    print("="*70)
    
    classifier = CppErrorClassifier()
    
    # Load dataset
    csv_file = 'cpp_error_dataset.csv'
    data = classifier.load_dataset(csv_file)
    
    if data is None:
        print(f"\n✗ Failed to load dataset '{csv_file}'")
        print("Please ensure the CSV file exists with columns: 'error_message', 'error_type'")
        return
    
    # Prepare features
    try:
        X, y = classifier.prepare_features()
    except Exception as e:
        print(f"\n✗ Error preparing features: {e}")
        return
    
    # Train models
    try:
        results = classifier.train_models(X, y)
    except Exception as e:
        print(f"\n✗ Error training models: {e}")
        return
    
    # Test predictions
    print("\n" + "="*70)
    print("TESTING PREDICTIONS ON SAMPLE ERRORS")
    print("="*70)
    
    test_messages = [
        "error: 'undefined_var' was not declared in this scope",
        "error: expected ';' before 'return' statement",
        "error: cannot convert 'std::string' to 'int' in assignment",
        "error: invalid use of incomplete type 'class MyClass'",
        "warning: unused variable 'x'"
    ]
    
    for msg in test_messages:
        try:
            pred_type, confidence = classifier.predict_error_type(msg)
            if pred_type:
                print(f"\nMessage: '{msg}'")
                print(f"  → Predicted: {pred_type.upper()}")
                print(f"  → Confidence: {max(confidence.values()):.2%}")
                print(f"  → All scores: {', '.join([f'{k}: {v:.2%}' for k, v in sorted(confidence.items(), key=lambda x: x[1], reverse=True)])}")
        except Exception as e:
            print(f"\nError predicting for '{msg}': {e}")
    
    # Save model
    try:
        classifier.save_model()
    except Exception as e:
        print(f"\n✗ Error saving model: {e}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()