# mlbec4.py - Enhanced C++ Error Classifier with GridSearch and Gradient Boosting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score
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
                # Use download only if the resource is missing
                nltk.download(download_name, quiet=True)
            except:
                pass

download_nltk_data()

class CppErrorClassifier:
    """
    Tuned and enhanced classifier using TF-IDF, GridSearchCV (Random Forest), 
    and Gradient Boosting to improve C++ error classification accuracy.
    """
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
        """Load dataset from CSV."""
        try:
            if not os.path.exists(csv_file_path):
                # We assume the CSV exists for the production Streamlit app
                # but handle the file-not-found case gracefully.
                return None 
            
            self.data = pd.read_csv(csv_file_path)
            print(f"✓ Dataset loaded: {len(self.data)} examples")
            return self.data
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None
    
    def preprocess_text(self, text):
        """Enhanced preprocessing for C++ error messages, including error code normalization."""
        try:
            text = str(text).lower()
            
            cpp_keywords = {'int', 'char', 'float', 'double', 'void', 'const', 'static', 
                           'class', 'struct', 'template', 'namespace', 'using', 'std',
                           'cout', 'cin', 'endl', 'vector', 'string', 'nullptr', 'auto'}
            
            # Feature Engineering: Normalize C++ error codes (e.g., C2065)
            text = re.sub(r'(error|warning)\s+C\d+', ' MSVC_CODE ', text)
            
            # Normalize other common identifiers
            text = re.sub(r'\b\w+\.(cpp|h|hpp|c)\b', 'FILENAME', text)
            text = re.sub(r':\d+:\d+', ' LINECOLNUM ', text)
            text = re.sub(r'line\s+\d+', 'line NUMBER', text)
            text = re.sub(r'column\s+\d+', 'column NUMBER', text)
            text = re.sub(r"'[^']*'", ' QUOTED_ID ', text)
            text = re.sub(r'"[^"]*"', ' STRING_LIT ', text)
            text = re.sub(r'[^a-zA-Z0-9\s\(\)\{\}\[\]\;\:\.\,\_\-\<\>]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            programming_terms = {'error', 'expected', 'missing', 'undefined', 'undeclared',
                               'identifier', 'variable', 'function', 'type', 'cannot',
                               'invalid', 'before', 'after', 'declaration', 'statement',
                               'expression', 'token', 'warning', 'fatal'}
            
            filtered_tokens = []
            for token in tokens:
                if (token not in self.stop_words or 
                    token in programming_terms or 
                    token in cpp_keywords or
                    len(token) <= 3):
                    filtered_tokens.append(token)
            
            # Use lemmatization
            try:
                processed_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            except:
                processed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
            
            return ' '.join(processed_tokens)
        except Exception as e:
            return str(text).lower()
    
    def prepare_features(self):
        """Prepare features and labels for training."""
        if self.data is None:
            raise ValueError("No data loaded!")
        
        self.data['processed_text'] = self.data['error_message'].apply(self.preprocess_text)
        self.data = self.data[self.data['processed_text'].str.strip() != '']
        
        X = self.data['processed_text']
        y = self.data['error_type']
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Trains multiple models, including a tuned Random Forest and Gradient Boosting."""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Features: {X_train_tfidf.shape[1]}")
        
        # Hyperparameter Tuning for Random Forest
        rf_base = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
        }
        scorer = make_scorer(f1_score, average='weighted', zero_division=0)
        
        # Use a smaller CV split (3) for faster tuning
        grid_search_rf = GridSearchCV(
            estimator=rf_base, 
            param_grid=rf_param_grid, 
            scoring=scorer, 
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state), 
            verbose=0, 
            n_jobs=-1
        )
        grid_search_rf.fit(X_train_tfidf, y_train)
        rf_model = grid_search_rf.best_estimator_
        print(f"Best RF Params: {grid_search_rf.best_params_}")

        models = {
            'Random Forest (Tuned)': rf_model, 
            'SVM': SVC(kernel='rbf', C=10.0, probability=True, random_state=random_state, class_weight='balanced'),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(C=10.0, max_iter=1000, random_state=random_state, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state)
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        results = {}
        
        # Process all models
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # RF is already trained by GridSearchCV
            if name != 'Random Forest (Tuned)':
                model.fit(X_train_tfidf, y_train)
            
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=skf)
                cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
            except:
                cv_mean, cv_std = accuracy, 0.0
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred
            }
            
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        
        self.best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\nBest model: {self.best_model_name}")
        self.models, self.X_test, self.y_test = results, X_test_tfidf, y_test
        
        return results
    
    # ... (evaluate_detailed, predict_error_type, save_model, load_model are unchanged) ...
    def evaluate_detailed(self, y_true, y_pred):
        """Evaluation metrics"""
        target_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
    
    def predict_error_type(self, error_message):
        """Predict error type"""
        if not self.best_model or not self.vectorizer:
            return None, None
        
        processed = self.preprocess_text(error_message)
        if not processed.strip():
            return None, None
        
        tfidf = self.vectorizer.transform([processed])
        # Check if the model supports predict_proba (e.g., GB doesn't if it's not explicitly set)
        if hasattr(self.best_model, 'predict_proba'):
             proba = self.best_model.predict_proba(tfidf)[0]
        else:
             # Fallback: Assign uniform probability if not available
             num_classes = len(self.label_encoder.classes_)
             proba = [0.0] * num_classes
             # Find index of predicted class and set a high probability (e.g., 90%)
             prediction_index = self.best_model.predict(tfidf)[0]
             proba[prediction_index] = 0.90 
             
        prediction = self.best_model.predict(tfidf)[0]
        
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence_dict = dict(zip(self.label_encoder.classes_, proba))
        
        return predicted_class, confidence_dict
    
    def save_model(self, filepath='cpp_error_classifier.pkl'):
        """Save model"""
        if not self.best_model:
            return
        
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
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.best_model = data['best_model']
        self.best_model_name = data.get('best_model_name', 'Unknown')
        self.vectorizer = data['vectorizer']
        self.label_encoder = data['label_encoder']
        self.models = data.get('models', {})
        
        print(f"✓ Model loaded: {self.best_model_name}")

def main():
    """Main training function"""
    print("\n" + "="*70)
    print("C++ ERROR CLASSIFIER - TRAINING")
    print("="*70)
    
    classifier = CppErrorClassifier()
    
    # Load dataset
    data = classifier.load_dataset('cpp_error_dataset.csv')
    if data is None:
        print("\n✗ Please provide 'cpp_error_dataset.csv'")
        return
    
    # Prepare and train
    X, y = classifier.prepare_features()
    # Check if dataset is large enough for split
    if len(X) < 10:
        print("\n✗ Dataset is too small to proceed with training.")
        return
        
    results = classifier.train_models(X, y)
    
    # Test predictions
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    tests = [
        "error: 'undefined_var' was not declared",
        "expected ';' before 'return'",
        "cannot convert 'string' to 'int'",
        "error C2065: 'count' : undeclared identifier" 
    ]
    
    for msg in tests:
        pred, conf = classifier.predict_error_type(msg)
        if pred:
            print(f"\n'{msg}'")
            print(f"  → {pred.upper()} ({max(conf.values()):.2%})")
            print(f"  → All: {conf}")
    
    # Save
    classifier.save_model()
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
