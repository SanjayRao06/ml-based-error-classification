# Complete C++ Error Classifier with Real Dataset
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

# Download required NLTK data with proper error handling
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
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
            print(f"✓ {download_name} already available")
        except LookupError:
            print(f"Downloading {download_name}...")
            try:
                nltk.download(download_name, quiet=True)
                print(f"✓ {download_name} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {download_name}: {e}")

# Initialize NLTK data
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
        
        # Initialize stopwords with error handling
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Warning: Stopwords not available, using empty set")
            self.stop_words = set()
    
    def create_comprehensive_dataset(self):
        """Create the comprehensive C++ error dataset"""
        dataset_content = '''error_message,error_type
"main.cpp:15:8: error: 'x' was not declared in this scope",lexical
"error C2065: 'count' : undeclared identifier",lexical
"/home/user/project.cpp:42:12: error: 'temp' undeclared (first use this function)",lexical
"Line 25: Identifier 'func' not found in current scope",lexical
"[Error] main.cpp (18): 'index' was not declared in this scope",lexical
"fatal error: 'result' has not been declared",lexical
"compilation error: undeclared variable 'data' used at line 33",lexical
"Error (line 67): variable 'value' not found",lexical
"main.cpp:89:5: error: 'calculate' was not declared in this scope",lexical
"error C2065: 'sum' : undeclared identifier at line 12",lexical
"src/main.cpp:156: error: 'ptr' undeclared",lexical
"Line 203: unknown variable 'length'",lexical
"main.cpp:45:10: error: 'buffer' undeclared (first use this function)",lexical
"Error: variable 'counter' not in scope at line 78",lexical
"fatal error: identifier 'array' not declared",lexical
"compilation terminated: 'size' undeclared at main.cpp:91",lexical
"[Error] Line 134: unknown identifier 'item'",lexical
"error: 'flag' was not declared in this scope",lexical
"main.cpp:167:3: error: 'node' undeclared",lexical
"Line 88: variable 'key' undefined",lexical
"main.cpp:25:10: error: expected ';' before 'return'",syntactic
"error C2143: syntax error : missing '}' before end of file",syntactic
"Line 34: expected ')' before ';' token",syntactic
"[Error] main.cpp (67): expected '{' before 'else'",syntactic
"fatal error: missing semicolon at end of statement (line 45)",syntactic
"src/main.cpp:123: error: expected '(' after 'if'",syntactic
"compilation error: missing closing parenthesis ')' at line 89",syntactic
"Error: expected ';' after expression on line 156",syntactic
"main.cpp:78:5: error: expected '{' before 'for'",syntactic
"Line 203: expected ')' after 'for' statement",syntactic
"error C2059: syntax error : missing ',' between function arguments",syntactic
"[Error] main.cpp (234): expected ';' before 'int'",syntactic
"fatal error: missing closing quote '\\"' at line 67",syntactic
"compilation terminated: expected '}' before 'else' at line 134",syntactic
"main.cpp:89:12: error: expected '(' after 'while'",syntactic
"Error (line 167): expected ':' after 'case'",syntactic
"Line 234: missing semicolon after variable declaration",syntactic
"src/main.cpp:301: error: expected ')' before '{'",syntactic
"main.cpp:45:8: error: expected ';' after 'break'",syntactic
"[Error] Line 178: missing opening bracket '['",syntactic
"main.cpp:15:5: error: cannot assign to 'x' (incompatible types 'int' and 'string')",semantic
"error C2440: cannot convert from 'int' to 'string'",semantic
"Line 45: function 'calculate' expects 2 arguments but 3 given",semantic
"compilation error: division by zero detected at line 78",semantic
"runtime error: array index 15 out of bounds (size 10)",semantic
"main.cpp:89:12: error: null pointer dereference",semantic
"Error: incompatible types in assignment at line 134",semantic
"[Warning] main.cpp (167): function returns void but value expected",semantic
"fatal error: cannot call method on null reference",semantic
"main.cpp:203:8: error: 'int' object has no member named 'length'",semantic
"Line 234: cannot convert 'float' to 'bool' in condition",semantic
"error C2440: 'main' function must return int",semantic
"runtime error: array subscript -1 out of range",semantic
"main.cpp:67:5: error: assignment of read-only variable 'CONST_VAL'",semantic
"compilation error: pure virtual function call detected",semantic
"runtime error: stack overflow detected in recursive function",semantic
"segmentation fault: memory access violation at address 0x0",semantic
"main.cpp:45:12: error: invalid conversion from 'char*' to 'int'",semantic
"Error: cannot delete non-pointer variable at line 89",semantic
"runtime error: double free or corruption detected",semantic
"iostream: No such file or directory",lexical
"fatal error: 'vector' file not found",lexical
"error C1083: Cannot open include file: 'algorithm': No such file or directory",lexical
"Line 1: #include expects \\"filename\\" or <filename>",lexical
"compilation terminated: missing header file 'string'",lexical
"[Error] Cannot find include file 'iostream.h'",lexical
"fatal error: 'bits/stdc++.h' not found (non-standard header)",lexical
"preprocessor error: #include directive not recognized",lexical
"main.cpp:1:10: fatal error: 'cstdio' file not found",lexical
"Error: header file 'memory' not found in include path",lexical
"main.cpp:10:5: error: 'std::cout' is not declared (did you forget 'using namespace std;'?)",lexical
"error C2065: 'cout' : undeclared identifier (include <iostream>)",lexical
"Line 15: 'string' undeclared (did you mean 'std::string'?)",lexical
"[Error] 'vector' not found (include <vector>)",lexical
"compilation error: 'nullptr' not supported in C++98 mode",lexical
"main.cpp:25:8: error: 'auto' does not name a type (C++11 required)",lexical
"fatal error: lambda expressions not supported in C++98",lexical
"Error: 'constexpr' keyword requires C++11 or later",lexical
"Line 45: range-based for loops require C++11",lexical
"main.cpp:67:12: error: 'decltype' was not declared in this scope",lexical
"main.cpp:15:5: error: expected unqualified-id before numeric constant",syntactic
"error C2059: syntax error : 'numeric constant'",syntactic
"Line 25: expected declaration before '}' token",syntactic
"compilation error: expected primary-expression before ')' token at line 45",syntactic
"[Error] main.cpp (67): expected declaration specifiers",syntactic
"fatal error: expected identifier or '(' before '&' token",syntactic
"main.cpp:89:8: error: expected ')' before '*' token",syntactic
"Error (line 123): expected ';' after member declaration",syntactic
"Line 156: expected class-name before '{' token",syntactic
"src/main.cpp:189: error: expected constructor destructor or type conversion",syntactic
"main.cpp:15:5: error: no matching function for call to 'std::vector<int>::push_back(std::string)'",semantic
"error C2664: cannot convert argument 1 from 'const char *' to 'int'",semantic
"Line 25: template argument deduction/substitution failed",semantic
"compilation error: no viable conversion from 'double' to 'int'",semantic
"[Error] Cannot resolve overloaded function 'calculate'",semantic
"main.cpp:45:10: error: ambiguous overload for 'operator+' (operand types are 'A' and 'B')",semantic
"fatal error: 'class MyClass' has no member named 'undefinedMethod'",semantic
"Line 67: redefinition of 'int variable' (previously declared at line 23)",semantic
"main.cpp:89:5: error: multiple definition of 'function'",semantic
"Error: circular dependency between classes A and B",semantic
"warning: unused variable 'temp' [-Wunused-variable]",semantic
"main.cpp:15:8: warning: comparison between signed and unsigned integer expressions",semantic
"Line 25: warning: deprecated conversion from string constant to 'char*'",semantic
"[Warning] Possible memory leak: pointer 'ptr' not freed",semantic
"compilation warning: unreachable code after return statement",semantic
"main.cpp:45:5: warning: control reaches end of non-void function",semantic
"Warning (line 67): variable 'count' may be used uninitialized",semantic
"runtime warning: potential buffer overflow detected",semantic
"main.cpp:89:12: warning: dereferencing type-punned pointer",semantic
"Line 123: warning: format '%d' expects argument type 'int' but got 'char*'",semantic
"undefined variable 'x'",lexical
"undeclared identifier 'count'",lexical
"variable 'temp' not declared in this scope",lexical
"identifier 'func' has not been declared",lexical
"undeclared variable 'index' used",lexical
"unknown identifier 'result'",lexical
"variable 'data' was not declared in this scope",lexical
"identifier 'value' not found",lexical
"undeclared function 'calculate'",lexical
"variable 'sum' not defined",lexical
"expected ';' before 'return'",syntactic
"missing closing bracket '}'",syntactic
"expected ')' before ';'",syntactic
"expected '{' before 'else'",syntactic
"missing semicolon at end of statement",syntactic
"expected '(' after 'if'",syntactic
"missing closing parenthesis ')'",syntactic
"expected ';' after expression",syntactic
"missing opening bracket '{'",syntactic
"expected ')' after 'for'",syntactic
"type mismatch: cannot assign string to integer",semantic
"function expects 2 arguments but 3 given",semantic
"cannot convert 'int' to 'string'",semantic
"division by zero error",semantic
"array index out of bounds",semantic
"null pointer dereference",semantic
"incompatible types in assignment",semantic
"function returns void but value expected",semantic
"cannot call method on null reference",semantic
"type 'int' has no member 'length'",semantic'''
        
        # Write to CSV file
        with open('cpp_error_dataset.csv', 'w', encoding='utf-8') as f:
            f.write(dataset_content)
        
        print("Dataset created successfully as 'cpp_error_dataset.csv'")
        return 'cpp_error_dataset.csv'
    
    def load_dataset(self, csv_file_path):
        """Load the dataset from CSV file"""
        try:
            if not os.path.exists(csv_file_path):
                print(f"Dataset file '{csv_file_path}' not found. Creating it...")
                csv_file_path = self.create_comprehensive_dataset()
            
            self.data = pd.read_csv(csv_file_path)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.data.shape}")
            print(f"\nClass distribution:")
            class_dist = self.data['error_type'].value_counts()
            print(class_dist)
            
            # Show sample data
            print(f"\nSample data:")
            print(self.data.head())
            
            return self.data
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_text(self, text):
        """Enhanced preprocessing for C++ error messages"""
        try:
            # Convert to lowercase
            text = str(text).lower()
            
            # Preserve important C++ keywords and symbols
            cpp_keywords = {'int', 'char', 'float', 'double', 'void', 'const', 'static', 
                           'class', 'struct', 'template', 'namespace', 'using', 'std',
                           'cout', 'cin', 'endl', 'vector', 'string', 'nullptr', 'auto'}
            
            # Replace file paths and line numbers with generic tokens
            text = re.sub(r'\\b\\w+\\.(cpp|h|hpp|c)\\b', 'FILENAME', text)
            text = re.sub(r':\\d+:\\d+', ':LINE:COL', text)
            text = re.sub(r'line\\s+\\d+', 'line NUMBER', text)
            text = re.sub(r'column\\s+\\d+', 'column NUMBER', text)
            
            # Normalize quotes and brackets
            text = re.sub(r"'[^']*'", 'QUOTED_IDENTIFIER', text)
            text = re.sub(r'"[^"]*"', 'STRING_LITERAL', text)
            
            # Keep programming symbols but clean others
            text = re.sub(r'[^a-zA-Z0-9\\s\\(\\)\\{\\}\\[\\]\\;\\:\\.\\,\\_\\-\\<\\>]', ' ', text)
            text = re.sub(r'\\s+', ' ', text).strip()
            
            # Tokenization with fallback
            try:
                tokens = word_tokenize(text)
            except LookupError:
                tokens = text.split()
            
            # Enhanced stopword filtering
            programming_terms = {'error', 'expected', 'missing', 'undefined', 'undeclared',
                               'identifier', 'variable', 'function', 'type', 'cannot',
                               'invalid', 'before', 'after', 'declaration', 'statement',
                               'expression', 'token', 'syntax', 'semantic', 'lexical',
                               'warning', 'fatal', 'compilation'}
            
            filtered_tokens = []
            for token in tokens:
                if (token not in self.stop_words or 
                    token in programming_terms or 
                    token in cpp_keywords or
                    len(token) <= 3):  # Keep short tokens that might be important
                    filtered_tokens.append(token)
            
            # Lemmatization with fallback
            try:
                processed_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            except LookupError:
                processed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return str(text).lower()
    
    def prepare_features(self):
        """Prepare features and labels for training"""
        if self.data is None:
            raise ValueError("No data loaded. Please load dataset first.")
        
        print("Preprocessing text data...")
        self.data['processed_text'] = self.data['error_message'].apply(self.preprocess_text)
        
        # Remove any rows with empty processed text
        initial_size = len(self.data)
        self.data = self.data[self.data['processed_text'].str.strip() != '']
        final_size = len(self.data)
        
        if initial_size != final_size:
            print(f"Removed {initial_size - final_size} rows with empty processed text")
        
        # Prepare features and labels
        X = self.data['processed_text']
        y = self.data['error_type']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Final dataset size: {len(X)}")
        
        return X, y_encoded
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models and compare performance"""
        print(f"Training models with {len(X)} samples...")
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        print(f"Training: {len(X_train)}, Test: {len(X_test)}")
        print(f"Class distribution in training:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  {self.label_encoder.inverse_transform([cls])[0]}: {count}")
        
        # Enhanced TF-IDF with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85,
            stop_words='english' if self.stop_words else None,
            sublinear_tf=True,
            norm='l2'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        
        # Enhanced model selection with better hyperparameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=25, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=random_state
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
                multi_class='ovr',
                class_weight='balanced'
            )
        }
        
        # Use stratified k-fold for better cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        results = {}
        
        for name, model in models.items():
            print(f"\\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train_tfidf, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_tfidf)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Stratified cross-validation
                try:
                    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=skf, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as e:
                    print(f"Cross-validation failed for {name}: {e}")
                    cv_mean, cv_std = accuracy, 0.0
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred
                }
                
                print(f"{name} - Test Accuracy: {accuracy:.4f}")
                print(f"{name} - CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not results:
            raise RuntimeError("All models failed to train")
        
        # Select best model based on CV score
        self.best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\\nBest model: {self.best_model_name}")
        print(f"Best CV Score: {results[self.best_model_name]['cv_mean']:.4f}")
        
        # Detailed evaluation
        self.evaluate_detailed(y_test, results[self.best_model_name]['predictions'])
        
        self.models = results
        self.X_test = X_test_tfidf
        self.y_test = y_test
        
        return results
    
    def evaluate_detailed(self, y_true, y_pred):
        """Enhanced evaluation with more metrics"""
        print(f"\\n=== Detailed Evaluation for {self.best_model_name} ===")
        
        # Classification report
        target_names = self.label_encoder.classes_
        print("\\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class accuracy
        print("\\nPer-class Accuracy:")
        for i, class_name in enumerate(target_names):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_pred == y_true) & class_mask) / np.sum(class_mask)
                print(f"{class_name}: {class_accuracy:.4f}")
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def predict_error_type(self, error_message):
        """Predict error type for a new error message"""
        if self.best_model is None or self.vectorizer is None:
            print("Model not trained yet. Please train the model first.")
            return None, None
        
        # Preprocess the input
        processed_message = self.preprocess_text(error_message)
        
        if not processed_message.strip():
            print("Warning: Input message resulted in empty processed text")
            return None, None
        
        # Vectorize
        message_tfidf = self.vectorizer.transform([processed_message])
        
        # Predict
        prediction = self.best_model.predict(message_tfidf)[0]
        prediction_proba = self.best_model.predict_proba(message_tfidf)[0]
        
        # Decode prediction
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores
        confidence_dict = dict(zip(
            self.label_encoder.classes_, 
            prediction_proba
        ))
        
        return predicted_class, confidence_dict
    
    def save_model(self, filepath='cpp_error_classifier.pkl'):
        """Save the trained model and components"""
        if self.best_model is None:
            print("No trained model to save")
            return
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'models': self.models
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath='cpp_error_classifier.pkl'):
        """Load a pre-trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_model = model_data['best_model']
            self.best_model_name = model_data.get('best_model_name', 'Unknown')
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.models = model_data.get('models', {})
            
            print(f"Model loaded from {filepath}")
            print(f"Best model: {self.best_model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")

def main():
    """Main function to run the complete classifier"""
    try:
        print("=== C++ Error Classifier ===")
        print("Initializing classifier...")
        
        # Initialize classifier
        classifier = CppErrorClassifier()
        
        # Load dataset
        print("\\nLoading dataset...")
        data = classifier.load_dataset('cpp_error_dataset.csv')
        
        if data is None:
            print("Failed to load dataset. Exiting.")
            return
        
        # Prepare features
        print("\\nPreparing features...")
        X, y = classifier.prepare_features()
        
        # Train models
        print("\\nTraining models...")
        results = classifier.train_models(X, y)
        
        # Test with various examples
        test_messages = [
            "main.cpp:45:12: error: 'undefined_var' was not declared in this scope",
            "error C2143: syntax error: missing ';' before 'return'",
            "Line 67: cannot convert 'string' to 'int' in assignment",
            "fatal error: 'iostream' file not found",
            "runtime error: array index out of bounds",
            "expected ')' after function arguments",
            "undefined reference to 'external_function'",
            "template argument deduction failed",
            "pure virtual function call detected"
        ]
        
        print("\\n=== Testing Predictions ===")
        for i, msg in enumerate(test_messages, 1):
            print(f"\\n{i}. Testing: {msg[:60]}...")
            predicted_type, confidence = classifier.predict_error_type(msg)
            
            if predicted_type:
                print(f"   Predicted: {predicted_type}")
                print(f"   Confidence: {max(confidence.values()):.4f}")
                print(f"   All probabilities: {confidence}")
            else:
                print("   Prediction failed")
            print("-" * 70)
        
        # Save the trained model
        print("\\nSaving model...")
        classifier.save_model()
        
        print("\\n=== Training Complete ===")
        print(f"Best model: {classifier.best_model_name}")
        print(f"Dataset size: {len(data)} examples")
        print(f"Features: {classifier.vectorizer.get_feature_names_out().shape[0] if classifier.vectorizer else 'Unknown'}")
        print("Model saved as 'cpp_error_classifier.pkl'")
        
        # Demonstrate loading saved model
        print("\\n=== Testing Model Loading ===")
        new_classifier = CppErrorClassifier()
        new_classifier.load_model()
        
        test_msg = "Line 25: expected semicolon before return statement"
        pred, conf = new_classifier.predict_error_type(test_msg)
        print(f"Loaded model prediction for '{test_msg}': {pred} ({max(conf.values()):.4f})")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()