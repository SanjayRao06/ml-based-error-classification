import subprocess
import os
import tempfile
import re
from pathlib import Path
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Optional

class CppCodeErrorAnalyzer:
    def __init__(self, model_path='cpp_error_classifier.pkl'):
        """Initialize the analyzer with a trained model"""
        self.classifier = None
        self.model_path = model_path
        self.load_classifier()
    
    def load_classifier(self):
        """Load the trained error classification model"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a simple classifier object to hold the components
            class SimpleClassifier:
                def __init__(self, model_data):
                    self.best_model = model_data['best_model']
                    self.vectorizer = model_data['vectorizer']
                    self.label_encoder = model_data['label_encoder']
                    self.best_model_name = model_data.get('best_model_name', 'Unknown')
                
                def predict_error_type(self, error_message):
                    """Predict error type for a message"""
                    # Simple preprocessing (you might want to use the full preprocessing)
                    processed = error_message.lower()
                    
                    # Vectorize
                    message_tfidf = self.vectorizer.transform([processed])
                    
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
            
            self.classifier = SimpleClassifier(model_data)
            print(f"Model loaded successfully: {self.classifier.best_model_name}")
            
        except FileNotFoundError:
            print(f"Model file '{self.model_path}' not found.")
            print("Please run the training script first to create the model.")
            self.classifier = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.classifier = None
    
    def compile_cpp_code(self, code: str, compiler='g++', std='c++17') -> Tuple[bool, List[str]]:
        """
        Compile C++ code and capture error messages
        
        Args:
            code: C++ source code as string
            compiler: Compiler to use (g++, clang++, etc.)
            std: C++ standard to use
        
        Returns:
            (success, error_messages): Compilation success and list of error messages
        """
        # Create a temporary file for the C++ code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as temp_file:
            temp_file.write(code)
            temp_cpp_path = temp_file.name
        
        try:
            # Create temporary executable path
            temp_exe_path = temp_cpp_path.replace('.cpp', '.exe')
            
            # Compile command
            compile_cmd = [
                compiler,
                f'-std={std}',
                '-o', temp_exe_path,
                temp_cpp_path
            ]
            
            # Run compilation
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Parse error messages
            error_messages = []
            if result.returncode != 0:
                # Combine stdout and stderr
                full_output = (result.stdout + '\n' + result.stderr).strip()
                
                # Split into individual error lines
                lines = full_output.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and ('error' in line.lower() or 'warning' in line.lower()):
                        error_messages.append(line)
            
            success = result.returncode == 0
            
        except subprocess.TimeoutExpired:
            error_messages = ["Compilation timeout - code took too long to compile"]
            success = False
        except FileNotFoundError:
            error_messages = [f"Compiler '{compiler}' not found. Please install {compiler}."]
            success = False
        except Exception as e:
            error_messages = [f"Compilation error: {str(e)}"]
            success = False
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_cpp_path)
                if os.path.exists(temp_exe_path):
                    os.unlink(temp_exe_path)
            except:
                pass
        
        return success, error_messages
    
    def analyze_code(self, code: str, compiler='g++') -> Dict:
        """
        Analyze C++ code and classify any errors found
        
        Args:
            code: C++ source code as string
            compiler: Compiler to use
        
        Returns:
            Dictionary with analysis results
        """
        if not self.classifier:
            return {
                'success': False,
                'error': 'Model not loaded. Please train the model first.',
                'compilation_success': False,
                'errors': [],
                'classifications': []
            }
        
        # Compile the code
        print("Compiling C++ code...")
        compilation_success, error_messages = self.compile_cpp_code(code, compiler)
        
        # Analyze each error message
        classifications = []
        if error_messages:
            print(f"Found {len(error_messages)} error/warning messages")
            
            for i, error_msg in enumerate(error_messages, 1):
                try:
                    predicted_type, confidence = self.classifier.predict_error_type(error_msg)
                    
                    classifications.append({
                        'error_number': i,
                        'error_message': error_msg,
                        'predicted_type': predicted_type,
                        'confidence': max(confidence.values()),
                        'all_probabilities': confidence
                    })
                    
                except Exception as e:
                    classifications.append({
                        'error_number': i,
                        'error_message': error_msg,
                        'predicted_type': 'unknown',
                        'confidence': 0.0,
                        'error': str(e)
                    })
        
        return {
            'success': True,
            'compilation_success': compilation_success,
            'total_errors': len(error_messages),
            'errors': error_messages,
            'classifications': classifications,
            'summary': self._generate_summary(classifications)
        }
    
    def _generate_summary(self, classifications: List[Dict]) -> Dict:
        """Generate a summary of error types found"""
        if not classifications:
            return {'total': 0, 'by_type': {}}
        
        type_counts = {}
        total_confidence = 0
        
        for cls in classifications:
            error_type = cls.get('predicted_type', 'unknown')
            confidence = cls.get('confidence', 0)
            
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(classifications) if classifications else 0
        
        return {
            'total': len(classifications),
            'by_type': type_counts,
            'average_confidence': avg_confidence,
            'most_common_type': max(type_counts, key=type_counts.get) if type_counts else 'none'
        }
    
    def analyze_file(self, file_path: str, compiler='g++') -> Dict:
        """Analyze a C++ file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.analyze_code(code, compiler)
        except Exception as e:
            return {
                'success': False,
                'error': f'Error reading file: {str(e)}',
                'compilation_success': False,
                'errors': [],
                'classifications': []
            }
    
    def print_analysis_results(self, results: Dict):
        """Print analysis results in a formatted way"""
        print("\n" + "="*70)
        print("C++ CODE ANALYSIS RESULTS")
        print("="*70)
        
        if not results['success']:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        print(f"✅ Compilation Success: {'Yes' if results['compilation_success'] else 'No'}")
        print(f"📊 Total Errors/Warnings: {results['total_errors']}")
        
        if results['total_errors'] == 0:
            print("🎉 No errors found! Your code compiles successfully.")
            return
        
        print("\n" + "-"*50)
        print("ERROR CLASSIFICATIONS:")
        print("-"*50)
        
        for cls in results['classifications']:
            error_num = cls['error_number']
            error_msg = cls['error_message']
            error_type = cls['predicted_type']
            confidence = cls['confidence']
            
            # Truncate long error messages
            if len(error_msg) > 80:
                error_msg = error_msg[:77] + "..."
            
            print(f"\n{error_num}. {error_msg}")
            print(f"   🏷️  Type: {error_type.upper()}")
            print(f"   📈 Confidence: {confidence:.2%}")
            
            if 'all_probabilities' in cls:
                probs = cls['all_probabilities']
                print(f"   📊 All probabilities:")
                for error_type_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    print(f"      {error_type_name}: {prob:.2%}")
        
        # Summary
        summary = results['summary']
        print("\n" + "-"*50)
        print("SUMMARY:")
        print("-"*50)
        print(f"📋 Total errors analyzed: {summary['total']}")
        print(f"🎯 Average confidence: {summary['average_confidence']:.2%}")
        print(f"🏆 Most common error type: {summary['most_common_type']}")
        print("\nError type breakdown:")
        for error_type, count in summary['by_type'].items():
            print(f"   {error_type}: {count}")

# Example usage and testing
def main():
    """Demonstrate the complete system"""
    
    # Initialize analyzer
    analyzer = CppCodeErrorAnalyzer()
    
    if not analyzer.classifier:
        print("Please run the training script first to create the model!")
        return
    
    # Test cases with different types of errors
    test_codes = [
        {
            'name': 'Lexical Error Example',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5;
    cout << undeclared_variable << endl;  // Lexical error
    return 0;
}
'''
        },
        {
            'name': 'Syntactic Error Example',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5  // Missing semicolon - Syntactic error
    cout << x << endl;
    return 0;
}
'''
        },
        {
            'name': 'Semantic Error Example',
            'code': '''
#include <iostream>
#include <string>
using namespace std;

int main() {
    int x = 5;
    string y = "hello";
    x = y;  // Type mismatch - Semantic error
    cout << x << endl;
    return 0;
}
'''
        },
        {
            'name': 'Multiple Errors Example',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5  // Missing semicolon
    string y = "hello";
    x = y;  // Type mismatch
    cout << undeclared_var << endl;  // Undeclared variable
    return 0
}  // Missing semicolon
'''
        },
        {
            'name': 'No Errors Example',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5;
    cout << "Hello, World! x = " << x << endl;
    return 0;
}
'''
        }
    ]
    
    # Analyze each test case
    for i, test_case in enumerate(test_codes, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'='*70}")
        
        # Show the code
        print("CODE:")
        print("-" * 30)
        print(test_case['code'])
        
        # Analyze the code
        results = analyzer.analyze_code(test_case['code'])
        analyzer.print_analysis_results(results)
        
        print("\n" + "="*70)
    
    print("\nDemo complete! You can now use this system with any C++ code.")

if __name__ == "__main__":
    main()