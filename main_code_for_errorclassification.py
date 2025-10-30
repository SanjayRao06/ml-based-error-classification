# main_code_for_errorclassification.py
# This file bridges the CppErrorClassifier (ML model) with the g++ compiler.

import subprocess
import os
import re
import pickle
import numpy as np

# *** Imports the CppErrorClassifier from mlbec4.py ***
from mlbec4 import CppErrorClassifier

class CppCodeErrorAnalyzer:
    """
    Handles C++ code compilation, captures compiler output, and uses 
    the CppErrorClassifier to classify each error message.
    """
    def __init__(self, model_path='cpp_error_classifier.pkl', data_path='cpp_error_dataset.csv'):
        self.model_path = model_path
        self.data_path = data_path
        self.classifier = CppErrorClassifier()
        self.load_or_train_model()

    def check_compiler_availability(self):
        """Checks if the 'g++' compiler is available in the system's PATH."""
        try:
            # Check g++ version to confirm availability
            subprocess.run(['g++', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def load_or_train_model(self):
        """Loads the pre-trained model or trains it if the file is missing."""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            try:
                self.classifier.load_model(self.model_path)
                return True
            except Exception as e:
                print(f"Error loading model: {e}. Retrying with training.")
        
        # Training logic if model load failed or file not found
        print("Model file not found or failed to load. Initiating training...")
        data = self.classifier.load_dataset(self.data_path)
        if data is None or len(data) < 10:
            print("Cannot train: Dataset is missing or too small.")
            return False

        try:
            X, y = self.classifier.prepare_features()
            self.classifier.train_models(X, y)
            self.classifier.save_model(self.model_path)
            print("Training complete and new model saved.")
            return True
        except Exception as e:
            print(f"Error during training: {e}")
            return False

    def compile_and_analyze(self, code_string: str) -> dict:
        """
        1. Writes code to a temporary file.
        2. Compiles the code using g++.
        3. Parses the output and uses the ML classifier for analysis.
        4. Cleans up temporary files.
        """
        temp_filename = 'temp_code.cpp'
        exec_filename = 'a.out' if os.name != 'nt' else 'a.exe'
        
        try:
            # 1. Write to temp file
            with open(temp_filename, 'w') as f:
                f.write(code_string)

            # 2. Compile the code using g++
            compile_command = ['g++', temp_filename, '-o', exec_filename, '-std=c++17', '-Wall']
            result = subprocess.run(compile_command, capture_output=True, text=True, timeout=10)
            
            compiler_output = result.stdout + result.stderr
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'compilation_success': True,
                    'message': "Compilation Successful.",
                    'output': compiler_output
                }

            # 3. Compilation failed, parse errors
            return self._parse_and_classify_errors(compiler_output)

        except FileNotFoundError:
            return {
                'success': False,
                'error': "g++ compiler not found in PATH."
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': "Compilation timed out (max 10 seconds)."
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"An unexpected error occurred: {str(e)}"
            }
        finally:
            # 4. Cleanup
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            if os.path.exists(exec_filename):
                os.remove(exec_filename)


    def _parse_and_classify_errors(self, output: str) -> dict:
        """Parses compiler output into individual messages and classifies them."""
        
        # Regex to capture individual error/warning lines.
        error_pattern = re.compile(r'^(.*?:\d+:\d+:\s*(?:error|warning|fatal error):.*|.*?(?:error|warning|fatal error):.*)', re.MULTILINE | re.IGNORECASE)
        
        # Split output into individual messages based on the pattern
        messages = error_pattern.findall(output)
        
        analysis_results = []
        total_errors = 0
        total_warnings = 0
        type_counts = {'lexical': 0, 'syntactic': 0, 'semantic': 0, 'other': 0}
        
        for msg in messages:
            # Clean up the message slightly for better ML input
            cleaned_msg = msg.strip()
            
            # Predict the error type
            predicted_type, confidences = self.classifier.predict_error_type(cleaned_msg)
            
            # Tally counts
            if 'error' in cleaned_msg.lower() or 'fatal' in cleaned_msg.lower():
                total_errors += 1
            elif 'warning' in cleaned_msg.lower():
                total_warnings += 1
                
            if predicted_type:
                type_counts[predicted_type] += 1
            else:
                type_counts['other'] += 1 # Should rarely happen if classification is good

            # Prepare confidence for display
            top_confidence = f"{max(confidences.values()):.2%}" if confidences else "N/A"
            
            analysis_results.append({
                'message': cleaned_msg,
                'predicted_type': predicted_type,
                'confidence': top_confidence,
                'all_confidences': confidences
            })

        return {
            'success': True,
            'compilation_success': False,
            'total_errors': total_errors + total_warnings,
            'errors': analysis_results,
            'summary': type_counts,
            'raw_compiler_output': output
        }

# If this file is run directly (for local testing/training)
if __name__ == '__main__':
    print("--- C++ Error Analyzer Test ---")
    analyzer = CppCodeErrorAnalyzer()
    
    # Test code with a simple syntactic error
    test_code = """
    #include <iostream>
    int main() {
        std::cout << "Hello" 
        return 0;
    } 
    """
    
    if analyzer.check_compiler_availability():
        results = analyzer.compile_and_analyze(test_code)
        print("\n--- Analysis Results ---")
        print(results)
    else:
        print("\nSkipping live test: g++ not found.")
