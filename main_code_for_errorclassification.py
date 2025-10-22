"""
CppCodeErrorAnalyzer.py - FIXED VERSION
Analyzes C++ code and classifies compilation errors
"""
import subprocess
import os
import tempfile
import shutil
import pickle
from typing import List, Dict, Tuple, Optional


class CppCodeErrorAnalyzer:
    def __init__(self, model_path='cpp_error_classifier.pkl'):
        """Initialize the analyzer with a trained model"""
        self.classifier = None
        self.model_path = model_path
        self.compiler_available = self._check_compiler()
        self.load_classifier()
    
    def _check_compiler(self) -> bool:
        """Check if g++ compiler is available"""
        return shutil.which('g++') is not None
    
    def load_classifier(self):
        """Load the trained error classification model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file '{self.model_path}' not found.")
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a simple classifier wrapper
            class SimpleClassifier:
                def __init__(self, model_data):
                    self.best_model = model_data['best_model']
                    self.vectorizer = model_data['vectorizer']
                    self.label_encoder = model_data['label_encoder']
                    self.best_model_name = model_data.get('best_model_name', 'Unknown')
                
                def predict_error_type(self, error_message):
                    """Predict error type for a message"""
                    # Preprocess (simplified)
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
            print(f"✓ Model loaded successfully: {self.classifier.best_model_name}")
            
            if not self.compiler_available:
                print("\n⚠️  WARNING: g++ compiler not found!")
                print("Install g++ to compile and analyze C++ code:")
                print("  - Ubuntu/Debian: sudo apt-get install g++")
                print("  - macOS: xcode-select --install")
                print("  - Windows: Install MinGW or use WSL")
            
        except FileNotFoundError:
            print(f"✗ Model file '{self.model_path}' not found.")
            print("Please run ML_Based_Error_Classification.py first to train the model.")
            self.classifier = None
        except Exception as e:
            print(f"✗ Error loading model: {e}")
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
        if not self.compiler_available:
            return False, [f"Compiler '{compiler}' not found. Please install {compiler}."]
        
        # Create a temporary file for the C++ code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_cpp_path = temp_file.name
        
        try:
            # Create temporary executable path
            temp_exe_path = temp_cpp_path.replace('.cpp', '')
            if os.name == 'nt':  # Windows
                temp_exe_path += '.exe'
            
            # Compile command
            compile_cmd = [
                compiler,
                f'-std={std}',
                '-o', temp_exe_path,
                temp_cpp_path,
                '-Wall'  # Enable all warnings
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
                
                # Split into lines and filter for errors/warnings
                lines = full_output.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for actual error/warning messages
                    if line and ('error:' in line.lower() or 'warning:' in line.lower()):
                        # Clean up the line
                        # Remove file path prefix to get just the error message
                        if ':' in line:
                            parts = line.split(':', 3)
                            if len(parts) >= 4:
                                # Format: file:line:col: error: message
                                error_messages.append(parts[3].strip())
                            elif len(parts) >= 2:
                                error_messages.append(':'.join(parts[1:]).strip())
                            else:
                                error_messages.append(line)
                        else:
                            error_messages.append(line)
                
                # If no structured errors found, add raw output
                if not error_messages and full_output:
                    error_messages.append(full_output)
            
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
                if os.path.exists(temp_cpp_path):
                    os.unlink(temp_cpp_path)
                if 'temp_exe_path' in locals() and os.path.exists(temp_exe_path):
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
            print(f"Found {len(error_messages)} error/warning message(s)\n")
            
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
        else:
            print("No errors found!")
        
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
            return {'total': 0, 'by_type': {}, 'average_confidence': 0.0, 'most_common_type': 'none'}
        
        type_counts = {}
        total_confidence = 0
        
        for cls in classifications:
            error_type = cls.get('predicted_type', 'unknown')
            confidence = cls.get('confidence', 0)
            
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(classifications) if classifications else 0
        most_common = max(type_counts, key=type_counts.get) if type_counts else 'none'
        
        return {
            'total': len(classifications),
            'by_type': type_counts,
            'average_confidence': avg_confidence,
            'most_common_type': most_common
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
            print("\n🎉 No errors found! Your code compiles successfully.")
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
            if len(error_msg) > 100:
                error_msg = error_msg[:97] + "..."
            
            print(f"\n{error_num}. {error_msg}")
            print(f"   🏷️  Type: {error_type.upper()}")
            print(f"   📈 Confidence: {confidence:.2%}")
            
            if 'all_probabilities' in cls:
                probs = cls['all_probabilities']
                print(f"   📊 All probabilities:")
                for type_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    print(f"      {type_name}: {prob:.2%}")
        
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
        print("\n✗ Please run ML_Based_Error_Classification.py first to create the model!")
        return
    
    if not analyzer.compiler_available:
        print("\n✗ g++ compiler not installed. Please install g++ to use this analyzer.")
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
    
    print("\n✓ Demo complete! You can now use this system with any C++ code.")


if __name__ == "__main__":
    main()