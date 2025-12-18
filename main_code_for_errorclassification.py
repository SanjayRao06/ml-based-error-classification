# main_code_for_errorclassification.py
"""
Demonstration script for the CppCodeErrorAnalyzer.
The class itself is now defined in analyzers.py

This script should continue to work as the public 'analyze'
method signature has not changed.
"""

# Import the class from its new location
# This will also import 'error_classifier.py' and check NLTK data
from analyzers import CppCodeErrorAnalyzer, PythonCodeErrorAnalyzer
import sys

# ... [All the helper functions for printing] ...

def print_analysis_results(results: dict, language: str):
    """Print analysis results in a formatted way"""
    print("\n" + "="*70)
    print(f"{language.upper()} CODE ANALYSIS RESULTS")
    print("="*70)

    if not results['success']:
        print(
            f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        return

    success_key = "compilation_success" if language == "C++" else "execution_success"
    success_label = "Compilation" if language == "C++" else "Execution"
    
    print(
        f"✅ {success_label} Success: {'Yes' if results[success_key] else 'No'}")
    print(f"📊 Total Errors/Warnings: {results['total_errors']}")

    if results['total_errors'] == 0 and results[success_key]:
        print(f"\n🎉 No errors found! Your code {success_label.lower()}s successfully.")
        return

    print("\n" + "-"*50)
    print("ERROR CLASSIFICATIONS:")
    print("~"*50)

    for cls in results['classifications']:
        error_num = cls['error_number']
        error_msg = cls['error_message'].replace('\n', ' ')
        error_type = cls['predicted_type']
        confidence = cls['confidence']

        print(f"\n[ Error {error_num} ]")
        print(f"  Message:    {error_msg[:100]}...")
        print(f"  Prediction: {error_type.upper()} (Confidence: {confidence:.2%})")

    print("\n" + "-"*50)
    print("SUMMARY:")
    print("~"*50)
    for err_type, count in results['summary'].items():
        print(f"  {err_type.capitalize()}: {count}")

    print("\n" + "-"*50)
    print("RAW STDERR:")
    print("~"*50)
    print(results.get('raw_stderr', 'N/A'))
    print("="*70)


def run_cpp_tests(analyzer: CppCodeErrorAnalyzer):
    """Runs a series of C++ test cases."""
    print("\n--- RUNNING C++ TEST CASES ---")
    
    test_cases = [
        {
            'name': 'Lexical Error Example (Undeclared Identifier)',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5;
    cout << undeclared_variable << endl;  // Lexical/Semantic
    return 0;
}
'''
        },
        {
            'name': 'Syntactic Error Example (Missing Semicolon)',
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
            'name': 'Semantic Error Example (Type Mismatch)',
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
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        results = analyzer.analyze(test_case['code'])
        print_analysis_results(results, "C++")


def run_python_tests(analyzer: PythonCodeErrorAnalyzer):
    """Runs a series of Python test cases."""
    print("\n--- RUNNING PYTHON TEST CASES ---")

    test_cases = [
        {
            'name': 'NameError (Undeclared Variable)',
            'code': '''
x = 5
print(y)  # NameError
'''
        },
        {
            'name': 'SyntaxError (Missing Parenthesis)',
            'code': '''
print("hello"   # SyntaxError
'''
        },
        {
            'name': 'TypeError (Wrong Type Operation)',
            'code': '''
x = 5
y = "hello"
print(x + y)  # TypeError
'''
        },
        {
            'name': 'No Errors Example',
            'code': '''
x = 5
y = "hello"
print(f"x is {x} and y is {y}")
'''
        }
    ]
    
    # Analyze each test case
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        results = analyzer.analyze(test_case['code'])
        print_analysis_results(results, "Python")


# --- Main execution ---

if __name__ == "__main__":
    
    # --- C++ Analyzer Test ---
    print("Initializing C++ Analyzer...")
    cpp_analyzer = CppCodeErrorAnalyzer()
    
    if not cpp_analyzer.compiler_available:
        print("\n❌ g++ compiler not found. Skipping C++ tests.")
    elif cpp_analyzer.classifier is None:
        print(f"\n❌ C++ model '{cpp_analyzer.model_path}' not loaded. Skipping C++ tests.")
        print("   Please run 'python mlbec3.py' to train the model.")
    else:
        print(f"C++ model '{cpp_analyzer.classifier.best_model_name}' loaded.")
        run_cpp_tests(cpp_analyzer)

    # --- Python Analyzer Test ---
    print("\nInitializing Python Analyzer...")
    py_analyzer = PythonCodeErrorAnalyzer()
    
    if py_analyzer.classifier is None:
        print(f"\n❌ Python model '{py_analyzer.model_path}' not loaded. Skipping Python tests.")
        print("   Please run 'python mlbec3.py' to train the model.")
    else:
        print(f"Python model '{py_analyzer.classifier.best_model_name}' loaded.")
        run_python_tests(py_analyzer)
    
    print("\n--- Demo script finished. ---")