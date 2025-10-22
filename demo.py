"""
demo.py - Complete demonstration of C++ Error Classification System
Run this after training the model with ML_Based_Error_Classification.py
"""
from CppCodeErrorAnalyzer import CppCodeErrorAnalyzer


def main():
    """Run comprehensive demo of the error classification system"""
    
    print("\n" + "="*70)
    print("C++ ERROR CLASSIFICATION SYSTEM - DEMO")
    print("="*70)
    
    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = CppCodeErrorAnalyzer()
    
    # Check prerequisites
    if not analyzer.classifier:
        print("\n❌ ERROR: Model not loaded!")
        print("\nPlease run these steps first:")
        print("1. Ensure you have 'cpp_error_dataset.csv' with your training data")
        print("2. Run: python ML_Based_Error_Classification.py")
        print("3. Then run this demo script again")
        return
    
    if not analyzer.compiler_available:
        print("\n❌ ERROR: g++ compiler not found!")
        print("\nPlease install g++ compiler:")
        print("  - Ubuntu/Debian: sudo apt-get install g++")
        print("  - macOS: xcode-select --install")
        print("  - Windows: Install MinGW-w64 or use WSL")
        return
    
    print("✓ All prerequisites satisfied!\n")
    
    # Test cases covering all error types
    test_cases = [
        {
            'name': 'Lexical Error - Undeclared Variable',
            'description': 'Using a variable that was never declared',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5;
    cout << undeclared_variable << endl;
    return 0;
}
'''
        },
        {
            'name': 'Syntactic Error - Missing Semicolon',
            'description': 'Missing semicolon at end of statement',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5
    cout << x << endl;
    return 0;
}
'''
        },
        {
            'name': 'Syntactic Error - Unmatched Braces',
            'description': 'Missing closing brace',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5;
    if (x > 0) {
        cout << "Positive" << endl;
    
    return 0;
}
'''
        },
        {
            'name': 'Semantic Error - Type Mismatch',
            'description': 'Attempting to assign string to int',
            'code': '''
#include <iostream>
#include <string>
using namespace std;

int main() {
    int x = 5;
    string y = "hello";
    x = y;
    cout << x << endl;
    return 0;
}
'''
        },
        {
            'name': 'Semantic Error - Wrong Function Arguments',
            'description': 'Calling function with wrong argument types',
            'code': '''
#include <iostream>
#include <cmath>
using namespace std;

int main() {
    string text = "hello";
    double result = sqrt(text);
    cout << result << endl;
    return 0;
}
'''
        },
        {
            'name': 'Multiple Errors - Mixed Types',
            'description': 'Multiple different types of errors',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5
    string y = "hello";
    x = y;
    cout << undefined_var << endl;
    return 0;
}
'''
        },
        {
            'name': 'Lexical Error - Undefined Function',
            'description': 'Calling a function that does not exist',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int result = undefined_function(5);
    cout << result << endl;
    return 0;
}
'''
        },
        {
            'name': 'Valid Code - No Errors',
            'description': 'Correctly written C++ code',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int x = 5;
    int y = 10;
    cout << "Sum: " << (x + y) << endl;
    return 0;
}
'''
        }
    ]
    
    # Run analysis on each test case
    results_summary = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'='*70}")
        print(f"Description: {test_case['description']}")
        
        # Show the code
        print("\nCODE:")
        print("-" * 50)
        print(test_case['code'])
        print("-" * 50)
        
        # Analyze the code
        results = analyzer.analyze_code(test_case['code'])
        analyzer.print_analysis_results(results)
        
        # Store summary
        if results['success']:
            results_summary.append({
                'test_name': test_case['name'],
                'errors_found': results['total_errors'],
                'most_common_type': results['summary'].get('most_common_type', 'none'),
                'avg_confidence': results['summary'].get('average_confidence', 0.0)
            })
    
    # Print overall summary
    print("\n" + "="*70)
    print("OVERALL DEMO SUMMARY")
    print("="*70)
    print(f"\nTotal test cases: {len(test_cases)}")
    print("\nResults:")
    
    for i, summary in enumerate(results_summary, 1):
        print(f"\n{i}. {summary['test_name']}")
        print(f"   Errors found: {summary['errors_found']}")
        if summary['errors_found'] > 0:
            print(f"   Most common type: {summary['most_common_type']}")
            print(f"   Average confidence: {summary['avg_confidence']:.2%}")
    
    print("\n" + "="*70)
    print("✓ DEMO COMPLETE!")
    print("="*70)
    print("\nYou can now use the CppCodeErrorAnalyzer class to:")
    print("1. Analyze any C++ code string: analyzer.analyze_code(code)")
    print("2. Analyze C++ files: analyzer.analyze_file('path/to/file.cpp')")
    print("3. Get detailed error classifications with confidence scores")


if __name__ == "__main__":
    main()