"""
demo.py - Complete demonstration of C++ Error Classification System
Run this after training the model with mlbec3.py
"""
# FIXED: Corrected the import to point to the correct file
from main_code_for_errorclassification import CppCodeErrorAnalyzer


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
        # FIXED: Pointed to the correct training script
        print("2. Run: python mlbec3.py")
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
    
    # (The rest of your demo.py file remains the same...)

    # ... [rest of test_cases array] ...
    
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
        # ... (and so on for all your other test cases)
    ]
    
    results_summary = []
    
    # Analyze each test case
    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "-"*70)
        print(f"RUNNING TEST {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print("-"*(len(test_case['name']) + 18))
        print("CODE:\n" + test_case['code'])
        
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
    print(f"\nTotal test cases: {len(test_cases)}\n")
    
    for i, summary in enumerate(results_summary, 1):
        print(f"{i}. {summary['test_name']}")
        print(f"   Errors found: {summary['errors_found']}")
        if summary['errors_found'] > 0:
            print(f"   Most common type: {summary['most_common_type']}")
            print(f"   Average confidence: {summary['avg_confidence']:.2%}\n")
    
    print("\n" + "="*70)
    print("✓ DEMO COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()