#include <iostream>
using namespace std;

int main() {
    int x = 5  // Missing semicolon
    string y = "hello";
    x = y;  // Type mismatch
    cout << undeclared_var << endl;  // Undeclared variable
    return 0;
}