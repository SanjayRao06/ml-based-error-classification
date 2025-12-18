import csv
import random

# ============================================================
# CONFIGURATION
# ============================================================
NUM_ENTRIES = 750
OUTPUT_FILE = "python_error_dataset.csv"

# ============================================================
# ERROR PATTERNS (TEMPLATES)
# ============================================================

lexical_errors = [
    "SyntaxError: invalid character in identifier",
    "IndentationError: unexpected indent",
    "TabError: inconsistent use of tabs and spaces in indentation",
    "UnicodeError: unicodeescape codec can't decode bytes",
    "SyntaxError: EOL while scanning string literal",
    "ImportError: cannot import name '{name}' from '{module}'",
    "ModuleNotFoundError: No module named '{module}'",
    "SyntaxError: invalid decimal literal near '{token}'",
    "NameError: name '{var}' is not defined",
    "KeyError: '{key}' not found in mapping",
    "AttributeError: module '{module}' has no attribute '{attr}'",
    "TypeError: '{obj}' object is not callable",
    "ImportError: cannot import '{name}' from '{module}'",
    "SyntaxError: unterminated string literal at line {line}",
    "UnicodeEncodeError: 'ascii' codec can't encode character",
    "SyntaxError: invalid syntax near '{token}'",
    "IndentationError: unexpected unindent at line {line}",
    "ModuleNotFoundError: No module named '{module}'",
    "ImportError: cannot import name '{name}' from '{module}'",
    "SyntaxError: invalid escape sequence '\\{char}'",
    "SyntaxError: invalid encoding or BOM in source file",
    "SyntaxError: stray character '{char}' in expression",
    "SyntaxError: unknown token before '{token}'",
]

syntactic_errors = [
    "SyntaxError: unexpected EOF while parsing",
    "SyntaxError: invalid syntax near '{token}'",
    "SyntaxError: unmatched ')' at line {line}",
    "SyntaxError: unexpected token '{token}'",
    "SyntaxError: expected ':' after 'if' statement",
    "IndentationError: expected an indented block",
    "SyntaxError: cannot assign to literal",
    "SyntaxError: invalid target for augmented assignment",
    "SyntaxError: 'return' outside function",
    "SyntaxError: 'break' outside loop",
    "SyntaxError: invalid non-printable character U+{code}",
    "SyntaxError: unexpected indent at line {line}",
    "SyntaxError: invalid character in f-string",
    "SyntaxError: duplicate argument '{var}' in function definition",
    "SyntaxError: invalid type annotation",
    "SyntaxError: unterminated triple-quoted string literal",
    "SyntaxError: illegal expression for augmented assignment",
    "SyntaxError: missing parentheses in call to 'print'",
    "SyntaxError: invalid use of keyword '{token}'",
    "SyntaxError: cannot mix tabs and spaces",
    "SyntaxError: invalid statement at global scope",
]

semantic_errors = [
    "ValueError: invalid literal for int() with base 10: '{num}'",
    "ZeroDivisionError: division by zero",
    "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
    "IndexError: list index out of range",
    "KeyError: '{key}'",
    "AttributeError: '{obj}' object has no attribute '{attr}'",
    "FileNotFoundError: [Errno 2] No such file or directory: '{filename}'",
    "MemoryError: cannot allocate memory",
    "RecursionError: maximum recursion depth exceeded",
    "AssertionError: expected {a}, got {b}",
    "TypeError: object of type 'NoneType' has no len()",
    "RuntimeError: dictionary changed size during iteration",
    "ValueError: math domain error",
    "ImportError: cannot import name '{name}' from '{module}'",
    "TypeError: missing 1 required positional argument: '{arg}'",
    "ValueError: not enough values to unpack (expected {a}, got {b})",
    "TypeError: sequence item {a}: expected str instance, found int",
    "IndexError: tuple index out of range",
    "OSError: [Errno 22] Invalid argument: '{filename}'",
    "RuntimeError: generator raised StopIteration",
    "TypeError: can't multiply sequence by non-int of type '{type}'",
    "NameError: free variable '{var}' referenced before assignment",
    "TypeError: can't compare '{type}' and '{type2}'",
    "OverflowError: numerical result out of range",
    "RuntimeError: failed to acquire lock for thread",
    "TypeError: unhashable type: '{obj}'",
    "KeyError: missing key '{key}' during lookup",
    "RecursionError: infinite recursion detected in function '{name}'",
    "ValueError: could not convert string to float: '{num}'",
    "ImportError: cannot find module '{module}' in sys.path",
]

# ============================================================
# RANDOM DATA POOLS
# ============================================================
modules = [
    "os", "sys", "numpy", "pandas", "math", "random", "datetime", "re",
    "torch", "cv2", "tensorflow", "sklearn", "matplotlib", "scipy", "json", "http"
]
names = [
    "load", "fit", "train", "test", "run", "predict", "save", "get",
    "connect", "read", "write", "fetch", "analyze", "process", "execute"
]
attrs = [
    "data", "model", "shape", "dtype", "append", "execute", "plot", "transform",
    "reset", "config", "apply", "calculate", "length"
]
tokens = ["=", "==", "+", "-", "*", "/", "%", "//",
          ":", "::", "->", ".", ";", ",", "**", "and", "or"]
variables = [
    "x", "y", "result", "value", "data", "temp", "output", "counter", "array",
    "index", "flag", "user", "error", "config"
]
files = [
    "'data.csv'", "'input.json'", "'config.yaml'", "'dataset.txt'", "'output.log'",
    "'missing.txt'", "'data_backup.pkl'", "'report.xml'"
]
types = ["int", "str", "float", "list", "dict", "tuple"]

# ============================================================
# DATASET GENERATION
# ============================================================

rows = []
for _ in range(NUM_ENTRIES):
    category = random.choice(["lexical", "syntactic", "semantic"])
    template = (
        random.choice(lexical_errors)
        if category == "lexical"
        else random.choice(syntactic_errors)
        if category == "syntactic"
        else random.choice(semantic_errors)
    )

    # Fill in placeholders dynamically
    error_message = template.format(
        name=random.choice(names),
        module=random.choice(modules),
        var=random.choice(variables),
        attr=random.choice(attrs),
        token=random.choice(tokens),
        num=str(random.randint(1000, 9999)) + random.choice(["a", "b", "c"]),
        key=random.choice(
            ["user", "id", "config", "session", "file", "token"]),
        filename=random.choice(files),
        line=random.randint(1, 200),
        a=random.randint(1, 5),
        b=random.randint(1, 5),
        char=random.choice(["x", "z", "q", "y"]),
        code=random.choice(["201", "345", "400", "567"]),
        arg=random.choice(["input", "data", "path", "value", "text"]),
        obj=random.choice(["list", "dict", "DataFrame", "NoneType", "Series"]),
        type=random.choice(types),
        type2=random.choice(types),
    )

    # Inject controlled randomness for uniqueness
    error_message += f"  [line:{random.randint(1, 500)} col:{random.randint(1, 80)} id:{random.randint(1000, 9999)}]"
    rows.append([error_message, category])

# ============================================================
# WRITE TO CSV
# ============================================================
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["error_message", "error_type"])
    writer.writerows(rows)

print(f"✅ Generated {len(rows)} Python errors -> saved to {OUTPUT_FILE}")
