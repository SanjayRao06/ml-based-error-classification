# app.py
import streamlit as st
import os
import io
import pandas as pd # Needed for summary table

# Import your main analyzer class from its .py file
# This class imports the ML model from mlbec4.py
from main_code_for_errorclassification import CppCodeErrorAnalyzer

# --- Helper Functions ---

# Use a specific cache based on the model file's modification time
@st.cache_resource
def load_analyzer_cached(model_path, data_path):
    """
    Load the analyzer once and cache it. If the model file doesn't exist, 
    it attempts to train and save it first.
    """
    analyzer = CppCodeErrorAnalyzer(model_path=model_path, data_path=data_path)
    return analyzer

def display_streamlit_results(results: dict):
    """
    A custom function to display the analysis results using Streamlit components.
    """
    if not results['success']:
        st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        return

    # Case 1: Compilation was successful
    if results['compilation_success']:
        st.success("🎉 **Compilation Successful!** No errors found.")
        st.balloons()
        return

    # Case 2: Compilation failed
    st.warning(f"**Compilation Failed.** Found {results['total_errors']} error(s)/warning(s).")
    
    # --- Summary Section ---
    st.subheader("Classification Summary")
    summary_df = pd.DataFrame(results['summary'].items(), columns=['Error Type', 'Count'])
    summary_df.columns = ['Error Type', 'Count']
    
    st.dataframe(
        summary_df.style.background_gradient(cmap='Reds', subset=['Count']),
        hide_index=True,
        use_container_width=True
    )
    
    # --- Detailed Analysis Section ---
    st.subheader("Detailed Error Analysis")
    
    for i, error in enumerate(results['errors']):
        # Use a collapsible expander for each error
        with st.expander(f"Error {i+1}: Predicted as **{error['predicted_type'].upper()}** (Conf: {error['confidence']})", expanded=i == 0):
            st.code(error['message'], language="text")
            
            # Show all confidence scores as a small table/chart
            if error['all_confidences']:
                st.markdown("**Confidence Scores:**")
                # Convert confidence dictionary to a DataFrame for display
                conf_df = pd.DataFrame(
                    error['all_confidences'].items(), 
                    columns=['Type', 'Confidence']
                ).sort_values(by='Confidence', ascending=False).reset_index(drop=True)
                
                # Format the confidence column to percentage
                conf_df['Confidence'] = conf_df['Confidence'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(conf_df, hide_index=True, use_container_width=True)

    # --- Raw Output ---
    with st.expander("Show Raw Compiler Output"):
        st.code(results['raw_compiler_output'], language="bash")


# --- Main Application Logic ---

st.set_page_config(
    page_title="C++ Error Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 C++ Error Type Classifier")
st.caption("Upload your code to see if the errors are **Lexical**, **Syntactic**, or **Semantic** using an ML model trained on `mlbec4.py`.")

model_filepath = 'cpp_error_classifier.pkl'
data_filepath = 'cpp_error_dataset.csv'

# Load analyzer (handles training if model if model/data is present)
analyzer = load_analyzer_cached(model_filepath, data_filepath)

# --- Status Check ---
st.sidebar.header("Status")

if analyzer is None or analyzer.classifier.best_model is None:
    # Check if the training failed
    st.sidebar.error("❌ ML Model Not Ready.")
    st.error("❌ **Initialization Failed!** The model could not be loaded or trained. Ensure 'cpp_error_dataset.csv' is present and contains enough data.")
    st.stop()


# Check compiler availability
compiler_available = analyzer.check_compiler_availability()
st.sidebar.markdown(f"**ML Model:** {analyzer.classifier.best_model_name}")

if not compiler_available:
    st.error(
        "❌ **g++ compiler not found!**\n"
        "Please install g++ and ensure it's in your system's PATH."
    )
    st.stop()

st.sidebar.success("✅ g++ compiler found.")
st.success(f"✅ Analyzer loaded (Model: **{analyzer.classifier.best_model_name}**). Ready!")

st.divider()

# --- File Uploader ---
st.header("1. Upload your C++ Code File")
uploaded_file = st.file_uploader(
    "Drag and drop your .cpp, .c, .h, or .hpp file here",
    type=["cpp", "c", "h", "hpp", "txt"]
)

if uploaded_file is not None:
    # To read file as string:
    string_data = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    
    # Display the uploaded code in an expander
    with st.expander("Show Uploaded Code"):
        st.code(string_data, language="cpp")
    
    # --- Analyze Button ---
    if st.button("🚀 Analyze Code", type="primary", use_container_width=True):
        st.subheader("2. Analysis Results")
        with st.spinner("Compiling and classifying errors..."):
            # Run compilation and classification
            analysis_results = analyzer.compile_and_analyze(string_data)
            
            # Display results
            display_streamlit_results(analysis_results)
            
    st.divider()

# --- Example Input ---
st.header("3. Or Try with an Example")
example_code = """
#include <iostream>

void print_sum(int a, int b) {
    int c = a + b;
    std::cout << "Sum: " << c << std::endl;
} // Missing semicolon after function declaration (Syntactic)

int main() {
    int x = 10;
    int y = "20"; // Type mismatch (Semantic)
    
    x = undeclared_variable; // Undeclared variable (Lexical)

    print_sum(x, y);
    return 0;
}
"""
st.code(example_code, language="cpp")

if st.button("🔬 Analyze Example", use_container_width=True):
    st.subheader("Analysis Results for Example Code")
    with st.spinner("Compiling and classifying example errors..."):
        analysis_results = analyzer.compile_and_analyze(example_code)
        display_streamlit_results(analysis_results)
