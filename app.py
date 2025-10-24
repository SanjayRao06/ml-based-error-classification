# app.py
import streamlit as st
import os
import io

# Import your main analyzer class from its .py file
from main_code_for_errorclassification import CppCodeErrorAnalyzer

# --- Helper Functions ---

@st.cache_resource
def load_analyzer_cached():
    """
    Load the analyzer once and cache it.
    This stops Streamlit from reloading the model on every interaction.
    """
    model_path = 'cpp_error_classifier.pkl'
    if not os.path.exists(model_path):
        return None  # Model file not found
    
    analyzer = CppCodeErrorAnalyzer(model_path=model_path)
    return analyzer

def display_streamlit_results(results: dict):
    """
    A custom function to display the analysis results using Streamlit components.
    This replaces the console-based 'print_analysis_results'.
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
    summary = results.get('summary', {})
    if summary:
        st.subheader("📊 Analysis Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Errors Analyzed", summary.get('total', 0))
        col2.metric("Most Common Type", summary.get('most_common_type', 'N/A').upper())
        col3.metric("Average Confidence", f"{summary.get('average_confidence', 0.0):.2%}")
        
        with st.expander("Show Error Type Breakdown"):
            st.json(summary.get('by_type', {}))

    # --- Detailed Error List ---
    st.subheader("📋 Detailed Error List")
    classifications = results.get('classifications', [])
    
    if not classifications:
        st.info("No detailed classifications were generated.")
        return

    for cls in classifications:
        error_type = cls.get('predicted_type', 'unknown').upper()
        confidence = cls.get('confidence', 0.0)
        
        with st.container(border=True):
            st.markdown(f"**Error {cls['error_number']}: {error_type}**")
            
            # Display the raw error message
            st.code(cls['error_message'], language="bash")
            
            # Display confidence and probabilities
            col1, col2 = st.columns([1, 2])
            col1.metric("Confidence", f"{confidence:.2%}")
            
            with col2.expander("Show all probabilities"):
                probs = cls.get('all_probabilities', {})
                # Sort probabilities from high to low for easy reading
                sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                # Format them nicely
                formatted_probs = {k: f"{v:.2%}" for k, v in sorted_probs}
                st.json(formatted_probs)

# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="C++ Error Classifier", layout="wide")
    st.title("🤖 C++ Code Error Classification System")
    
    # Load the analyzer
    analyzer = load_analyzer_cached()

    # --- Prerequisite Checks ---
    if analyzer is None:
        st.error(
            "❌ **Model file not found!**\n"
            "Please run `python mlbec3.py` first to train and save the `cpp_error_classifier.pkl` model."
        )
        st.stop()  # Stop the app if model isn't found
    
    if not analyzer.compiler_available:
        st.error(
            "❌ **g++ compiler not found!**\n"
            "Please install g++ and ensure it's in your system's PATH."
            "\n(e.g., MinGW on Windows, `sudo apt-get install g++` on Linux, or Xcode tools on macOS)."
        )
        st.stop()  # Stop the app if compiler isn't found
    
    st.success(f"✅ Analyzer loaded (Model: {analyzer.classifier.best_model_name}) and g++ compiler found. Ready!")
    
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
            with st.spinner("Compiling and classifying errors..."):
                # Run the analysis
                results = analyzer.analyze_code(string_data)
            
            # Display the results
            st.header("2. Analysis Results")
            display_streamlit_results(results)

if __name__ == "__main__":
    main()