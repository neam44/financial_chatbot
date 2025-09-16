import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Chatbot",
    page_icon="📊",
    layout="wide"
)

# Initialize chatbot
@st.cache_resource
def load_chatbot():
    return FinancialChatbot(os.getenv('OPENAI_API_KEY'))

def main():
    st.title("💼 Financial Analysis Chatbot")
    st.markdown("Upload your financial data and ask questions for AI-powered analysis")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Financial Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Loaded {len(df)} rows of data")
            st.dataframe(df.head())
    
    # Main chat interface
    chatbot = load_chatbot()
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your financial data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                if 'df' in locals():
                    response = chatbot.process_query(prompt, df)
                else:
                    response = chatbot.process_query(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()