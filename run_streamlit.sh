#!/bin/bash
# Run the Streamlit Travel Assistant app

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Streamlit app
streamlit run streamlit_app.py
