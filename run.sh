#!/bin/bash

# Multi-Modal Travel Assistant - Run Script

echo "🌍 Multi-Modal Travel Assistant"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -e ."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from template..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your GROQ_API_KEY"
    echo "Get your API key from: https://console.groq.com/keys"
    echo ""
    read -p "Press Enter after you've added your API key..."
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Run Streamlit
echo "🚀 Starting Streamlit app..."
echo ""
streamlit run streamlit_app.py
