#!/bin/bash
# Start manga translator server with DeepL

echo "🚀 Starting manga translator server with DeepL..."
echo "📋 Current translator: $(grep TRANSLATOR .env || echo 'TRANSLATOR=deepl')"

# Activate virtual environment and start server
source venv/bin/activate && python server/main.py --use-gpu

echo "🌐 Server should be running at:"
echo "   Web UI: http://127.0.0.1:8000"
echo "   API: http://127.0.0.1:8001"