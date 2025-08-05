#!/bin/bash
# Check if manga translator server is running

echo "🔍 Checking manga translator server status..."

# Check if server is running on port 8000
if lsof -i :8000 > /dev/null 2>&1; then
    echo "✅ Server is running on port 8000"
    PID=$(lsof -ti :8000)
    echo "   Process ID: $PID"
    
    # Test the main endpoint
    echo "🌐 Testing server endpoints..."
    
    if curl -s http://localhost:8000/ > /dev/null; then
        echo "✅ Main page accessible"
    else
        echo "❌ Main page not accessible"
    fi
    
    if curl -s -X POST http://localhost:8000/queue-size > /dev/null; then
        echo "✅ Queue-size endpoint accessible"
    else
        echo "❌ Queue-size endpoint not accessible"
    fi
    
else
    echo "❌ No server running on port 8000"
    echo "💡 Start the server with: ./start_server.sh"
fi

# Check port 8001 as well
if lsof -i :8001 > /dev/null 2>&1; then
    echo "ℹ️  Something is also running on port 8001"
    PID=$(lsof -ti :8001)
    echo "   Process ID: $PID"
else
    echo "ℹ️  Nothing running on port 8001"
fi

echo ""
echo "🚀 To start the server: ./start_server.sh"
echo "🛑 To stop the server: ./stop_server.sh"