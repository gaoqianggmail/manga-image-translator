#!/bin/bash
# Stop manga translator server by killing processes on ports 8000-8003

echo "🛑 Stopping manga translator server..."

# Function to kill process on a specific port
kill_port() {
    local port=$1
    local pid=$(lsof -ti :$port)
    if [ ! -z "$pid" ]; then
        echo "   Killing process $pid on port $port"
        kill $pid
        sleep 1
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            echo "   Force killing process $pid"
            kill -9 $pid
        fi
    else
        echo "   No process found on port $port"
    fi
}

# Kill processes on ports 8000, 8001, 8002, 8003
for port in 8000 8001 8002 8003; do
    kill_port $port
done

echo "✅ Server shutdown complete"