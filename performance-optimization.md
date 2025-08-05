# Mac Mini M4 Performance Optimization for 100 Concurrent Users

## 🎯 Performance Goals
- Support 100 users per day
- Handle 10-20 concurrent translation requests
- Maintain 6-8 second translation times
- Minimize server crashes and timeouts

## 🏗️ Multi-Instance Architecture

### 1. Start Multiple Translation Instances

Create a script to start multiple translation workers:

```bash
#!/bin/bash
# start-multiple-instances.sh

echo "🚀 Starting multiple manga translator instances..."

# Kill existing instances
./stop_server.sh

# Start main server (port 8001)
echo "Starting main server on port 8001..."
source venv/bin/activate && python server/main.py --use-gpu --host 0.0.0.0 --port 8001 &
MAIN_PID=$!

# Wait for main server to start
sleep 5

# Start additional translation instances
for port in 8002 8003 8004; do
    echo "Starting translation instance on port $port..."
    source venv/bin/activate && python -m manga_translator shared --host 127.0.0.1 --port $port --use-gpu --nonce $(cat .nonce 2>/dev/null || echo "default") &
    sleep 2
done

echo "✅ All instances started!"
echo "Main server: http://0.0.0.0:8001"
echo "Translation instances: 8002, 8003, 8004"

# Save PIDs for cleanup
echo $MAIN_PID > .main_server.pid
```

### 2. Update Stop Script

```bash
#!/bin/bash
# stop-all-instances.sh

echo "🛑 Stopping all manga translator instances..."

# Kill processes on all ports
for port in 8001 8002 8003 8004; do
    echo "Stopping processes on port $port..."
    lsof -ti :$port | xargs kill -9 2>/dev/null || true
done

# Kill by PID if available
if [ -f .main_server.pid ]; then
    kill -9 $(cat .main_server.pid) 2>/dev/null || true
    rm .main_server.pid
fi

# Kill any remaining python processes
pkill -f "manga_translator" 2>/dev/null || true

echo "✅ All instances stopped!"
```

## ⚙️ System Optimization

### 1. macOS System Settings

```bash
# Increase file descriptor limits
echo "kern.maxfiles=65536" | sudo tee -a /etc/sysctl.conf
echo "kern.maxfilesperproc=32768" | sudo tee -a /etc/sysctl.conf

# Increase shared memory
echo "kern.sysv.shmmax=1073741824" | sudo tee -a /etc/sysctl.conf
echo "kern.sysv.shmall=262144" | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

### 2. Python/Virtual Environment Optimization

```bash
# Install performance packages
pip install uvloop  # Faster event loop
pip install orjson  # Faster JSON parsing
pip install pillow-simd  # Faster image processing (if available)

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4  # Optimize for M4 cores
```

### 3. Memory Management

```python
# Add to server/main.py
import gc
import psutil
import threading
import time

def memory_monitor():
    """Monitor and manage memory usage"""
    while True:
        memory = psutil.virtual_memory()
        if memory.percent > 85:  # If memory usage > 85%
            print(f"⚠️  High memory usage: {memory.percent}%")
            gc.collect()  # Force garbage collection
        time.sleep(30)

# Start memory monitor thread
threading.Thread(target=memory_monitor, daemon=True).start()
```

## 🔧 Server Configuration Optimization

### 1. Enhanced Server Configuration

```python
# server/config.py
import os

class PerformanceConfig:
    # Instance management
    MAX_TRANSLATION_INSTANCES = 4
    INSTANCE_PORTS = [8002, 8003, 8004, 8005]
    
    # Queue management
    MAX_QUEUE_SIZE = 50
    QUEUE_TIMEOUT = 300  # 5 minutes
    
    # Request handling
    MAX_CONCURRENT_REQUESTS = 20
    REQUEST_TIMEOUT = 120
    
    # Memory management
    MODELS_TTL = 1800  # 30 minutes
    MAX_MEMORY_USAGE = 85  # Percentage
    
    # GPU optimization
    GPU_MEMORY_FRACTION = 0.8
    ENABLE_MPS = True  # For M4 Metal Performance Shaders
    
    # File handling
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    TEMP_DIR = "/tmp/manga_translator"
    CLEANUP_INTERVAL = 3600  # 1 hour
```

### 2. Load Balancer Implementation

```python
# server/load_balancer.py
import asyncio
import aiohttp
from typing import List, Optional
import random

class LoadBalancer:
    def __init__(self, instances: List[str]):
        self.instances = instances
        self.health_status = {instance: True for instance in instances}
        self.request_counts = {instance: 0 for instance in instances}
    
    async def get_best_instance(self) -> Optional[str]:
        """Get the instance with lowest load"""
        healthy_instances = [
            instance for instance in self.instances 
            if self.health_status[instance]
        ]
        
        if not healthy_instances:
            return None
        
        # Return instance with lowest request count
        return min(healthy_instances, key=lambda x: self.request_counts[x])
    
    async def health_check(self):
        """Periodically check instance health"""
        while True:
            for instance in self.instances:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"http://{instance}/health", timeout=5) as response:
                            self.health_status[instance] = response.status == 200
                except:
                    self.health_status[instance] = False
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    def increment_request(self, instance: str):
        self.request_counts[instance] += 1
    
    def decrement_request(self, instance: str):
        self.request_counts[instance] = max(0, self.request_counts[instance] - 1)
```

## 📊 Monitoring and Alerting

### 1. Performance Monitor

```python
# server/monitor.py
import psutil
import time
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'requests_per_minute': 0,
            'average_response_time': 0,
            'memory_usage': 0,
            'cpu_usage': 0,
            'gpu_usage': 0,
            'queue_size': 0,
            'active_translations': 0
        }
    
    def log_metrics(self):
        """Log performance metrics"""
        self.metrics.update({
            'timestamp': datetime.now().isoformat(),
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(),
            'disk_usage': psutil.disk_usage('/').percent
        })
        
        # Log to file
        with open('performance.log', 'a') as f:
            f.write(json.dumps(self.metrics) + '\n')
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        while True:
            self.log_metrics()
            await asyncio.sleep(60)  # Log every minute
```

### 2. Alert System

```python
# server/alerts.py
import smtplib
from email.mime.text import MIMEText

class AlertSystem:
    def __init__(self, email_config):
        self.email_config = email_config
    
    def send_alert(self, message: str, severity: str = "WARNING"):
        """Send alert email"""
        if severity == "CRITICAL":
            subject = f"🚨 CRITICAL: Manga Translator Server Alert"
        else:
            subject = f"⚠️  WARNING: Manga Translator Server Alert"
        
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.email_config['from']
        msg['To'] = self.email_config['to']
        
        try:
            with smtplib.SMTP(self.email_config['smtp_server']) as server:
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send alert: {e}")
    
    def check_thresholds(self, metrics):
        """Check if any metrics exceed thresholds"""
        if metrics['memory_usage'] > 90:
            self.send_alert(f"High memory usage: {metrics['memory_usage']}%", "CRITICAL")
        
        if metrics['queue_size'] > 30:
            self.send_alert(f"High queue size: {metrics['queue_size']}", "WARNING")
        
        if metrics['average_response_time'] > 15:
            self.send_alert(f"Slow response time: {metrics['average_response_time']}s", "WARNING")
```

## 🚀 Deployment Script

```bash
#!/bin/bash
# deploy-production.sh

echo "🚀 Deploying Manga Translator for Production..."

# 1. System optimization
echo "Optimizing system settings..."
sudo sysctl -w kern.maxfiles=65536
sudo sysctl -w kern.maxfilesperproc=32768

# 2. Create necessary directories
mkdir -p logs temp result

# 3. Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MPS_ENABLED=1

# 4. Start services
echo "Starting translation services..."
./start-multiple-instances.sh

# 5. Start monitoring
echo "Starting monitoring..."
python -c "
import asyncio
from server.monitor import PerformanceMonitor
monitor = PerformanceMonitor()
asyncio.run(monitor.start_monitoring())
" &

echo "✅ Production deployment complete!"
echo "🌐 Server running at: http://0.0.0.0:8001"
echo "📊 Monitor logs: tail -f performance.log"
```

## 📈 Expected Performance Metrics

### With Optimizations:

- **Concurrent Users**: 100+ per day
- **Simultaneous Requests**: 15-20
- **Translation Time**: 6-8 seconds (DeepL)
- **Memory Usage**: 8-12GB (with 4 instances)
- **CPU Usage**: 60-80% under load
- **Queue Wait Time**: < 30 seconds

### Scaling Recommendations:

1. **Light Load (1-20 users/day)**: 2 instances
2. **Medium Load (20-50 users/day)**: 3 instances  
3. **Heavy Load (50-100 users/day)**: 4 instances
4. **Enterprise Load (100+ users/day)**: Consider multiple Mac Minis

## 🔍 Troubleshooting

### Common Issues:

1. **Memory Leaks**: Monitor with `top` and restart instances if needed
2. **GPU Overload**: Reduce concurrent instances or add delays
3. **Network Timeouts**: Increase timeout values in Next.js client
4. **Queue Overflow**: Implement request rate limiting

### Debug Commands:

```bash
# Monitor system resources
htop

# Check GPU usage (if available)
sudo powermetrics -n 1 -s gpu_power

# Monitor network connections
netstat -an | grep :800

# Check translation logs
tail -f logs/translation.log
```

This setup should handle 100 concurrent users efficiently on your Mac Mini M4! 🚀