# Troubleshooting Guide - Manga Image Translator with DeepSeek

This guide helps you resolve common issues when deploying the Manga Image Translator with DeepSeek using Docker on Windows.

## Quick Diagnostics

Run these commands to check your setup:

```bash
# Check if Docker is running
docker version

# Check if container is running
docker ps

# Check container logs
docker-compose -f docker-compose-deepseek.yml logs

# Test API connectivity
python test-api.py
```

## Common Issues & Solutions

### 1. Container Won't Start

**Symptoms:**
- `docker-compose up` fails
- Container exits immediately
- "Port already in use" error

**Solutions:**

**A. Port Conflict**
```bash
# Check what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use a different port in docker-compose-deepseek.yml
ports:
  - "8080:5003"  # Use port 8080 instead
```

**B. Docker Issues**
```bash
# Restart Docker Desktop
# Or restart the service
docker-compose -f docker-compose-deepseek.yml down
docker-compose -f docker-compose-deepseek.yml up -d
```

### 2. API Key Problems

**Symptoms:**
- "Invalid API key" errors
- Translation requests fail
- 401/403 HTTP errors

**Solutions:**

**A. Verify API Key**
1. Check your DeepSeek dashboard: https://platform.deepseek.com/
2. Ensure the key starts with `sk-`
3. Verify you have sufficient credits

**B. Update Configuration**
```yaml
# In docker-compose-deepseek.yml
environment:
  DEEPSEEK_API_KEY: 'sk-your-actual-key-here'  # Remove quotes if causing issues
```

**C. Restart After Changes**
```bash
docker-compose -f docker-compose-deepseek.yml down
docker-compose -f docker-compose-deepseek.yml up -d
```

### 3. Memory/Performance Issues

**Symptoms:**
- Container crashes with OOM (Out of Memory)
- Very slow processing
- System becomes unresponsive

**Solutions:**

**A. Increase Docker Memory**
1. Open Docker Desktop
2. Go to Settings → Resources
3. Increase Memory to at least 8GB (16GB recommended)
4. Apply & Restart

**B. Optimize Settings**
```json
{
  "detector": {
    "detection_size": 1024  // Reduce from 2048
  },
  "ocr": {
    "ocr": "32px"  // Use smaller model
  },
  "inpainter": {
    "inpainting_size": 1024  // Reduce from 2048
  }
}
```

**C. Process Images One at a Time**
- Don't upload multiple large images simultaneously
- Wait for one translation to complete before starting another

### 4. Network/Connection Issues

**Symptoms:**
- "Cannot connect to API" errors
- Timeouts
- Web interface not loading

**Solutions:**

**A. Check Container Status**
```bash
docker ps
# Should show manga_translator_deepseek_cpu as "Up"
```

**B. Check Logs**
```bash
docker-compose -f docker-compose-deepseek.yml logs
# Look for error messages
```

**C. Firewall Issues**
1. Windows Defender Firewall might block Docker
2. Add Docker Desktop to firewall exceptions
3. Or temporarily disable firewall for testing

**D. Proxy Issues**
If you're behind a corporate proxy:
```yaml
# Add to docker-compose-deepseek.yml
environment:
  HTTP_PROXY: "http://your-proxy:port"
  HTTPS_PROXY: "http://your-proxy:port"
```

### 5. Translation Quality Issues

**Symptoms:**
- Poor translation quality
- Text not detected
- Original text not removed properly

**Solutions:**

**A. Improve Text Detection**
```json
{
  "detector": {
    "detector": "ctd",  // Try different detector
    "detection_size": 2048,  // Increase for high-res images
    "box_threshold": 0.5  // Lower to detect more text
  }
}
```

**B. Better OCR**
```json
{
  "ocr": {
    "ocr": "48px",  // Best for Japanese
    "min_text_length": 1  // Don't skip short text
  }
}
```

**C. Image Preprocessing**
```json
{
  "upscale": {
    "upscale_ratio": 2  // Upscale small images
  }
}
```

### 6. Docker Image Issues

**Symptoms:**
- "Image not found" errors
- Download failures
- Corrupted image

**Solutions:**

**A. Clean Docker Cache**
```bash
docker system prune -a
docker pull zyddnys/manga-image-translator:main
```

**B. Check Disk Space**
- Ensure at least 20GB free space
- The Docker image is ~15GB

**C. Network Issues**
```bash
# Try pulling manually
docker pull zyddnys/manga-image-translator:main

# If it fails, check your internet connection
# Or try using a VPN if there are regional restrictions
```

### 7. Web Interface Issues

**Symptoms:**
- Page won't load
- Upload button not working
- Results not displaying

**Solutions:**

**A. Browser Issues**
- Try a different browser (Chrome, Firefox, Edge)
- Clear browser cache
- Disable browser extensions
- Try incognito/private mode

**B. Check Console**
1. Press F12 in browser
2. Check Console tab for JavaScript errors
3. Check Network tab for failed requests

**C. Direct API Test**
```bash
# Test the API directly
python test-api.py
```

## Advanced Troubleshooting

### Enable Debug Mode

Add verbose logging to see more details:

```yaml
# In docker-compose-deepseek.yml
command: server/main.py --verbose --start-instance --host=0.0.0.0 --port=5003
```

### Check Container Resources

```bash
# Monitor container resource usage
docker stats manga_translator_deepseek_cpu

# Check container details
docker inspect manga_translator_deepseek_cpu
```

### Manual Container Testing

```bash
# Run container interactively for debugging
docker run -it --rm zyddnys/manga-image-translator:main /bin/bash

# Test translation manually
python -m manga_translator local -i /path/to/image
```

## Getting Help

If you're still having issues:

1. **Check the logs**: `docker-compose -f docker-compose-deepseek.yml logs`
2. **Run diagnostics**: `python test-api.py`
3. **Check system resources**: Task Manager → Performance
4. **Visit the project**: https://github.com/zyddnys/manga-image-translator
5. **Join Discord**: https://discord.gg/Ak8APNy4vb

## Useful Commands Reference

```bash
# Start service
docker-compose -f docker-compose-deepseek.yml up -d

# Stop service
docker-compose -f docker-compose-deepseek.yml down

# View logs
docker-compose -f docker-compose-deepseek.yml logs -f

# Restart service
docker-compose -f docker-compose-deepseek.yml restart

# Check status
docker-compose -f docker-compose-deepseek.yml ps

# Update image
docker pull zyddnys/manga-image-translator:main
docker-compose -f docker-compose-deepseek.yml up -d

# Clean up
docker-compose -f docker-compose-deepseek.yml down -v
docker system prune -a
```

## Performance Optimization

For better performance on Windows:

1. **Use WSL2 backend** in Docker Desktop settings
2. **Allocate more resources** to Docker Desktop
3. **Close unnecessary applications** while translating
4. **Use SSD storage** for Docker data
5. **Process smaller images** or reduce detection size

Remember: The first run will be slow due to model loading. Subsequent translations should be faster!