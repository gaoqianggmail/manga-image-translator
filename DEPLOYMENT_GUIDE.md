# Manga Image Translator - DeepSeek Docker Deployment Guide

This guide will help you deploy the Manga Image Translator using Docker with DeepSeek as the translation service on Windows.

## Prerequisites

1. **Docker Desktop for Windows**
   - Download and install from: https://www.docker.com/products/docker-desktop/
   - Make sure Docker Desktop is running
   - **Important**: Ensure you have at least 20GB free disk space (Docker image is ~15GB)

2. **DeepSeek API Key**
   - Sign up at: https://platform.deepseek.com/
   - Get your API key from the dashboard
   - DeepSeek offers competitive pricing and good translation quality

3. **System Requirements**
   - Windows 10/11 with Docker Desktop
   - At least 8GB RAM (16GB recommended)
   - 20GB+ free disk space
   - Stable internet connection for first-time setup

## Quick Setup Steps

### Step 1: Get Your DeepSeek API Key

1. Visit https://platform.deepseek.com/
2. Sign up for an account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the API key (starts with `sk-`)

### Step 2: Configure the Docker Compose File

1. Open the `docker-compose-deepseek.yml` file
2. Replace `YOUR_DEEPSEEK_API_KEY_HERE` with your actual DeepSeek API key:
   ```yaml
   DEEPSEEK_API_KEY: 'sk-your-actual-api-key-here'
   ```

### Step 3: Deploy the Container

Open Command Prompt or PowerShell in the project directory and run:

```bash
# Pull the latest image
docker pull zyddnys/manga-image-translator:main

# Start the service
docker-compose -f docker-compose-deepseek.yml up -d
```

### Step 4: Access the Web Interface

1. Wait for the container to start (may take 1-2 minutes on first run)
2. The Docker image is quite large (~15GB), so first-time download will take time
3. Open your web browser
4. Navigate to: http://localhost:8000
5. You should see the Manga Image Translator web interface

**Note**: The container runs the server on port 5003 internally, but it's mapped to port 8000 on your host machine for easy access.

## Usage Instructions

### Basic Translation

1. **Upload Image**: Click "Choose File" or drag and drop your manga/image
2. **Configure Settings**:
   - **Translator**: Select "deepseek"
   - **Target Language**: Choose your desired language (e.g., "ENG" for English)
   - **Source Language**: Usually leave as "auto" for automatic detection
3. **Start Translation**: Click "Translate" button
4. **Download Result**: Once complete, download the translated image

### Advanced Configuration

You can customize the translation by adjusting these settings in the web interface:

**Recommended Settings (based on official README):**
- **Translator**: "deepseek" (for this setup)
- **OCR Model**: "48px" (recommended for Japanese)
- **Inpainter**: "lama_large" (recommended)
- **Detector**: "default" (works well for most cases, "ctd" for more text lines)
- **Detection Size**: 2048 (increase for high-res images, decrease for low-res)

**Performance Tuning:**
- **For small/low-res images**: Use `upscale_ratio: 2` to improve detection
- **For better text fitting**: Use `manga2eng` renderer
- **For mask coverage**: Set `mask_dilation_offset: 10-30`
- **For filtering OCR errors**: Increase `box_threshold` (default 0.7)

## Configuration Options

### DeepSeek-Specific Settings

The DeepSeek translator supports these environment variables:

```yaml
DEEPSEEK_API_KEY: 'your-api-key'           # Required
DEEPSEEK_API_BASE: 'https://api.deepseek.com'  # API endpoint
DEEPSEEK_MODEL: 'deepseek-chat'            # Model to use
```

### Performance Tuning for CPU

The Docker Compose file includes CPU optimization settings:

```yaml
PYTORCH_ENABLE_MPS_FALLBACK: '1'  # Enable fallback for better compatibility
OMP_NUM_THREADS: '4'              # Limit OpenMP threads
MKL_NUM_THREADS: '4'              # Limit MKL threads
```

You can adjust the thread count based on your CPU:
- 4 threads: Good for most systems
- 8 threads: For high-end CPUs
- 2 threads: For lower-end systems

## Troubleshooting

### Container Won't Start

```bash
# Check container logs
docker-compose -f docker-compose-deepseek.yml logs

# Check if container is running
docker ps
```

### API Key Issues

- Verify your DeepSeek API key is correct
- Check if you have sufficient credits in your DeepSeek account
- Ensure the API key has proper permissions

### Memory Issues

If you encounter out-of-memory errors:

1. **Reduce Detection Size**: Set to 1024 or 1536
2. **Use Smaller Models**: Choose "32px" OCR instead of "48px"
3. **Limit Concurrent Processing**: Process one image at a time

### Network Issues

- Ensure Docker Desktop is running
- Check Windows Firewall settings
- Verify port 8000 is not blocked

## Stopping the Service

```bash
# Stop the container
docker-compose -f docker-compose-deepseek.yml down

# Stop and remove volumes (clears cache)
docker-compose -f docker-compose-deepseek.yml down -v
```

## File Locations

- **Translated Images**: Saved in `./result` folder
- **Logs**: Available via `docker-compose logs`
- **Configuration**: Modify `docker-compose-deepseek.yml`

## Cost Considerations

DeepSeek pricing (as of 2024):
- Input: ~$0.14 per 1M tokens
- Output: ~$0.28 per 1M tokens

Typical manga page translation costs:
- Simple page: $0.001 - $0.005
- Complex page: $0.005 - $0.015

## Alternative Translation Services

If you want to try other services, modify the environment variables:

### OpenAI GPT
```yaml
OPENAI_API_KEY: 'your-openai-key'
OPENAI_MODEL: 'gpt-4o-mini'  # Cost-effective option
```

### DeepL (High Quality)
```yaml
DEEPL_AUTH_KEY: 'your-deepl-key'
```

### Groq (Fast & Free Tier)
```yaml
GROQ_API_KEY: 'your-groq-key'
GROQ_MODEL: 'mixtral-8x7b-32768'
```

## Testing Your Setup

After deployment, you can test if everything is working:

```bash
# Run the test script
python test-api.py

# Or manually check the web interface
# Navigate to http://localhost:8000 and try uploading an image
```

## Troubleshooting

If you encounter issues, check the `TROUBLESHOOTING.md` file for detailed solutions to common problems.

Quick diagnostics:
```bash
# Check container status
docker ps

# View logs
docker-compose -f docker-compose-deepseek.yml logs

# Restart if needed
docker-compose -f docker-compose-deepseek.yml restart
```

## Support

- **Troubleshooting Guide**: See `TROUBLESHOOTING.md`
- **Project Repository**: https://github.com/zyddnys/manga-image-translator
- **Discord Community**: https://discord.gg/Ak8APNy4vb
- **Issues**: Report bugs on GitHub Issues

## Security Notes

- Keep your API keys secure
- Don't commit API keys to version control
- Consider using environment files (.env) for production deployments
- Regularly rotate your API keys

Happy translating! 🎌📚