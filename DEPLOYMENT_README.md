# Manga Image Translator - DeepSeek Docker Deployment

This directory contains everything you need to deploy the Manga Image Translator with DeepSeek translation service using Docker on Windows.

## 📁 Files Overview

| File | Description |
|------|-------------|
| `docker-compose-deepseek.yml` | Docker Compose configuration for DeepSeek setup |
| `DEPLOYMENT_GUIDE.md` | Complete step-by-step deployment guide |
| `TROUBLESHOOTING.md` | Solutions for common issues |
| `deploy-windows.bat` | One-click deployment script for Windows |
| `check-setup.bat` | Setup verification script |
| `test-api.py` | API testing script |
| `example-deepseek-config.json` | Example configuration for API usage |

## 🚀 Quick Start

1. **Get DeepSeek API Key**: Sign up at https://platform.deepseek.com/
2. **Edit Configuration**: Update `docker-compose-deepseek.yml` with your API key
3. **Deploy**: Double-click `deploy-windows.bat` or run:
   ```bash
   docker-compose -f docker-compose-deepseek.yml up -d
   ```
4. **Access**: Open http://localhost:8000 in your browser

## 📖 Documentation

- **New to this project?** Start with `DEPLOYMENT_GUIDE.md`
- **Having issues?** Check `TROUBLESHOOTING.md`
- **Want to test?** Run `python test-api.py`

## ⚙️ Configuration

The setup is optimized for:
- **CPU processing** (no GPU required)
- **DeepSeek translator** (cost-effective, high-quality)
- **Windows environment** (with Docker Desktop)
- **Web interface** (user-friendly drag-and-drop)

## 🔧 Key Features

- **One-click deployment** with Windows batch scripts
- **Automatic setup verification** and health checks
- **Comprehensive troubleshooting** guide
- **Cost-effective translation** (~$0.001-0.015 per manga page)
- **Professional quality** text detection, OCR, and rendering

## 💡 Tips

- First-time setup downloads ~15GB Docker image
- Ensure you have sufficient disk space and RAM
- DeepSeek offers excellent translation quality at low cost
- Results are saved in the `./result` directory

## 🆘 Need Help?

1. Run `check-setup.bat` to verify your configuration
2. Check `TROUBLESHOOTING.md` for common solutions
3. Join the Discord community: https://discord.gg/Ak8APNy4vb
4. Report issues: https://github.com/zyddnys/manga-image-translator/issues

---

**Note**: This deployment setup is based on the official manga-image-translator project by zyddnys. All credit goes to the original developers and contributors.