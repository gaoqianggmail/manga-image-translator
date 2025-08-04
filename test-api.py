#!/usr/bin/env python3
"""
Simple test script to verify the Manga Image Translator API is working
with DeepSeek translator.
"""

import requests
import json
import base64
import time
from PIL import Image
import io

def create_test_image():
    """Create a simple test image with Japanese text"""
    # Create a simple white image with black text
    img = Image.new('RGB', (400, 200), color='white')
    # Note: This is just a placeholder. In real usage, you'd load an actual manga image
    return img

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_api():
    """Test the manga translator API"""
    print("🧪 Testing Manga Image Translator API...")
    
    # API endpoint
    url = "http://localhost:8000/translate/json"
    
    # Create test image
    test_img = create_test_image()
    img_base64 = image_to_base64(test_img)
    
    # Configuration for DeepSeek translator
    config = {
        "translator": {
            "translator": "deepseek",
            "target_lang": "ENG"
        },
        "detector": {
            "detector": "default",
            "detection_size": 1024
        },
        "ocr": {
            "ocr": "48px"
        },
        "inpainter": {
            "inpainter": "lama_large"
        }
    }
    
    # Request payload
    payload = {
        "image": img_base64,
        "config": config
    }
    
    try:
        print("📡 Sending request to API...")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API is working!")
            print(f"📊 Translation completed successfully")
            print(f"🔍 Detected {len(result.get('textlines', []))} text regions")
            return True
        else:
            print(f"❌ API returned error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the Docker container running?")
        print("💡 Try: docker-compose -f docker-compose-deepseek.yml up -d")
        return False
    except requests.exceptions.Timeout:
        print("⏰ Request timed out. The service might be starting up.")
        print("💡 Wait a few minutes and try again.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_health():
    """Test if the service is running"""
    print("🏥 Testing service health...")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200:
            print("✅ Web interface is accessible")
            return True
        else:
            print(f"⚠️  Web interface returned: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web interface")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🎌 Manga Image Translator - API Test")
    print("=" * 50)
    print()
    
    # Test health first
    if test_health():
        print()
        # If health check passes, test the API
        test_api()
    
    print()
    print("=" * 50)
    print("💡 Tips:")
    print("- If tests fail, check: docker-compose -f docker-compose-deepseek.yml logs")
    print("- Make sure your DeepSeek API key is configured")
    print("- Web interface: http://localhost:8000")
    print("=" * 50)