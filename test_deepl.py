#!/usr/bin/env python3
"""
Quick test script to verify DeepL translator is working
"""
import os
import sys
sys.path.append('.')

from manga_translator.translators.deepl import DeeplTranslator

def test_deepl():
    try:
        print("Testing DeepL translator...")
        translator = DeeplTranslator()
        
        # Test translation
        test_text = ["Hello, this is a test translation."]
        result = translator.translate(test_text, 'ENG', 'JPN')
        
        print(f"Original: {test_text[0]}")
        print(f"Translated: {result[0]}")
        print("✅ DeepL translator is working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing DeepL: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_deepl()