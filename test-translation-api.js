// Test script to verify translation API returns translated images
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const API_BASE_URL = 'http://localhost:8001'; // Adjust port as needed

async function testTranslationAPI() {
  console.log('🧪 Testing Translation API...\n');

  // Check if we have a test image
  const testImagePath = path.join(__dirname, 'test-image.jpg');
  if (!fs.existsSync(testImagePath)) {
    console.log('❌ No test image found. Please add a test-image.jpg file to test.');
    console.log('   You can use any manga/comic image for testing.');
    return;
  }

  try {
    // Test 1: JSON endpoint
    console.log('1. Testing JSON endpoint...');
    const formData = new FormData();
    formData.append('image', fs.createReadStream(testImagePath));
    formData.append('config', JSON.stringify({
      translator: { translator: 'deepl' },
      target_lang: 'ENG'
    }));

    const jsonResponse = await axios.post(`${API_BASE_URL}/translate/with-form/json`, formData, {
      headers: formData.getHeaders(),
      timeout: 60000 // 1 minute timeout
    });

    console.log('✅ JSON Response received');
    console.log(`   - Translations found: ${jsonResponse.data.translations?.length || 0}`);
    console.log(`   - Debug folder: ${jsonResponse.data.debug_folder || 'Not provided'}`);

    // Test 2: Check if final.png exists
    if (jsonResponse.data.debug_folder) {
      console.log('\n2. Testing final.png endpoint...');
      try {
        const imageResponse = await axios.get(`${API_BASE_URL}/result/${jsonResponse.data.debug_folder}/final.png`, {
          responseType: 'stream',
          timeout: 10000
        });
        
        console.log('✅ Translated image accessible');
        console.log(`   - Content-Type: ${imageResponse.headers['content-type']}`);
        console.log(`   - URL: ${API_BASE_URL}/result/${jsonResponse.data.debug_folder}/final.png`);
      } catch (imageError) {
        console.log('❌ Translated image not accessible:', imageError.message);
      }
    } else {
      console.log('\n2. ❌ No debug_folder provided, cannot test final.png');
    }

    // Test 3: Direct image endpoint
    console.log('\n3. Testing direct image endpoint...');
    try {
      const directImageResponse = await axios.post(`${API_BASE_URL}/translate/with-form/image`, formData, {
        headers: formData.getHeaders(),
        responseType: 'stream',
        timeout: 60000
      });
      
      console.log('✅ Direct image endpoint works');
      console.log(`   - Content-Type: ${directImageResponse.headers['content-type']}`);
    } catch (directError) {
      console.log('❌ Direct image endpoint failed:', directError.message);
    }

  } catch (error) {
    console.log('❌ Translation test failed:', error.message);
    if (error.response) {
      console.log(`   - Status: ${error.response.status}`);
      console.log(`   - Response: ${error.response.data}`);
    }
  }
}

// Test server health first
async function testServerHealth() {
  try {
    const response = await axios.get(`${API_BASE_URL}/`, { timeout: 5000 });
    console.log('✅ Server is running');
    return true;
  } catch (error) {
    console.log('❌ Server is not accessible:', error.message);
    return false;
  }
}

async function main() {
  const serverOnline = await testServerHealth();
  if (serverOnline) {
    await testTranslationAPI();
  }
}

main().catch(console.error);