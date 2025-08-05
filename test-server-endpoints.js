// Test script to check server endpoints
const axios = require('axios');

const servers = [
  'http://localhost:8000',
  'http://localhost:8001',
  'http://127.0.0.1:8000',
  'http://127.0.0.1:8001'
];

async function testEndpoint(baseUrl, endpoint, method = 'GET') {
  try {
    const response = method === 'POST' 
      ? await axios.post(`${baseUrl}${endpoint}`, {}, { timeout: 5000 })
      : await axios.get(`${baseUrl}${endpoint}`, { timeout: 5000 });
    
    console.log(`✅ ${method} ${baseUrl}${endpoint} - Status: ${response.status}`);
    return true;
  } catch (error) {
    console.log(`❌ ${method} ${baseUrl}${endpoint} - Error: ${error.message}`);
    return false;
  }
}

async function testAllServers() {
  console.log('🔍 Testing server endpoints...\n');
  
  for (const server of servers) {
    console.log(`Testing ${server}:`);
    
    // Test main page
    await testEndpoint(server, '/', 'GET');
    
    // Test queue-size endpoint
    await testEndpoint(server, '/queue-size', 'POST');
    
    // Test translate endpoint
    await testEndpoint(server, '/translate/with-form/json', 'POST');
    
    console.log('');
  }
}

testAllServers().catch(console.error);