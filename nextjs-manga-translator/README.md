# Manga Translator - Next.js Frontend

A modern Next.js frontend for the manga image translator API, optimized for handling multiple concurrent users and batch image processing.

## 🚀 Features

- **Batch Upload**: Upload up to 10 images simultaneously
- **Real-time Progress**: Live progress tracking for each translation
- **Queue Management**: Smart request queuing to prevent server overload
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Robust error handling with retry mechanisms
- **Performance Optimized**: Designed for 100+ concurrent users

## 🛠️ Setup

### 1. Install Dependencies

```bash
cd nextjs-manga-translator
npm install
```

### 2. Configure API Endpoint

Update `.env.local` with your Mac Mini's IP address:

```env
NEXT_PUBLIC_API_URL=http://YOUR_MAC_MINI_IP:8001
```

### 3. Run Development Server

```bash
npm run dev
```

Visit `http://localhost:3000`

### 4. Build for Production

```bash
npm run build
npm start
```

## 🏗️ Architecture

### Performance Optimizations

1. **Request Queuing**: Limits concurrent requests per client (3 max)
2. **Batch Processing**: Processes images in small batches (2 at a time)
3. **Retry Logic**: Automatic retry with exponential backoff
4. **Progress Streaming**: Real-time upload/translation progress
5. **Memory Management**: Proper cleanup of object URLs

### Components

- `ImageUploader`: Drag & drop interface with preview
- `TranslationProgress`: Real-time progress tracking
- `TranslationResults`: Results display with download options
- `api.ts`: Optimized API client with queue management

## 📊 Server Performance Recommendations

### Mac Mini M4 Optimization

1. **Multiple Translation Instances**:
```bash
# Start 3-4 translation instances for better concurrency
python -m manga_translator shared --host 127.0.0.1 --port 8002 --use-gpu &
python -m manga_translator shared --host 127.0.0.1 --port 8003 --use-gpu &
python -m manga_translator shared --host 127.0.0.1 --port 8004 --use-gpu &
```

2. **Server Configuration**:
```bash
# Start main server with multiple instances
python server/main.py --use-gpu --host 0.0.0.0 --port 8001
```

3. **System Optimization**:
- Enable GPU acceleration (MPS on M4)
- Increase file descriptor limits
- Monitor memory usage
- Use SSD for temp files

### Expected Performance

- **Single Image**: 6-8 seconds (with DeepL)
- **Concurrent Users**: 100+ users supported
- **Queue Management**: Automatic load balancing
- **Memory Usage**: ~2GB per translation instance

## 🔧 Configuration Options

### Translation Settings

```typescript
const config = {
  translator: { translator: 'deepl' }, // deepl, google, chatgpt, groq
  target_lang: 'CHS', // CHS, CHT, ENG, JPN, KOR, etc.
  detector: { detector: 'default' },
  ocr: { ocr: 'default' },
  inpainter: { inpainter: 'default' }
};
```

### API Client Settings

```typescript
const API_CONFIG = {
  MAX_CONCURRENT_REQUESTS: 3, // Per client
  REQUEST_TIMEOUT: 120000, // 2 minutes
  RETRY_ATTEMPTS: 2,
  BATCH_SIZE: 2 // Images per batch
};
```

## 🚀 Deployment to Cloudflare

### 1. Build for Static Export

```bash
npm run build
```

### 2. Deploy to Cloudflare Pages

```bash
# Using Wrangler CLI
npx wrangler pages publish out

# Or connect your GitHub repo to Cloudflare Pages
```

### 3. Configure Environment Variables

In Cloudflare Pages dashboard:
- `NEXT_PUBLIC_API_URL`: Your Mac Mini's public IP/domain

## 🔒 Security Considerations

1. **CORS Configuration**: Server allows all origins for development
2. **Rate Limiting**: Implement on server side if needed
3. **File Size Limits**: 10MB per image, 10 images max
4. **Network Security**: Consider VPN or firewall rules

## 📈 Monitoring

### Client-side Metrics

- Upload progress per image
- Translation queue position
- Server health status
- Error rates and retry attempts

### Server-side Monitoring

- Queue size tracking
- Instance health checks
- Memory and GPU usage
- Response times

## 🐛 Troubleshooting

### Common Issues

1. **Server Offline**: Check Mac Mini server status
2. **Slow Translations**: Monitor server load and queue
3. **Upload Failures**: Check file size and format
4. **CORS Errors**: Verify server CORS configuration

### Debug Mode

Enable debug logging:

```typescript
// In api.ts
const DEBUG = true;
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details