'use client';

import React, { useState, useEffect } from 'react';
import { Settings, Zap, Globe, Server } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import TranslationProgress, { TranslationStatus } from '../components/TranslationProgress';
import TranslationResults from '../components/TranslationResults';
import { translateBatch, TranslationConfig, TranslationResult, checkServerHealth, getQueueSize } from '../lib/api';

export default function Home() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [translations, setTranslations] = useState<TranslationStatus[]>([]);
  const [results, setResults] = useState<Array<{
    filename: string;
    result: TranslationResult;
    originalFile: File;
  }>>([]);
  const [isTranslating, setIsTranslating] = useState(false);
  const [config, setConfig] = useState<TranslationConfig>({
    translator: { translator: 'deepl' },
    target_lang: 'CHS'
  });
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [queueSize, setQueueSize] = useState<number>(0);

  // Check server health on mount
  useEffect(() => {
    const checkHealth = async () => {
      const isOnline = await checkServerHealth();
      setServerOnline(isOnline);
    };
    
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  // Update queue size periodically
  useEffect(() => {
    const updateQueueSize = async () => {
      if (serverOnline) {
        try {
          const size = await getQueueSize();
          setQueueSize(size);
        } catch (error) {
          console.error('Failed to get queue size:', error);
        }
      }
    };

    if (serverOnline) {
      updateQueueSize();
      const interval = setInterval(updateQueueSize, 5000); // Update every 5 seconds
      return () => clearInterval(interval);
    }
  }, [serverOnline]);

  const handleTranslate = async () => {
    if (selectedFiles.length === 0) return;

    setIsTranslating(true);
    setResults([]);

    // Initialize translation status for all files
    const initialTranslations: TranslationStatus[] = selectedFiles.map((file, index) => ({
      id: `${index}-${file.name}`,
      filename: file.name,
      status: 'pending',
      progress: 0
    }));

    setTranslations(initialTranslations);

    try {
      const translationResults = await translateBatch(
        selectedFiles,
        config,
        (imageIndex, progress) => {
          setTranslations(prev => prev.map((t, index) => 
            index === imageIndex 
              ? { 
                  ...t, 
                  status: progress.percentage < 100 ? 'uploading' : 'translating',
                  progress: progress.percentage 
                }
              : t
          ));
        }
      );

      // Update final results
      const finalResults = selectedFiles.map((file, index) => ({
        filename: file.name,
        result: translationResults[index],
        originalFile: file
      }));

      setResults(finalResults);

      // Update translation status to completed
      setTranslations(prev => prev.map((t, index) => ({
        ...t,
        status: 'completed',
        progress: 100,
        result: translationResults[index]
      })));

    } catch (error: any) {
      console.error('Translation failed:', error);
      
      // Mark all as failed
      setTranslations(prev => prev.map(t => ({
        ...t,
        status: 'error',
        error: error.message || 'Translation failed'
      })));
    } finally {
      setIsTranslating(false);
    }
  };

  const resetAll = () => {
    setSelectedFiles([]);
    setTranslations([]);
    setResults([]);
    setIsTranslating(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Globe className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Manga Translator</h1>
                <p className="text-sm text-gray-500">AI-powered image translation</p>
              </div>
            </div>
            
            {/* Server Status */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Server className="h-4 w-4" />
                <span className="text-sm">
                  Server: 
                  <span className={`ml-1 font-medium ${
                    serverOnline === null ? 'text-gray-500' : 
                    serverOnline ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {serverOnline === null ? 'Checking...' : 
                     serverOnline ? 'Online' : 'Offline'}
                  </span>
                </span>
              </div>
              
              {queueSize > 0 && (
                <div className="text-sm text-gray-600">
                  Queue: {queueSize}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Upload & Settings */}
          <div className="lg:col-span-2 space-y-6">
            {/* Configuration */}
            <div className="bg-white rounded-lg border p-6">
              <div className="flex items-center gap-2 mb-4">
                <Settings className="h-5 w-5 text-gray-600" />
                <h2 className="text-lg font-medium text-gray-900">Translation Settings</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Translator
                  </label>
                  <select
                    value={config.translator?.translator || 'deepl'}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      translator: { translator: e.target.value }
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="deepl">DeepL (Recommended)</option>
                    <option value="google">Google Translate</option>
                    <option value="chatgpt">ChatGPT</option>
                    <option value="groq">Groq</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Target Language
                  </label>
                  <select
                    value={config.target_lang || 'CHS'}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      target_lang: e.target.value
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="CHS">Chinese (Simplified)</option>
                    <option value="CHT">Chinese (Traditional)</option>
                    <option value="ENG">English</option>
                    <option value="JPN">Japanese</option>
                    <option value="KOR">Korean</option>
                    <option value="ESP">Spanish</option>
                    <option value="FRA">French</option>
                    <option value="DEU">German</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Image Upload */}
            <div className="bg-white rounded-lg border p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Upload Images</h2>
              <ImageUploader
                onImagesChange={setSelectedFiles}
                maxFiles={10}
                maxSize={10}
              />
            </div>

            {/* Action Buttons */}
            {selectedFiles.length > 0 && (
              <div className="flex gap-4">
                <button
                  onClick={handleTranslate}
                  disabled={isTranslating || !serverOnline}
                  className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  <Zap className="h-5 w-5" />
                  {isTranslating ? 'Translating...' : `Translate ${selectedFiles.length} Image${selectedFiles.length > 1 ? 's' : ''}`}
                </button>
                
                <button
                  onClick={resetAll}
                  disabled={isTranslating}
                  className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  Reset
                </button>
              </div>
            )}
          </div>

          {/* Right Column - Progress */}
          <div className="space-y-6">
            <TranslationProgress 
              translations={translations}
              queueSize={queueSize}
            />
          </div>
        </div>

        {/* Results Section */}
        {results.length > 0 && (
          <div className="mt-12">
            <TranslationResults results={results} />
          </div>
        )}
      </main>
    </div>
  );
}