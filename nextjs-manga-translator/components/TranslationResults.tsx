'use client';

import React, { useState } from 'react';
import { Download, Eye, EyeOff, ZoomIn, X } from 'lucide-react';
import { TranslationResult } from '../lib/api';

interface TranslationResultsProps {
  results: Array<{
    filename: string;
    result: TranslationResult;
    originalFile: File;
  }>;
}

export default function TranslationResults({ results }: TranslationResultsProps) {
  const [selectedImage, setSelectedImage] = useState<number | null>(null);
  const [showOriginal, setShowOriginal] = useState<{ [key: number]: boolean }>({});

  // No cleanup needed since we're using server URLs

  if (results.length === 0) return null;

  const downloadResult = async (filename: string, result: TranslationResult) => {
    try {
      if (result.translatedImageUrl) {
        // Download the translated image
        const exportFileDefaultName = `${filename.split('.')[0]}_translated.png`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', result.translatedImageUrl);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.setAttribute('target', '_blank');
        linkElement.click();
      } else {
        // Fallback: download JSON result
        const dataStr = JSON.stringify(result, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `${filename.split('.')[0]}_translated.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const toggleOriginalView = (index: number) => {
    setShowOriginal(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <div className="w-full space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">
          Translation Results ({results.length})
        </h2>
        <button
          onClick={() => {
            // Download all results as a zip file (simplified version)
            results.forEach(({ filename, result }) => {
              downloadResult(filename, result);
            });
          }}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Download className="h-4 w-4" />
          Download All
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map(({ filename, result, originalFile }, index) => (
          <div key={index} className="bg-white rounded-lg border shadow-sm overflow-hidden">
            {/* Image Preview */}
            <div className="relative aspect-square bg-gray-100">
              <img
                src={result.translatedImageUrl || URL.createObjectURL(originalFile)}
                alt={filename}
                className="w-full h-full object-cover"
                onError={(e) => {
                  // Fallback to original image if translated image fails to load
                  if (result.translatedImageUrl) {
                    console.warn(`Failed to load translated image: ${result.translatedImageUrl}`);
                    (e.target as HTMLImageElement).src = URL.createObjectURL(originalFile);
                  }
                }}
              />
              
              {/* Overlay Controls */}
              <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-30 transition-all duration-200 flex items-center justify-center opacity-0 hover:opacity-100">
                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedImage(index)}
                    className="p-2 bg-white rounded-full shadow-lg hover:bg-gray-50 transition-colors"
                  >
                    <ZoomIn className="h-5 w-5 text-gray-700" />
                  </button>
                  <button
                    onClick={() => toggleOriginalView(index)}
                    className="p-2 bg-white rounded-full shadow-lg hover:bg-gray-50 transition-colors"
                    title={showOriginal[index] ? "Show translated" : "Show original"}
                  >
                    {showOriginal[index] ? (
                      <EyeOff className="h-5 w-5 text-gray-700" />
                    ) : (
                      <Eye className="h-5 w-5 text-gray-700" />
                    )}
                  </button>
                </div>
              </div>

              {/* Show original image toggle */}
              {showOriginal[index] && (
                <div className="absolute inset-0">
                  <img
                    src={URL.createObjectURL(originalFile)}
                    alt={`${filename} (original)`}
                    className="w-full h-full object-cover"
                  />
                </div>
              )}

              {/* Translation Overlay - only show on original image */}
              {showOriginal[index] && result.translations && (
                <div className="absolute inset-0">
                  {result.translations.map((translation, tIndex) => (
                    <div
                      key={tIndex}
                      className="absolute border-2 border-blue-500 bg-blue-500 bg-opacity-20"
                      style={{
                        left: `${(translation.minX / 1000) * 100}%`,
                        top: `${(translation.minY / 1000) * 100}%`,
                        width: `${((translation.maxX - translation.minX) / 1000) * 100}%`,
                        height: `${((translation.maxY - translation.minY) / 1000) * 100}%`,
                      }}
                    >
                      <div className="absolute -top-6 left-0 bg-blue-500 text-white text-xs px-1 rounded">
                        {Object.values(translation.text)[1] || Object.values(translation.text)[0]}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* File Info */}
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-gray-900 truncate">{filename}</h3>
                {result.translatedImageUrl && !showOriginal[index] && (
                  <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                    Translated
                  </span>
                )}
                {showOriginal[index] && (
                  <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                    Original
                  </span>
                )}
              </div>
              
              <div className="flex items-center justify-between text-sm text-gray-500 mb-3">
                <span>{result.translations?.length || 0} text regions</span>
                <span>{(originalFile.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>

              {/* Translation Preview */}
              {result.translations && result.translations.length > 0 && (
                <div className="space-y-2 mb-3">
                  <h4 className="text-sm font-medium text-gray-700">Translations:</h4>
                  <div className="max-h-20 overflow-y-auto space-y-1">
                    {result.translations.slice(0, 3).map((translation, tIndex) => (
                      <div key={tIndex} className="text-xs bg-gray-50 p-2 rounded">
                        <div className="text-gray-600">
                          {Object.values(translation.text)[0]}
                        </div>
                        <div className="text-blue-600 font-medium">
                          {Object.values(translation.text)[1] || 'No translation'}
                        </div>
                      </div>
                    ))}
                    {result.translations.length > 3 && (
                      <div className="text-xs text-gray-500 text-center">
                        +{result.translations.length - 3} more...
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={() => downloadResult(filename, result)}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors text-sm"
                >
                  <Download className="h-4 w-4" />
                  Download
                </button>
                <button
                  onClick={() => setSelectedImage(index)}
                  className="px-3 py-2 border border-gray-300 rounded hover:bg-gray-50 transition-colors text-sm"
                >
                  View
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for detailed view */}
      {selectedImage !== null && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl max-h-full overflow-auto">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-medium">
                {results[selectedImage].filename}
              </h3>
              <button
                onClick={() => setSelectedImage(null)}
                className="p-2 hover:bg-gray-100 rounded-full"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="p-4">
              <div className="relative">
                <img
                  src={results[selectedImage].result.translatedImageUrl || URL.createObjectURL(results[selectedImage].originalFile)}
                  alt={results[selectedImage].filename}
                  className="max-w-full h-auto"
                />
                
                {/* Toggle button for modal */}
                <div className="absolute top-4 right-4">
                  <button
                    onClick={() => toggleOriginalView(selectedImage)}
                    className="p-2 bg-white rounded-full shadow-lg hover:bg-gray-50 transition-colors"
                    title={showOriginal[selectedImage] ? "Show translated" : "Show original"}
                  >
                    {showOriginal[selectedImage] ? (
                      <EyeOff className="h-5 w-5 text-gray-700" />
                    ) : (
                      <Eye className="h-5 w-5 text-gray-700" />
                    )}
                  </button>
                </div>

                {/* Show original image in modal when toggled */}
                {showOriginal[selectedImage] && (
                  <div className="absolute inset-0">
                    <img
                      src={URL.createObjectURL(results[selectedImage].originalFile)}
                      alt={`${results[selectedImage].filename} (original)`}
                      className="max-w-full h-auto"
                    />
                  </div>
                )}
                
                {/* Translation overlays for modal - only show on original */}
                {showOriginal[selectedImage] && results[selectedImage].result.translations?.map((translation, tIndex) => (
                  <div
                    key={tIndex}
                    className="absolute border-2 border-red-500 bg-red-500 bg-opacity-20"
                    style={{
                      left: `${(translation.minX / 1000) * 100}%`,
                      top: `${(translation.minY / 1000) * 100}%`,
                      width: `${((translation.maxX - translation.minX) / 1000) * 100}%`,
                      height: `${((translation.maxY - translation.minY) / 1000) * 100}%`,
                    }}
                  >
                    <div className="absolute -top-8 left-0 bg-red-500 text-white text-sm px-2 py-1 rounded max-w-xs">
                      {Object.values(translation.text)[1] || Object.values(translation.text)[0]}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}