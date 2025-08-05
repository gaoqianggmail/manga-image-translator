'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Image as ImageIcon, AlertCircle } from 'lucide-react';

interface ImageFile {
  file: File;
  preview: string;
  id: string;
}

interface ImageUploaderProps {
  onImagesChange: (files: File[]) => void;
  maxFiles?: number;
  maxSize?: number; // in MB
}

export default function ImageUploader({ 
  onImagesChange, 
  maxFiles = 10, 
  maxSize = 10 
}: ImageUploaderProps) {
  const [images, setImages] = useState<ImageFile[]>([]);
  const [error, setError] = useState<string>('');

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError('');

    // Handle rejected files
    if (rejectedFiles.length > 0) {
      const errors = rejectedFiles.map(({ errors }) => errors[0]?.message).join(', ');
      setError(`Some files were rejected: ${errors}`);
    }

    // Process accepted files
    const newImages: ImageFile[] = acceptedFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file),
      id: Math.random().toString(36).substr(2, 9)
    }));

    setImages(prev => {
      const updated = [...prev, ...newImages];
      if (updated.length > maxFiles) {
        setError(`Maximum ${maxFiles} files allowed`);
        return prev;
      }
      
      // Update parent component
      onImagesChange(updated.map(img => img.file));
      return updated;
    });
  }, [maxFiles, onImagesChange]);

  const removeImage = (id: string) => {
    setImages(prev => {
      const updated = prev.filter(img => img.id !== id);
      // Revoke object URL to prevent memory leaks
      const removed = prev.find(img => img.id === id);
      if (removed) {
        URL.revokeObjectURL(removed.preview);
      }
      
      onImagesChange(updated.map(img => img.file));
      return updated;
    });
  };

  const clearAll = () => {
    images.forEach(img => URL.revokeObjectURL(img.preview));
    setImages([]);
    onImagesChange([]);
    setError('');
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    maxSize: maxSize * 1024 * 1024, // Convert MB to bytes
    maxFiles: maxFiles - images.length,
    disabled: images.length >= maxFiles
  });

  return (
    <div className="w-full space-y-4">
      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          upload-area border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : images.length >= maxFiles 
              ? 'border-gray-300 bg-gray-50 cursor-not-allowed' 
              : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }
        `}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        
        {images.length >= maxFiles ? (
          <p className="text-gray-500">Maximum files reached ({maxFiles})</p>
        ) : isDragActive ? (
          <p className="text-blue-600 font-medium">Drop the images here...</p>
        ) : (
          <div>
            <p className="text-gray-600 font-medium mb-2">
              Drag & drop manga images here, or click to select
            </p>
            <p className="text-sm text-gray-500">
              Supports: JPG, PNG, GIF, WebP (max {maxSize}MB each, {maxFiles} files total)
            </p>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
          <AlertCircle className="h-5 w-5 text-red-500" />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {/* Image Preview Grid */}
      {images.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">
              Selected Images ({images.length})
            </h3>
            <button
              onClick={clearAll}
              className="text-sm text-red-600 hover:text-red-800 font-medium"
            >
              Clear All
            </button>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {images.map((image) => (
              <div key={image.id} className="relative group">
                <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                  <img
                    src={image.preview}
                    alt="Preview"
                    className="w-full h-full object-cover"
                  />
                </div>
                
                {/* Remove Button */}
                <button
                  onClick={() => removeImage(image.id)}
                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 
                           opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
                >
                  <X className="h-4 w-4" />
                </button>
                
                {/* File Info */}
                <div className="mt-2 text-xs text-gray-500 truncate">
                  <div className="flex items-center gap-1">
                    <ImageIcon className="h-3 w-3" />
                    {image.file.name}
                  </div>
                  <div className="text-gray-400">
                    {(image.file.size / 1024 / 1024).toFixed(1)} MB
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}