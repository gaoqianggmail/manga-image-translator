'use client';

import React from 'react';
import { CheckCircle, Clock, AlertCircle, Loader2 } from 'lucide-react';

export interface TranslationStatus {
  id: string;
  filename: string;
  status: 'pending' | 'uploading' | 'translating' | 'completed' | 'error';
  progress: number;
  error?: string;
  result?: any;
}

interface TranslationProgressProps {
  translations: TranslationStatus[];
  queueSize?: number;
}

export default function TranslationProgress({ translations, queueSize }: TranslationProgressProps) {
  if (translations.length === 0) return null;

  const getStatusIcon = (status: TranslationStatus['status']) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-5 w-5 text-gray-400" />;
      case 'uploading':
      case 'translating':
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
    }
  };

  const getStatusText = (status: TranslationStatus['status']) => {
    switch (status) {
      case 'pending':
        return 'Waiting in queue...';
      case 'uploading':
        return 'Uploading...';
      case 'translating':
        return 'Translating...';
      case 'completed':
        return 'Completed';
      case 'error':
        return 'Failed';
    }
  };

  const getProgressColor = (status: TranslationStatus['status']) => {
    switch (status) {
      case 'uploading':
        return 'bg-blue-500';
      case 'translating':
        return 'bg-purple-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-300';
    }
  };

  const completedCount = translations.filter(t => t.status === 'completed').length;
  const errorCount = translations.filter(t => t.status === 'error').length;
  const totalCount = translations.length;

  return (
    <div className="w-full space-y-4">
      {/* Overall Progress */}
      <div className="bg-white rounded-lg border p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-medium text-gray-900">Translation Progress</h3>
          <div className="text-sm text-gray-500">
            {completedCount}/{totalCount} completed
            {errorCount > 0 && (
              <span className="text-red-500 ml-2">({errorCount} failed)</span>
            )}
          </div>
        </div>
        
        {/* Overall Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${(completedCount / totalCount) * 100}%` }}
          />
        </div>

        {/* Queue Info */}
        {queueSize !== undefined && queueSize > 0 && (
          <div className="text-sm text-gray-600 mb-3">
            <Clock className="inline h-4 w-4 mr-1" />
            {queueSize} items in server queue
          </div>
        )}
      </div>

      {/* Individual File Progress */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {translations.map((translation) => (
          <div
            key={translation.id}
            className="bg-white rounded-lg border p-4 transition-all duration-200"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                {getStatusIcon(translation.status)}
                <div>
                  <p className="font-medium text-gray-900 truncate max-w-xs">
                    {translation.filename}
                  </p>
                  <p className="text-sm text-gray-500">
                    {getStatusText(translation.status)}
                  </p>
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-sm font-medium text-gray-900">
                  {translation.progress}%
                </div>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div
                className={`h-1.5 rounded-full transition-all duration-300 ${getProgressColor(translation.status)}`}
                style={{ width: `${translation.progress}%` }}
              />
            </div>

            {/* Error Message */}
            {translation.error && (
              <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                {translation.error}
              </div>
            )}

            {/* Success Info */}
            {translation.status === 'completed' && translation.result && (
              <div className="mt-2 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-700">
                Found {translation.result.translations?.length || 0} text regions
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}