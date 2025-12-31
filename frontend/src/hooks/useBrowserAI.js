/**
 * React hook for browser-based AI analysis
 */

import { useState, useEffect, useRef, useCallback } from 'react';

export function useBrowserAI() {
  const [isLoading, setIsLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [error, setError] = useState(null);
  const workerRef = useRef(null);
  const pendingRequests = useRef({});
  
  // Initialize worker
  useEffect(() => {
    workerRef.current = new Worker(
      new URL('../workers/aiWorker.js', import.meta.url),
      { type: 'module' }
    );
    
    workerRef.current.onmessage = (e) => {
      const { type, id, ...data } = e.data;
      
      switch (type) {
        case 'progress':
          if (data.progress?.progress) {
            setLoadProgress(Math.round(data.progress.progress * 100));
          }
          break;
          
        case 'status':
          if (data.loaded !== undefined) {
            setIsModelLoaded(data.loaded);
          }
          break;
          
        case 'result':
          if (pendingRequests.current[id]) {
            pendingRequests.current[id].resolve(data.result);
            delete pendingRequests.current[id];
          }
          setIsLoading(false);
          setIsModelLoaded(true);
          break;
          
        case 'error':
          if (pendingRequests.current[id]) {
            pendingRequests.current[id].reject(new Error(data.error));
            delete pendingRequests.current[id];
          }
          setError(data.error);
          setIsLoading(false);
          break;
          
        default:
          // Unknown message type
          break;
      }
    };
    
    return () => {
      workerRef.current?.terminate();
    };
  }, []);
  
  // Analyze image
  const analyzeImage = useCallback(async (imageFile) => {
    if (!workerRef.current) {
      throw new Error('AI worker not initialized');
    }
    
    setIsLoading(true);
    setError(null);
    
    // Convert file to base64
    const imageData = await fileToDataUrl(imageFile);
    
    const id = Date.now().toString();
    
    return new Promise((resolve, reject) => {
      pendingRequests.current[id] = { resolve, reject };
      
      workerRef.current.postMessage({
        type: 'analyze',
        imageData,
        id
      });
    });
  }, []);
  
  return {
    analyzeImage,
    isLoading,
    isModelLoaded,
    loadProgress,
    error
  };
}

// Helper to convert file to data URL
function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default useBrowserAI;

