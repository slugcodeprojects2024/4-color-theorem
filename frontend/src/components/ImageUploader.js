import React, { useRef } from 'react';

function ImageUploader({ onImageSelect }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file (PNG, JPG, or JPEG)');
        return;
      }
      
      // Validate file size (10MB max)
      const maxSizeMB = 10;
      const maxSizeBytes = maxSizeMB * 1024 * 1024;
      if (file.size > maxSizeBytes) {
        alert(`Image file too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum size is ${maxSizeMB}MB.`);
        return;
      }
      
      // Validate image dimensions by loading it
      const img = new Image();
      img.onload = () => {
        const maxDimension = 4000;
        if (img.width > maxDimension || img.height > maxDimension) {
          alert(`Image dimensions too large (${img.width}x${img.height}). Maximum is ${maxDimension}x${maxDimension}px. The image will be automatically resized.`);
        }
        onImageSelect(file);
      };
      img.onerror = () => {
        alert('Invalid image file. Please use a valid PNG, JPG, or JPEG image.');
      };
      img.src = URL.createObjectURL(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        alert('Please drop an image file (PNG, JPG, or JPEG)');
        return;
      }
      
      // Validate file size
      const maxSizeMB = 10;
      const maxSizeBytes = maxSizeMB * 1024 * 1024;
      if (file.size > maxSizeBytes) {
        alert(`Image file too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum size is ${maxSizeMB}MB.`);
        return;
      }
      
      // Validate dimensions
      const img = new Image();
      img.onload = () => {
        const maxDimension = 4000;
        if (img.width > maxDimension || img.height > maxDimension) {
          alert(`Image dimensions too large (${img.width}x${img.height}). Maximum is ${maxDimension}x${maxDimension}px. The image will be automatically resized.`);
        }
        onImageSelect(file);
      };
      img.onerror = () => {
        alert('Invalid image file. Please use a valid PNG, JPG, or JPEG image.');
      };
      img.src = URL.createObjectURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <div 
      className="image-uploader"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      <button 
        className="upload-button"
        onClick={() => fileInputRef.current?.click()}
      >
        Choose Image
      </button>
      <p className="upload-hint">or drag and drop an image here</p>
      <p className="upload-info">Supports PNG, JPG, JPEG</p>
    </div>
  );
}

export default ImageUploader;

