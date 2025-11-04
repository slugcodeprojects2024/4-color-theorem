import React, { useRef } from 'react';

function ImageUploader({ onImageSelect }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
      }
      onImageSelect(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file);
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

