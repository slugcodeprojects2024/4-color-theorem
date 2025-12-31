import React, { useState, useEffect } from 'react';
import './PresetManager.css';

const DEFAULT_PRESETS = [
  {
    id: 'mosaic-sunset',
    name: 'Mosaic Sunset',
    description: 'Warm colors with stained glass effect',
    settings: { style: 'sunset', stainedGlass: true, lineArt: { enabled: false } },
    isDefault: true,
  },
  {
    id: 'soft-pastel',
    name: 'Soft Pastel',
    description: 'Gentle colors, no effects',
    settings: { style: 'pastel', stainedGlass: false, lineArt: { enabled: false } },
    isDefault: true,
  },
  {
    id: 'coloring-book',
    name: 'Coloring Book',
    description: 'Convert photo to line art with vibrant colors',
    settings: {
      style: 'vibrant',
      stainedGlass: false,
      lineArt: { enabled: true, lineThickness: 'medium', detailLevel: 'detailed', contrast: 1.0 },
    },
    isDefault: true,
  },
  {
    id: 'stained-glass-classic',
    name: 'Classic Stained Glass',
    description: 'Bold colors with prominent lead lines',
    settings: { style: 'vibrant', stainedGlass: true, lineArt: { enabled: false } },
    isDefault: true,
  },
  {
    id: 'ocean-dreams',
    name: 'Ocean Dreams',
    description: 'Cool ocean tones with glass effect',
    settings: { style: 'ocean', stainedGlass: true, lineArt: { enabled: false } },
    isDefault: true,
  },
  {
    id: 'neon-pop',
    name: 'Neon Pop',
    description: 'Bright fluorescent colors',
    settings: { style: 'neon', stainedGlass: false, lineArt: { enabled: false } },
    isDefault: true,
  },
];

const STORAGE_KEY = 'fourColorPresets';

function PresetManager({ currentSettings, onApplyPreset, onSavePreset }) {
  const [presets, setPresets] = useState([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDescription, setNewPresetDescription] = useState('');
  const [selectedPresetId, setSelectedPresetId] = useState(null);

  useEffect(() => {
    const savedPresets = localStorage.getItem(STORAGE_KEY);
    if (savedPresets) {
      try {
        const userPresets = JSON.parse(savedPresets);
        setPresets([...DEFAULT_PRESETS, ...userPresets]);
      } catch (e) {
        setPresets(DEFAULT_PRESETS);
      }
    } else {
      setPresets(DEFAULT_PRESETS);
    }
  }, []);

  const savePresetsToStorage = (allPresets) => {
    const userPresets = allPresets.filter(p => !p.isDefault);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(userPresets));
  };

  const handleApplyPreset = (preset) => {
    setSelectedPresetId(preset.id);
    onApplyPreset(preset.settings);
  };

  const handleSaveNewPreset = () => {
    if (!newPresetName.trim()) return;

    const newPreset = {
      id: `user-${Date.now()}`,
      name: newPresetName.trim(),
      description: newPresetDescription.trim() || 'Custom preset',
      settings: { ...currentSettings },
      isDefault: false,
    };

    const updatedPresets = [...presets, newPreset];
    setPresets(updatedPresets);
    savePresetsToStorage(updatedPresets);
    setNewPresetName('');
    setNewPresetDescription('');
    setShowSaveDialog(false);
  };

  const handleDeletePreset = (presetId, e) => {
    e.stopPropagation();
    if (!window.confirm('Delete this preset?')) return;
    const updatedPresets = presets.filter(p => p.id !== presetId);
    setPresets(updatedPresets);
    savePresetsToStorage(updatedPresets);
  };

  const defaultPresets = presets.filter(p => p.isDefault);
  const userPresets = presets.filter(p => !p.isDefault);

  return (
    <div className="preset-manager">
      <div className="preset-header">
        <h3>Presets</h3>
        <button className="preset-action-btn save" onClick={() => setShowSaveDialog(true)}>
          + Save Current
        </button>
      </div>

      <div className="preset-section">
        <h4>Quick Styles</h4>
        <div className="preset-grid">
          {defaultPresets.map(preset => (
            <button
              key={preset.id}
              className={`preset-card ${selectedPresetId === preset.id ? 'active' : ''}`}
              onClick={() => handleApplyPreset(preset)}
            >
              <span className="preset-name">{preset.name}</span>
              <span className="preset-desc">{preset.description}</span>
            </button>
          ))}
        </div>
      </div>

      {userPresets.length > 0 && (
        <div className="preset-section">
          <h4>My Presets</h4>
          <div className="preset-grid">
            {userPresets.map(preset => (
              <button
                key={preset.id}
                className={`preset-card user-preset ${selectedPresetId === preset.id ? 'active' : ''}`}
                onClick={() => handleApplyPreset(preset)}
              >
                <span className="preset-name">{preset.name}</span>
                <span className="preset-desc">{preset.description}</span>
                <button className="preset-delete" onClick={(e) => handleDeletePreset(preset.id, e)}>×</button>
              </button>
            ))}
          </div>
        </div>
      )}

      {showSaveDialog && (
        <div className="preset-dialog-overlay" onClick={() => setShowSaveDialog(false)}>
          <div className="preset-dialog" onClick={e => e.stopPropagation()}>
            <h4>Save Preset</h4>
            <input
              type="text"
              value={newPresetName}
              onChange={(e) => setNewPresetName(e.target.value)}
              placeholder="Preset name"
              autoFocus
            />
            <input
              type="text"
              value={newPresetDescription}
              onChange={(e) => setNewPresetDescription(e.target.value)}
              placeholder="Description (optional)"
            />
            <div className="dialog-actions">
              <button onClick={() => setShowSaveDialog(false)}>Cancel</button>
              <button onClick={handleSaveNewPreset}>Save</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PresetManager;

