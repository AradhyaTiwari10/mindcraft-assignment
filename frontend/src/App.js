import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [maskedImage, setMaskedImage] = useState('');
  const [comparisonImage, setComparisonImage] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setMaskedImage('');
    setComparisonImage('');
    setStats(null);
    setError('');
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setPreviewUrl(reader.result);
      reader.readAsDataURL(file);
    } else {
      setPreviewUrl('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    setError('');
    setMaskedImage('');
    setComparisonImage('');
    setStats(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('mask_method', 'black_box');
      formData.append('show_indicators', 'true');
      formData.append('padding', '5');
      const response = await axios.post('http://localhost:8000/mask', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMaskedImage(response.data.masked_image);
      setComparisonImage(response.data.comparison_image);
      setStats(response.data.statistics);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error processing image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-8 px-2">
      <h1 className="text-3xl font-bold text-primary-700 mb-4">PII Masking Demo</h1>
      <form onSubmit={handleSubmit} className="bg-white shadow rounded p-6 flex flex-col items-center w-full max-w-md">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="mb-4"
        />
        {previewUrl && (
          <img src={previewUrl} alt="Preview" className="w-72 h-auto mb-4 rounded shadow" />
        )}
        <button
          type="submit"
          disabled={!selectedFile || loading}
          className="bg-primary-600 hover:bg-primary-700 text-white font-semibold py-2 px-6 rounded disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Mask PII'}
        </button>
        {error && <div className="text-danger-600 mt-2">{error}</div>}
      </form>
      {comparisonImage && (
        <div className="mt-8 w-full max-w-4xl flex flex-col items-center">
          <h2 className="text-xl font-semibold mb-2">Comparison</h2>
          <img
            src={`data:image/jpeg;base64,${comparisonImage}`}
            alt="Comparison"
            className="w-full max-w-2xl rounded shadow border"
          />
        </div>
      )}
      {maskedImage && (
        <div className="mt-8 w-full max-w-2xl flex flex-col items-center">
          <h2 className="text-xl font-semibold mb-2">Masked Image</h2>
          <img
            src={`data:image/jpeg;base64,${maskedImage}`}
            alt="Masked"
            className="w-full rounded shadow border"
          />
          <a
            href={`data:image/jpeg;base64,${maskedImage}`}
            download="masked_image.jpg"
            className="mt-2 text-primary-700 underline"
          >
            Download Masked Image
          </a>
        </div>
      )}
      {stats && (
        <div className="mt-8 bg-white shadow rounded p-4 w-full max-w-md">
          <h3 className="font-semibold mb-2">Masking Statistics</h3>
          <div>Total Regions Masked: <b>{stats.total_regions}</b></div>
          <div>Total Characters Masked: <b>{stats.total_characters}</b></div>
          <div className="mt-2">
            <b>By Type:</b>
            <ul className="list-disc ml-6">
              {Object.entries(stats.by_type).map(([type, count]) => (
                <li key={type}>{type}: {count}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
      <footer className="mt-12 text-gray-400 text-sm">&copy; 2024 PII Masking System</footer>
    </div>
  );
}

export default App; 