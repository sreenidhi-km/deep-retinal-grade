import React, { useState, useCallback } from 'react';
import axios from 'axios';

// API base URL
const API_URL = 'http://localhost:8000';

// DR Grade info
const GRADE_INFO = {
  0: { name: 'No DR', color: 'green', urgency: 'None', bg: 'bg-green-100 text-green-800 border-green-300' },
  1: { name: 'Mild', color: 'yellow', urgency: 'Routine', bg: 'bg-yellow-100 text-yellow-800 border-yellow-300' },
  2: { name: 'Moderate', color: 'orange', urgency: 'Non-urgent', bg: 'bg-orange-100 text-orange-800 border-orange-300' },
  3: { name: 'Severe', color: 'red', urgency: 'Urgent', bg: 'bg-red-100 text-red-800 border-red-300' },
  4: { name: 'Proliferative DR', color: 'darkred', urgency: 'Emergent', bg: 'bg-red-200 text-red-900 border-red-500' }
};

// Decision flag display config
const DECISION_FLAGS = {
  OK: { icon: '✅', label: 'OK', color: 'bg-green-100 text-green-800 border-green-300', desc: 'Safe for clinical guidance' },
  REVIEW: { icon: '⚠️', label: 'REVIEW', color: 'bg-yellow-100 text-yellow-800 border-yellow-300', desc: 'Clinician review required' },
  RETAKE: { icon: '📸', label: 'RETAKE', color: 'bg-orange-100 text-orange-800 border-orange-300', desc: 'Image quality insufficient' },
  OOD: { icon: '❓', label: 'OUT OF DISTRIBUTION', color: 'bg-red-100 text-red-800 border-red-300', desc: 'Image may not be fundus photo' },
  UNKNOWN: { icon: '⚠️', label: 'UNKNOWN', color: 'bg-gray-100 text-gray-800 border-gray-300', desc: 'Safety check unavailable' }
};

function App() {
  // State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [useTTA, setUseTTA] = useState(true);  // TTA toggle (default on)

  // Handle file selection
  const handleFileSelect = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setExplanation(null);
      setError(null);
    }
  }, []);

  // Handle drag and drop
  const handleDrop = useCallback((event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setExplanation(null);
      setError(null);
    }
  }, []);

  // Analyze image
  const analyzeImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      // Get prediction (with TTA if enabled)
      const formData = new FormData();
      formData.append('file', selectedFile);

      const predictEndpoint = useTTA ? '/predict-with-tta' : '/predict';
      const predResponse = await axios.post(`${API_URL}${predictEndpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      // Also get safety flags from /predict endpoint
      // When useTTA is false, predResponse already comes from /predict so reuse it
      let safetyData = {};
      if (predictEndpoint === '/predict') {
        safetyData = predResponse.data;
      } else {
        try {
          const safetyFormData = new FormData();
          safetyFormData.append('file', selectedFile);
          const safetyResponse = await axios.post(`${API_URL}/predict`, safetyFormData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          safetyData = safetyResponse.data;
        } catch (safetyErr) {
          console.warn('Safety check failed, continuing without safety flags');
        }
      }

      // Merge prediction + safety data
      setPrediction({
        ...predResponse.data,
        decision: safetyData.decision || 'UNKNOWN',
        decision_reason: safetyData.decision_reason || '',
        quality_score: safetyData.quality_score ?? null,
        quality_issues: safetyData.quality_issues || [],
        uncertainty: safetyData.uncertainty ?? null,
        entropy: safetyData.entropy ?? null,
        is_ood: safetyData.is_ood ?? null,
        tta_enabled: useTTA,
        tta_confidence: predResponse.data.tta_confidence ?? null,
      });

      // Get explanation
      const explainFormData = new FormData();
      explainFormData.append('file', selectedFile);
      const explainResponse = await axios.post(`${API_URL}/explain`, explainFormData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setExplanation(explainResponse.data);

      setActiveTab('results');
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Reset
  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setExplanation(null);
    setError(null);
    setActiveTab('upload');
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="gradient-bg text-white py-6 shadow-lg">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold">🔬 Deep Retina Grade</h1>
          <p className="text-blue-100 mt-1">AI-Powered Diabetic Retinopathy Screening</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Tabs */}
        <div className="flex space-x-4 mb-6">
          <button
            className={`px-6 py-2 rounded-lg font-medium transition-all ${
              activeTab === 'upload' 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-600 hover:bg-gray-100'
            }`}
            onClick={() => setActiveTab('upload')}
          >
            📤 Upload
          </button>
          <button
            className={`px-6 py-2 rounded-lg font-medium transition-all ${
              activeTab === 'results' 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-600 hover:bg-gray-100'
            }`}
            onClick={() => setActiveTab('results')}
            disabled={!prediction}
          >
            📊 Results
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mb-6">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Upload Zone */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">Upload Fundus Image</h2>
              
              <div
                className={`upload-zone ${selectedFile ? 'active' : ''}`}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => document.getElementById('file-input').click()}
              >
                {previewUrl ? (
                  <img 
                    src={previewUrl} 
                    alt="Preview" 
                    className="max-h-64 mx-auto rounded-lg"
                  />
                ) : (
                  <>
                    <div className="text-5xl mb-4">📷</div>
                    <p className="text-gray-600">
                      Drag & drop a fundus image here, or click to select
                    </p>
                    <p className="text-gray-400 text-sm mt-2">
                      Supports: PNG, JPG, JPEG
                    </p>
                  </>
                )}
              </div>

              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />

              {/* Action Buttons */}
              <div className="flex space-x-4 mt-6">
                <button
                  className="btn-primary flex-1"
                  onClick={analyzeImage}
                  disabled={!selectedFile || loading}
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                      </svg>
                      Analyzing...
                    </span>
                  ) : (
                    '🔍 Analyze Image'
                  )}
                </button>
                
                <button
                  className="btn-secondary"
                  onClick={handleReset}
                  disabled={loading}
                >
                  🔄 Reset
                </button>
              </div>

              {/* TTA Toggle */}
              <div className="mt-4 flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <span className="text-sm font-medium text-gray-700">Test-Time Augmentation (TTA)</span>
                  <p className="text-xs text-gray-500">Averages multiple augmented predictions for +2-3% accuracy</p>
                </div>
                <button
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    useTTA ? 'bg-blue-600' : 'bg-gray-300'
                  }`}
                  onClick={() => setUseTTA(!useTTA)}
                  role="switch"
                  aria-checked={useTTA}
                  aria-label="Use Test-Time Augmentation"
                >
                  <span
                    aria-hidden="true"
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      useTTA ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>

            {/* Instructions */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">Instructions</h2>
              
              <div className="space-y-4">
                <div className="flex items-start">
                  <span className="text-2xl mr-3">1️⃣</span>
                  <div>
                    <h3 className="font-medium">Upload Image</h3>
                    <p className="text-gray-600 text-sm">
                      Upload a retinal fundus photograph (color photo of the back of the eye)
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <span className="text-2xl mr-3">2️⃣</span>
                  <div>
                    <h3 className="font-medium">AI Analysis</h3>
                    <p className="text-gray-600 text-sm">
                      Our deep learning model analyzes the image for signs of diabetic retinopathy
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <span className="text-2xl mr-3">3️⃣</span>
                  <div>
                    <h3 className="font-medium">View Results</h3>
                    <p className="text-gray-600 text-sm">
                      Get DR grade, confidence score, visual explanation, and clinical recommendations
                    </p>
                  </div>
                </div>
              </div>

              {/* Disclaimer */}
              <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                <p className="text-yellow-800 text-sm">
                  <strong>⚠️ Disclaimer:</strong> This tool is for screening purposes only and 
                  does not replace professional medical diagnosis. Always consult an ophthalmologist 
                  for clinical decisions.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && prediction && (
          <div className="space-y-6">
            {/* Decision Flag Banner */}
            {prediction.decision && (
              <div className={`p-4 rounded-lg border-2 flex items-center space-x-3 ${
                DECISION_FLAGS[prediction.decision]?.color || 'bg-gray-100'
              }`}>
                <span className="text-2xl">{DECISION_FLAGS[prediction.decision]?.icon}</span>
                <div>
                  <span className="font-bold text-lg">
                    Decision: {DECISION_FLAGS[prediction.decision]?.label || prediction.decision}
                  </span>
                  <p className="text-sm opacity-80">
                    {prediction.decision_reason || DECISION_FLAGS[prediction.decision]?.desc}
                  </p>
                </div>
              </div>
            )}

            <div className="grid md:grid-cols-2 gap-6">
            {/* Prediction Results */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
              
              {/* Grade Badge */}
              <div className={`grade-badge grade-${prediction.grade} text-lg mb-4`}>
                Grade {prediction.grade}: {prediction.grade_name}
              </div>
              
              {/* Confidence */}
              <div className="mb-4">
                <div className="flex justify-between mb-1">
                  <span className="text-gray-600">Confidence</span>
                  <span className="font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* TTA Confidence (if available) */}
              {prediction.tta_confidence != null && (
                <div className="mb-4">
                  <div className="flex justify-between mb-1">
                    <span className="text-gray-600">TTA Confidence</span>
                    <span className="font-bold text-blue-700">{(prediction.tta_confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-blue-400 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${prediction.tta_confidence * 100}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Safety Metrics Panel */}
              <div className="mb-4 p-3 bg-gray-50 rounded-lg space-y-2">
                <h3 className="font-medium text-sm text-gray-700 mb-2">🛡️ Safety Metrics</h3>
                
                {/* Image Quality */}
                {prediction.quality_score != null && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Image Quality</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            prediction.quality_score >= 0.7 ? 'bg-green-500' :
                            prediction.quality_score >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${prediction.quality_score * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium w-12 text-right">
                        {(prediction.quality_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                )}

                {/* Quality Issues */}
                {prediction.quality_issues && prediction.quality_issues.length > 0 && (
                  <div className="flex items-start space-x-2">
                    <span className="text-sm text-gray-600">Issues:</span>
                    <div className="flex flex-wrap gap-1">
                      {prediction.quality_issues.map((issue, i) => (
                        <span key={i} className="text-xs px-2 py-0.5 bg-orange-100 text-orange-700 rounded-full">
                          {issue}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Uncertainty */}
                {prediction.uncertainty != null && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Uncertainty (MC Dropout)</span>
                    <span className={`text-sm font-medium ${
                      prediction.uncertainty > 0.15 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {prediction.uncertainty.toFixed(3)}
                    </span>
                  </div>
                )}

                {/* Entropy */}
                {prediction.entropy != null && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Entropy</span>
                    <span className={`text-sm font-medium ${
                      prediction.entropy > 1.5 ? 'text-red-600' : 
                      prediction.entropy > 0.8 ? 'text-yellow-600' : 'text-green-600'
                    }`}>
                      {prediction.entropy.toFixed(3)}
                    </span>
                  </div>
                )}

                {/* OOD Flag */}
                {prediction.is_ood && (
                  <div className="flex items-center space-x-2 text-red-600">
                    <span className="text-sm font-medium">⚠️ Out-of-Distribution Detected</span>
                  </div>
                )}
              </div>
              
              {/* Probabilities */}
              <div className="mb-4">
                <h3 className="font-medium mb-2">Class Probabilities</h3>
                <div className="space-y-2">
                  {Object.entries(prediction.probabilities).map(([grade, prob]) => (
                    <div key={grade} className="flex items-center">
                      <span className="w-32 text-sm text-gray-600">{grade}</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-2 mx-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium w-16 text-right">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Recommendation */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <h3 className="font-medium text-blue-900 mb-1">
                  📋 Recommendation
                </h3>
                <p className="text-blue-800 text-sm">{prediction.recommendation}</p>
              </div>
              
              {/* Urgency */}
              <div className="mt-4 flex items-center">
                <span className="text-gray-600 mr-2">Referral Urgency:</span>
                <span className={`font-bold ${
                  prediction.grade >= 3 ? 'text-red-600' : 
                  prediction.grade >= 2 ? 'text-orange-600' : 'text-green-600'
                }`}>
                  {prediction.referral_urgency}
                </span>
              </div>

              {/* TTA Mode indicator */}
              {prediction.tta_enabled && (
                <div className="mt-3 text-xs text-gray-500 flex items-center space-x-1">
                  <span className="inline-block w-2 h-2 bg-blue-400 rounded-full"></span>
                  <span>Predicted with Test-Time Augmentation</span>
                </div>
              )}
            </div>

            {/* XAI Explanation */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">🔍 AI Explanation (GradCAM)</h2>
              
              {explanation && (
                <>
                  <div className="mb-4">
                    <img 
                      src={`data:image/png;base64,${explanation.gradcam_base64}`}
                      alt="GradCAM Explanation"
                      className="w-full rounded-lg shadow"
                    />
                    <p className="text-sm text-gray-500 mt-2 text-center">
                      Highlighted regions show areas the AI focused on
                    </p>
                  </div>
                  
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <h3 className="font-medium mb-2">Interpretation</h3>
                    <p className="text-gray-700 text-sm">{explanation.interpretation}</p>
                  </div>
                  
                  {/* Color Legend */}
                  <div className="mt-4">
                    <h4 className="text-sm font-medium mb-2">Heatmap Legend</h4>
                    <div className="flex items-center space-x-4 text-xs">
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-blue-500 rounded mr-1"></div>
                        <span>Low attention</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-green-500 rounded mr-1"></div>
                        <span>Medium</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-yellow-500 rounded mr-1"></div>
                        <span>High</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-red-500 rounded mr-1"></div>
                        <span>Critical</span>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-6 mt-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-400">
            Deep Retina Grade v2.0 | ALP Project 2026
          </p>
          <p className="text-gray-500 text-sm mt-1">
            Powered by EfficientNet-B0 with GradCAM Explainability | Rate-limited API with Safety Flags
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
