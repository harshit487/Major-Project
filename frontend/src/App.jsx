import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomeDashboard from './pages/HomeDashboard';
import ImageUploadPage from './pages/ImageUploadPage';
import PredictionResultsPage from './pages/PredictionResultsPage';
import ModelPerformancePage from './pages/ModelPerformancePage';
import AboutProjectPage from './pages/AboutProjectPage';
import { useModelInfo } from './hooks/useModelInfo';

export default function App() {
  const { health, modelInfo, loading, error, refresh } = useModelInfo();
  const [quickSelectedFile, setQuickSelectedFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar
          health={health}
          hasActivePrediction={Boolean(predictionResult)}
        />

        <main className="main-content">
          <Routes>
            <Route
              path="/"
              element={
                <HomeDashboard
                  modelInfo={modelInfo}
                  health={health}
                  onQuickSelectFile={(file) => setQuickSelectedFile(file)}
                />
              }
            />
            <Route
              path="/upload"
              element={
                <ImageUploadPage
                  initialFile={quickSelectedFile}
                  onClearInitialFile={() => setQuickSelectedFile(null)}
                  onPredictionSuccess={(result) => setPredictionResult(result)}
                  health={health}
                />
              }
            />
            <Route
              path="/results"
              element={
                <PredictionResultsPage
                  predictionResult={predictionResult}
                  onResetPrediction={() => setPredictionResult(null)}
                />
              }
            />
            <Route
              path="/performance"
              element={
                <ModelPerformancePage
                  modelInfo={modelInfo}
                  loading={loading}
                  error={error}
                  onRefresh={refresh}
                />
              }
            />
            <Route path="/about" element={<AboutProjectPage />} />
          </Routes>
        </main>

        <footer className="app-footer">
          <div className="footer-inner">
            <span>
              <strong>Skin Disease Classification Using Deep Learning on the HAM10000 Dataset</strong>{' '}
              • Graphic Era Hill University, Dehradun
            </span>
            <span>
              Team ID: <code>CSE27-304</code> • Guide: Assistant Prof. Nishant Bhandari • Educational
              Research Prototype
            </span>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}
