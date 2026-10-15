import React from 'react';
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  FileBarChart,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import DisclaimerBanner from '../components/DisclaimerBanner';
import { resolveBackendAssetUrl } from '../services/api';

export default function ModelPerformancePage({ modelInfo, loading, error, onRefresh }) {
  const evaluation = modelInfo?.evaluation;
  const datasetAudit = modelInfo?.dataset_audit;
  const mappingDetails = modelInfo?.original_to_binary_mapping || {};

  // Prepare 7-class dataset distribution chart data if audit is available
  const sevenClassChartData = datasetAudit?.original_seven_class_distribution
    ? Object.entries(datasetAudit.original_seven_class_distribution).map(([dx, info]) => ({
        dx,
        count: info.count,
        binaryClass: info.mapped_binary_class,
        fill: info.mapped_binary_label === 1 ? '#dc2626' : '#0d9488',
      }))
    : [];

  // Prepare training history chart data if evaluation/training history is available
  const trainingHistoryRaw = evaluation?.training_history || {};
  const epochsCount = trainingHistoryRaw.accuracy?.length || 0;
  const historyChartData = Array.from({ length: epochsCount }, (_, i) => ({
    epoch: i + 1,
    trainAcc: Number(((trainingHistoryRaw.accuracy?.[i] ?? 0) * 100).toFixed(2)),
    valAcc: Number(((trainingHistoryRaw.val_accuracy?.[i] ?? 0) * 100).toFixed(2)),
    trainLoss: Number((trainingHistoryRaw.loss?.[i] ?? 0).toFixed(4)),
    valLoss: Number((trainingHistoryRaw.val_loss?.[i] ?? 0).toFixed(4)),
  }));

  const rocPoints = evaluation?.curves?.roc_curve || [];
  const prPoints = evaluation?.curves?.pr_curve || [];

  const metricCards = evaluation
    ? [
        {
          label: 'Test Accuracy',
          value: `${(evaluation.accuracy * 100).toFixed(2)}%`,
          raw: evaluation.accuracy.toFixed(4),
          desc: 'Overall proportion of correct predictions on untouched test split',
        },
        {
          label: 'Malignant Recall (Sensitivity)',
          value: `${(evaluation.recall * 100).toFixed(2)}%`,
          raw: evaluation.recall.toFixed(4),
          desc: 'Clinically critical sensitivity for detecting malignant lesions (TP / (TP + FN))',
          highlight: true,
        },
        {
          label: 'Specificity (Benign Recall)',
          value: `${(evaluation.specificity * 100).toFixed(2)}%`,
          raw: evaluation.specificity.toFixed(4),
          desc: 'Calculated from confusion matrix: TN / (TN + FP)',
        },
        {
          label: 'Malignant Precision (PPV)',
          value: `${(evaluation.precision * 100).toFixed(2)}%`,
          raw: evaluation.precision.toFixed(4),
          desc: 'Proportion of positive malignant predictions that are truly malignant',
        },
        {
          label: 'F1-Score (Malignant)',
          value: `${(evaluation.f1_score * 100).toFixed(2)}%`,
          raw: evaluation.f1_score.toFixed(4),
          desc: 'Harmonic mean of Malignant Precision and Recall',
        },
        {
          label: 'ROC-AUC Score',
          value: evaluation.roc_auc.toFixed(4),
          raw: `PR-AUC: ${evaluation.pr_auc?.toFixed(4) ?? 'N/A'}`,
          desc: 'Area under the Receiver Operating Characteristic curve across thresholds',
        },
      ]
    : [];

  const sampleCounts = evaluation?.sample_counts;
  const cm = evaluation?.confusion_matrix;

  return (
    <div className="page-container">
      <div className="page-header results-header-row">
        <div>
          <h1>Untouched Test-Set Model Performance &amp; Audit</h1>
          <p>
            All metrics and figures are loaded dynamically from actual test-set evaluation artifacts
            (never hardcoded).
          </p>
        </div>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onRefresh}
          disabled={loading}
        >
          <RefreshCw size={16} className={loading ? 'spin-icon' : ''} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      <DisclaimerBanner compact />

      {error && (
        <div className="alert-box alert-error">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {!evaluation ? (
        <section className="card empty-evaluation-card">
          <div className="dropzone-icon-circle large">
            <FileBarChart size={34} />
          </div>
          <h2>Test-Set Evaluation Artifacts Not Yet Generated</h2>
          <p>
            In compliance with strict academic integrity rules, this dashboard never displays
            hardcoded or fabricated metrics. Run the dataset preparation, training, and evaluation
            pipeline on the HAM10000 dataset to populate live test-set metrics and plots:
          </p>
          <pre className="code-snippet-block">
{`.venv/bin/python scripts/prepare_dataset.py
.venv/bin/python scripts/train_model.py
.venv/bin/python scripts/evaluate_model.py`}
          </pre>
        </section>
      ) : (
        <>
          {/* 6 Core Metric Cards */}
          <div className="metrics-kpi-grid">
            {metricCards.map((m) => (
              <div
                key={m.label}
                className={`card kpi-card ${m.highlight ? 'kpi-highlight' : ''}`}
              >
                <span className="kpi-label">{m.label}</span>
                <div className="kpi-value-row">
                  <span className="kpi-value">{m.value}</span>
                  <span className="kpi-raw font-mono">{m.raw}</span>
                </div>
                <p className="kpi-desc">{m.desc}</p>
              </div>
            ))}
          </div>

          {/* Test Split Sample Breakdown & Confusion Matrix */}
          <div className="dashboard-grid-2col">
            <section className="card">
              <div className="card-header">
                <Database size={20} className="card-header-icon" />
                <div>
                  <h2>Test Split Sample &amp; Error Accounting</h2>
                  <p className="card-subtitle">
                    Evaluated on untouched test split at threshold ={' '}
                    {evaluation.decision_threshold}
                  </p>
                </div>
              </div>

              {sampleCounts && (
                <div className="spec-list">
                  <div className="spec-row">
                    <span className="spec-label">Total Untouched Test Samples</span>
                    <span className="spec-value font-mono">
                      {sampleCounts.total_test_samples.toLocaleString()}
                    </span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-label">Benign Test Samples (Class 0)</span>
                    <span className="spec-value font-mono text-benign">
                      {sampleCounts.benign_test_samples.toLocaleString()}
                    </span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-label">Malignant Test Samples (Class 1)</span>
                    <span className="spec-value font-mono text-malignant">
                      {sampleCounts.malignant_test_samples.toLocaleString()}
                    </span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-label">Correctly Classified Samples (TN + TP)</span>
                    <span className="spec-value font-mono status-text-ready">
                      <CheckCircle2 size={15} />{' '}
                      {sampleCounts.correctly_classified_samples.toLocaleString()}
                    </span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-label">Incorrectly Classified Samples (FP + FN)</span>
                    <span className="spec-value font-mono">
                      {sampleCounts.incorrectly_classified_samples.toLocaleString()}
                    </span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-label">
                      False Negatives (Missed Malignant Cases)
                    </span>
                    <span className="spec-value font-mono text-malignant">
                      <ShieldAlert size={15} />{' '}
                      {sampleCounts.false_negatives_missed_malignant.toLocaleString()}
                    </span>
                  </div>
                </div>
              )}
            </section>

            {/* Confusion Matrix Table */}
            <section className="card">
              <div className="card-header">
                <BarChart3 size={20} className="card-header-icon" />
                <div>
                  <h2>Confusion Matrix (Test Set)</h2>
                  <p className="card-subtitle">
                    Specificity = TN / (TN + FP) = {(evaluation.specificity * 100).toFixed(2)}%
                  </p>
                </div>
              </div>

              {cm && (
                <div className="cm-visual-grid">
                  <div className="cm-cell cm-tn">
                    <span className="cm-tag">True Negative (TN)</span>
                    <strong className="cm-count">{cm.tn}</strong>
                    <span className="cm-sub">Actual Benign → Pred Benign</span>
                  </div>
                  <div className="cm-cell cm-fp">
                    <span className="cm-tag">False Positive (FP)</span>
                    <strong className="cm-count">{cm.fp}</strong>
                    <span className="cm-sub">Actual Benign → Pred Malignant</span>
                  </div>
                  <div className="cm-cell cm-fn">
                    <span className="cm-tag">False Negative (FN)</span>
                    <strong className="cm-count">{cm.fn}</strong>
                    <span className="cm-sub">Actual Malignant → Pred Benign</span>
                  </div>
                  <div className="cm-cell cm-tp">
                    <span className="cm-tag">True Positive (TP)</span>
                    <strong className="cm-count">{cm.tp}</strong>
                    <span className="cm-sub">Actual Malignant → Pred Malignant</span>
                  </div>
                </div>
              )}
            </section>
          </div>

          {/* Interactive ROC & Precision-Recall Curves */}
          <div className="dashboard-grid-2col">
            <section className="card">
              <div className="card-header">
                <Activity size={20} className="card-header-icon" />
                <div>
                  <h2>ROC Curve (AUC = {evaluation.roc_auc.toFixed(4)})</h2>
                  <p className="card-subtitle">
                    True Positive Rate (Sensitivity) vs. False Positive Rate (1 - Specificity)
                  </p>
                </div>
              </div>
              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height={270}>
                  <AreaChart data={rocPoints} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="fpr"
                      type="number"
                      domain={[0, 1]}
                      label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis
                      dataKey="tpr"
                      type="number"
                      domain={[0, 1]}
                      label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="tpr"
                      stroke="#0d9488"
                      fill="#99f6e4"
                      fillOpacity={0.45}
                      strokeWidth={2.5}
                      name="TPR (Sensitivity)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="card">
              <div className="card-header">
                <Activity size={20} className="card-header-icon" />
                <div>
                  <h2>Precision-Recall Curve (PR-AUC = {evaluation.pr_auc?.toFixed(4)})</h2>
                  <p className="card-subtitle">
                    Malignant Precision vs. Malignant Recall across decision thresholds
                  </p>
                </div>
              </div>
              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height={270}>
                  <AreaChart data={prPoints} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="recall"
                      type="number"
                      domain={[0, 1]}
                      label={{ value: 'Recall (Sensitivity)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis
                      dataKey="precision"
                      type="number"
                      domain={[0, 1]}
                      label={{ value: 'Precision (PPV)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="precision"
                      stroke="#dc2626"
                      fill="#fecaca"
                      fillOpacity={0.45}
                      strokeWidth={2.5}
                      name="Precision"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>

          {/* Training vs Validation History Curves */}
          {historyChartData.length > 0 && (
            <div className="dashboard-grid-2col">
              <section className="card">
                <div className="card-header">
                  <BarChart3 size={20} className="card-header-icon" />
                  <div>
                    <h2>Training vs. Validation Accuracy</h2>
                    <p className="card-subtitle">Epoch-by-epoch convergence history</p>
                  </div>
                </div>
                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={historyChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" />
                      <YAxis unit="%" />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="trainAcc"
                        name="Train Accuracy (%)"
                        stroke="#0d9488"
                        strokeWidth={2.2}
                      />
                      <Line
                        type="monotone"
                        dataKey="valAcc"
                        name="Validation Accuracy (%)"
                        stroke="#2563eb"
                        strokeWidth={2.2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="card">
                <div className="card-header">
                  <BarChart3 size={20} className="card-header-icon" />
                  <div>
                    <h2>Training vs. Validation Loss</h2>
                    <p className="card-subtitle">Binary Cross-Entropy Loss history</p>
                  </div>
                </div>
                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={historyChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="trainLoss"
                        name="Train Loss"
                        stroke="#dc2626"
                        strokeWidth={2.2}
                      />
                      <Line
                        type="monotone"
                        dataKey="valLoss"
                        name="Validation Loss"
                        stroke="#d97706"
                        strokeWidth={2.2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>
            </div>
          )}

          {/* Saved High-Resolution Figures Gallery from results/figures/ */}
          <section className="card">
            <div className="card-header">
              <FileBarChart size={20} className="card-header-icon" />
              <div>
                <h2>Saved Evaluation Figures (`results/figures/`)</h2>
                <p className="card-subtitle">
                  High-resolution plots generated by scripts/evaluate_model.py
                </p>
              </div>
            </div>
            <div className="figures-gallery-grid">
              {[
                { file: 'confusion_matrix.png', title: 'Confusion Matrix Heatmap' },
                { file: 'roc_curve.png', title: 'ROC Curve' },
                { file: 'precision_recall_curve.png', title: 'Precision-Recall Curve' },
                { file: 'training_val_accuracy.png', title: 'Training vs. Validation Accuracy' },
                { file: 'training_val_loss.png', title: 'Training vs. Validation Loss' },
                { file: 'class_distribution.png', title: 'Dataset Class Distribution' },
              ].map((fig) => (
                <div key={fig.file} className="figure-card">
                  <h4>{fig.title}</h4>
                  <img
                    src={resolveBackendAssetUrl(`/api/figures/${fig.file}`)}
                    alt={fig.title}
                    loading="lazy"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                    }}
                  />
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {/* Dataset Class Distribution & Explicit Mapping Table */}
      {sevenClassChartData.length > 0 && (
        <section className="card">
          <div className="card-header">
            <Database size={20} className="card-header-icon" />
            <div>
              <h2>Audited HAM10000 7-Class Distribution</h2>
              <p className="card-subtitle">
                Verified image counts mapped to Benign (Teal) and Malignant (Red)
              </p>
            </div>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={270}>
              <BarChart data={sevenClassChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="dx" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Verified Images" radius={[6, 6, 0, 0]}>
                  {sevenClassChartData.map((entry, idx) => (
                    <Cell key={`dx-${idx}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section className="card">
        <div className="card-header">
          <Database size={20} className="card-header-icon" />
          <div>
            <h2>Explicit 7-to-2 Diagnostic Label Mapping Table</h2>
            <p className="card-subtitle">
              Clinical classification rationale and ambiguity audit for HAM10000
            </p>
          </div>
        </div>

        <div className="table-responsive">
          <table className="academic-table">
            <thead>
              <tr>
                <th>Code (dx)</th>
                <th>Original Diagnostic Category</th>
                <th>Target Binary Class</th>
                <th>Clinical Category &amp; Ambiguity Note</th>
              </tr>
            </thead>
            <tbody>
              {Object.values(mappingDetails).map((item) => (
                <tr key={item.original_code}>
                  <td className="font-mono font-bold">{item.original_code}</td>
                  <td>{item.full_name}</td>
                  <td>
                    <span
                      className={`badge ${
                        item.binary_label === 1 ? 'badge-malignant' : 'badge-benign'
                      }`}
                    >
                      {item.binary_label}: {item.binary_class}
                    </span>
                  </td>
                  <td>
                    <strong>{item.clinical_category}:</strong> {item.clinical_rationale}{' '}
                    <em className="text-muted">({item.ambiguity_note})</em>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
