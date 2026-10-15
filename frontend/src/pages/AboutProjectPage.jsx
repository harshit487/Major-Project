import React from 'react';
import {
  Award,
  BookOpen,
  BrainCircuit,
  Compass,
  Eye,
  GraduationCap,
  Layers,
  ShieldAlert,
  Target,
  Users,
} from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function AboutProjectPage() {
  const teamMembers = [
    { name: 'Kamal Kant Bisht', rollNo: '2318930', role: 'Machine Learning & Dataset Pipeline' },
    { name: 'Lakshya Tiwari', rollNo: '2319035', role: 'Deep Learning & Grad-CAM Explainability' },
    { name: 'Harshit Srikoti', rollNo: '2318847', role: 'Full-Stack Architecture & Integration' },
    { name: 'Manish Tinsola', rollNo: '2319073', role: 'Model Evaluation & Reporting System' },
  ];

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1>About the Project &amp; Academic Team</h1>
          <p>
            Skin Disease Classification Using Deep Learning on the HAM10000 Dataset • Graphic Era
            Hill University, Dehradun
          </p>
        </div>
      </div>

      {/* University & Project Team Card */}
      <section className="card team-showcase-card">
        <div className="team-institution-banner">
          <div className="institution-badge-icon">
            <GraduationCap size={32} />
          </div>
          <div>
            <span className="eyebrow-tag">FINAL-YEAR B.TECH AI &amp; ML MAJOR PROJECT</span>
            <h2>Graphic Era Hill University, Dehradun, Uttarakhand</h2>
            <p>
              Department of Computer Science and Engineering • Project Team ID:{' '}
              <strong className="font-mono">CSE27-304</strong>
            </p>
          </div>
        </div>

        <div className="guide-and-team-grid">
          <div className="faculty-guide-card">
            <div className="guide-header">
              <Award size={20} />
              <span>Under the Guidance of</span>
            </div>
            <h3>Nishant Bhandari</h3>
            <p>Assistant Professor</p>
            <span className="dept-sub">
              Department of Computer Science and Engineering, Graphic Era Hill University
            </span>
          </div>

          <div className="team-members-grid">
            {teamMembers.map((member) => (
              <div key={member.rollNo} className="member-card">
                <div className="member-avatar">
                  <Users size={18} />
                </div>
                <div>
                  <h4>{member.name}</h4>
                  <span className="member-roll font-mono">Roll No: {member.rollNo}</span>
                  <p className="member-role">{member.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <DisclaimerBanner />

      {/* Problem Statement & Objectives */}
      <div className="dashboard-grid-2col">
        <section className="card">
          <div className="card-header">
            <BookOpen size={20} className="card-header-icon" />
            <div>
              <h2>1. Problem Statement</h2>
              <p className="card-subtitle">Clinical challenge in early skin lesion triage</p>
            </div>
          </div>
          <p className="prose-text">
            Cutaneous malignancies—including malignant melanoma, basal cell carcinoma, and squamous
            cell carcinoma in situ—represent a major global public health challenge. Visual
            differentiation between benign pigmented lesions (such as melanocytic nevi or
            seborrheic keratoses) and malignant neoplasms under dermoscopy is complex due to
            overlapping morphological structures. Furthermore, black-box deep learning classifiers
            often suffer from data leakage across multi-image lesions and lack transparent visual
            explanations, limiting educational and clinical trust.
          </p>
        </section>

        <section className="card">
          <div className="card-header">
            <Target size={20} className="card-header-icon" />
            <div>
              <h2>2. Project Objectives</h2>
              <p className="card-subtitle">Core engineering &amp; research deliverables</p>
            </div>
          </div>
          <ul className="prose-list">
            <li>
              Audit the official <strong>HAM10000</strong> dataset and construct a clinically
              grounded binary mapping: <strong>Benign (0)</strong> vs.{' '}
              <strong>Malignant (1)</strong>.
            </li>
            <li>
              Eliminate data leakage using <strong>group-aware stratified splitting</strong> on{' '}
              <code>lesion_id</code> (70% Train / 15% Validation / 15% Test).
            </li>
            <li>
              Develop a transfer-learning pipeline with <strong>EfficientNetB0</strong> and
              validation-calibrated decision thresholds prioritizing malignant recall.
            </li>
            <li>
              Integrate <strong>Grad-CAM</strong> explainability to highlight spatial regions
              influencing model predictions and generate downloadable PDF research reports.
            </li>
          </ul>
        </section>
      </div>

      {/* Dataset, Binary Approach, Architecture & Explainable AI */}
      <div className="dashboard-grid-2col">
        <section className="card">
          <div className="card-header">
            <Layers size={20} className="card-header-icon" />
            <div>
              <h2>3. Dataset &amp; Binary Classification Approach</h2>
              <p className="card-subtitle">HAM10000 (Tschandl et al., Scientific Data 2018)</p>
            </div>
          </div>
          <p className="prose-text">
            The HAM10000 (&ldquo;Human Against Machine with 10000 training images&rdquo;) dataset
            comprises 10,015 dermatoscopic images collected from Austrian and Australian clinical
            centers across 7 diagnostic categories:
          </p>
          <ul className="prose-list">
            <li>
              <strong>Benign (Class 0 — Non-cancerous):</strong> Melanocytic nevi (<code>nv</code>),
              Benign keratosis-like lesions (<code>bkl</code>), Dermatofibroma (<code>df</code>),
              and Vascular lesions (<code>vasc</code>).
            </li>
            <li>
              <strong>Malignant (Class 1 — Cancerous / Neoplastic):</strong> Melanoma (
              <code>mel</code>), Basal cell carcinoma (<code>bcc</code>), and Actinic keratoses /
              intraepithelial carcinoma / Bowen&apos;s disease (<code>akiec</code>).
            </li>
            <li>
              <strong>Ambiguity Handling:</strong> While early actinic keratosis is precancerous
              dysplasia, the <code>akiec</code> category in HAM10000 includes Bowen&apos;s disease
              (squamous cell carcinoma <em>in situ</em>). Grouping <code>akiec</code> under
              Malignant ensures that in-situ carcinomas and pre-malignant lesions requiring biopsy
              are never dismissed as benign.
            </li>
          </ul>
        </section>

        <section className="card">
          <div className="card-header">
            <BrainCircuit size={20} className="card-header-icon" />
            <div>
              <h2>4. Model Architecture &amp; Grad-CAM XAI</h2>
              <p className="card-subtitle">EfficientNetB0 + Gradient-weighted Class Activation</p>
            </div>
          </div>
          <ul className="prose-list">
            <li>
              <strong>Backbone:</strong> Pretrained <code>EfficientNetB0</code> initialized on
              ImageNet accepting <code>224 × 224 × 3</code> RGB inputs.
            </li>
            <li>
              <strong>Classification Head:</strong> <code>GlobalAveragePooling2D</code> →{' '}
              <code>Dropout(0.35)</code> → <code>Dense(1, activation=&apos;sigmoid&apos;)</code>{' '}
              estimating <span className="font-mono">P(Malignant | Image)</span>.
            </li>
            <li>
              <strong>Two-Phase Training:</strong> Phase 1 trains the classification head with a
              frozen backbone; Phase 2 fine-tunes the top convolutional blocks with{' '}
              <code>BatchNormalization</code> frozen and inverse-frequency class weights.
            </li>
            <li>
              <strong>Grad-CAM Explainability:</strong> Computes gradients of the predicted class
              score with respect to the <code>top_activation</code> convolutional layer ($7 \times 7
              \times 1280$), pooling channel weights to render spatial heatmaps.
            </li>
          </ul>
        </section>
      </div>

      {/* Limitations & Future Scope */}
      <div className="dashboard-grid-2col">
        <section className="card">
          <div className="card-header">
            <ShieldAlert size={20} className="card-header-icon" />
            <div>
              <h2>5. Clinical &amp; Technical Limitations</h2>
              <p className="card-subtitle">Critical boundaries of this research system</p>
            </div>
          </div>
          <ul className="prose-list">
            <li>
              <strong>Population &amp; Skin-Tone Bias:</strong> HAM10000 was collected primarily
              from fair-skinned populations (Fitzpatrick types I–III) in Austria and Australia;
              performance does not generalize reliably to darker skin tones without multi-ethnic
              clinical validation.
            </li>
            <li>
              <strong>Dermoscopic vs. Macroscopic Images:</strong> The model is trained on
              magnified, polarized/fluid-immersion dermatoscopic images—not standard smartphone or
              macroscopic camera photos.
            </li>
            <li>
              <strong>Class Imbalance &amp; Uncalibrated Probabilities:</strong> Benign lesions
              outnumber malignant lesions by ~4.13:1 in HAM10000. Raw sigmoid outputs are
              uncalibrated estimates rather than true clinical post-test probabilities.
            </li>
            <li>
              <strong>Artifacts:</strong> Dermoscopic rulers, ink markings, air bubbles, and dense
              hair can attract spurious convolutional attention in Grad-CAM.
            </li>
          </ul>
        </section>

        <section className="card">
          <div className="card-header">
            <Compass size={20} className="card-header-icon" />
            <div>
              <h2>6. Future Scope &amp; Enhancements</h2>
              <p className="card-subtitle">Directions for clinical translation research</p>
            </div>
          </div>
          <ul className="prose-list">
            <li>
              <strong>Multi-Modal Fusion:</strong> Incorporate patient metadata (age, anatomical
              site, sex, lesion evolution history) alongside dermoscopic imagery.
            </li>
            <li>
              <strong>Hair Removal &amp; Artifact Inpainting:</strong> Integrate automated DullRazor
              morphological hair removal and lesion segmentation prior to classification.
            </li>
            <li>
              <strong>Probability Calibration &amp; Uncertainty Estimation:</strong> Apply
              temperature scaling and Monte Carlo Dropout for calibrated confidence intervals.
            </li>
            <li>
              <strong>Cross-Dataset Diverse Evaluation:</strong> Validate across diverse cohorts
              (ISIC 2020, Fitzpatrick17k, DDI — Diverse Dermatology Images).
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
