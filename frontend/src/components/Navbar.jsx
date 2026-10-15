import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  FileCheck2,
  Home,
  Info,
  Microscope,
  ShieldAlert,
  UploadCloud,
} from 'lucide-react';

export default function Navbar({ health, hasActivePrediction }) {
  const modelReady = Boolean(health?.model_loaded);

  const navItems = [
    { to: '/', label: 'Home Dashboard', icon: Home },
    { to: '/upload', label: 'Image Upload', icon: UploadCloud },
    {
      to: '/results',
      label: 'Prediction Results',
      icon: FileCheck2,
      badge: hasActivePrediction ? 'Ready' : null,
    },
    { to: '/performance', label: 'Model Performance', icon: BarChart3 },
    { to: '/about', label: 'About Project', icon: Info },
  ];

  return (
    <header className="top-navbar">
      <div className="navbar-container">
        <NavLink to="/" className="brand-link">
          <div className="brand-icon-wrap">
            <Microscope size={22} />
          </div>
          <div className="brand-text">
            <span className="brand-title">DermaAI HAM10000</span>
            <span className="brand-subtitle">GEHU • Team CSE27-304 • Binary Lesion Triage</span>
          </div>
        </NavLink>

        <nav className="nav-links" aria-label="Primary Navigation">
          {navItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              >
                <IconComponent size={16} />
                <span>{item.label}</span>
                {item.badge && <span className="nav-pill">{item.badge}</span>}
              </NavLink>
            );
          })}
        </nav>

        <div className="navbar-status">
          <div
            className={`status-indicator ${modelReady ? 'status-online' : 'status-warning'}`}
            title={health?.status_message || 'Checking backend status...'}
          >
            {modelReady ? <Activity size={14} /> : <ShieldAlert size={14} />}
            <span>{modelReady ? 'EfficientNetB0 Ready' : 'Model Untrained / Offline'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
