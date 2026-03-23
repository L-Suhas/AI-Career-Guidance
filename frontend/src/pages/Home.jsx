import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="page" style={{ background: 'linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%)', minHeight: '100vh' }}>
      <div className="container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '90vh', textAlign: 'center' }}>

        {/* Logo / Icon */}
        <div style={{ width: 80, height: 80, background: 'rgba(255,255,255,0.2)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 24, fontSize: 36 }}>
          🎯
        </div>

        {/* Title */}
        <h1 style={{ color: '#fff', fontSize: 40, fontWeight: 800, marginBottom: 16, lineHeight: 1.2 }}>
          AI Based Career<br />Guidance System
        </h1>

        <p style={{ color: 'rgba(255,255,255,0.85)', fontSize: 18, maxWidth: 520, marginBottom: 40, lineHeight: 1.7 }}>
          Discover your perfect career path using AI. Answer a few questions about
          your skills, interests and goals — get personalized recommendations in seconds.
        </p>

        {/* Features */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 48, width: '100%', maxWidth: 600 }}>
          {[
            { icon: '🤖', text: 'AI-powered matching' },
            { icon: '📊', text: 'Skill gap analysis' },
            { icon: '🗺️', text: 'Career roadmaps' },
          ].map((f, i) => (
            <div key={i} style={{ background: 'rgba(255,255,255,0.15)', borderRadius: 12, padding: '16px 12px', color: '#fff' }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>{f.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>{f.text}</div>
            </div>
          ))}
        </div>

        {/* CTA Button */}
        <button
          className="btn-primary"
          onClick={() => navigate('/quiz')}
          style={{ fontSize: 18, padding: '16px 48px', background: '#fff', color: '#4f46e5', borderRadius: 12 }}
        >
          Start Career Assessment →
        </button>

        <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: 13, marginTop: 16 }}>
          Takes about 2 minutes · No sign-up required
        </p>
      </div>
    </div>
  );
}
