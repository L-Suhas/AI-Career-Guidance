import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const OUTLOOK_COLOR = {
  'Excellent': 'badge-green',
  'Good': 'badge-blue',
  'Fair': 'badge-amber',
};

const RANK_COLORS = ['#4f46e5', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444'];

function ScoreBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 13, color: '#4b5563' }}>
        <span>{label}</span>
        <span style={{ fontWeight: 600, color }}>{value}%</span>
      </div>
      <div style={{ height: 8, background: '#f3f4f6', borderRadius: 99, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${value}%`, background: color, borderRadius: 99 }} />
      </div>
    </div>
  );
}

function CareerCard({ rec, isExpanded, onToggle, onFeedback, feedbackSent }) {
  const color = RANK_COLORS[rec.rank - 1];
  const acceptKey = `${rec.title}-accept`;
  const rejectKey = `${rec.title}-reject`;

  return (
    <div style={{ background: '#fff', borderRadius: 16, boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)', marginBottom: 20, overflow: 'hidden', border: `2px solid ${isExpanded ? color : 'transparent'}` }}>
      <div style={{ padding: '20px 24px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 16 }} onClick={onToggle}>
        <div style={{ width: 48, height: 48, borderRadius: '50%', background: color, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: 18, flexShrink: 0 }}>
          #{rec.rank}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <h3 style={{ fontSize: 18, fontWeight: 700, color: '#1f2937' }}>{rec.title}</h3>
            <span className={`badge ${OUTLOOK_COLOR[rec.job_outlook] || 'badge-blue'}`}>{rec.job_outlook} outlook</span>
            {rec.rank === 1 && <span style={{ background: '#fef3c7', color: '#92400e', padding: '3px 10px', borderRadius: 99, fontSize: 12, fontWeight: 600 }}>Top Pick</span>}
          </div>
          <p style={{ fontSize: 13, color: '#6b7280', marginTop: 2 }}>{rec.industry}</p>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div style={{ fontSize: 28, fontWeight: 800, color }}>{rec.match_score}%</div>
          <div style={{ fontSize: 12, color: '#9ca3af' }}>match</div>
        </div>
        <div style={{ color: '#9ca3af', fontSize: 20, transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>▾</div>
      </div>

      {isExpanded && (
        <div style={{ padding: '0 24px 24px', borderTop: '1px solid #f3f4f6' }}>
          <div style={{ marginTop: 20, marginBottom: 20 }}>
            <h4 style={{ fontSize: 14, fontWeight: 600, color: '#374151', marginBottom: 12 }}>Match breakdown</h4>
            <ScoreBar label="Overall match" value={rec.match_score} color={color} />
            <ScoreBar label="Skill alignment" value={rec.skill_match_percent} color="#0ea5e9" />
            <ScoreBar label="Personality fit" value={rec.trait_match_percent} color="#10b981" />
          </div>

          <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 10, padding: '14px 16px', marginBottom: 20 }}>
            <p style={{ fontSize: 14, color: '#166534', lineHeight: 1.6 }}>
              <strong>Why this fits you:</strong> {rec.explanation}
            </p>
          </div>

          <p style={{ fontSize: 14, color: '#4b5563', lineHeight: 1.7, marginBottom: 20 }}>{rec.description}</p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Salary range', value: rec.salary_range },
              { label: 'Work environment', value: rec.work_environment },
              { label: 'Education', value: rec.education_required },
              { label: 'Job outlook', value: rec.job_outlook },
            ].map((item, i) => (
              <div key={i} style={{ background: '#f9fafb', borderRadius: 8, padding: '12px 14px' }}>
                <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{item.label}</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1f2937' }}>{item.value}</div>
              </div>
            ))}
          </div>

          <div style={{ marginBottom: 20 }}>
            <h4 style={{ fontSize: 14, fontWeight: 600, color: '#374151', marginBottom: 10 }}>Skills you need</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {rec.skills_required.map(skill => (
                <span key={skill} className="tag">{skill}</span>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10, paddingTop: 16, borderTop: '1px solid #f3f4f6' }}>
            <button
              onClick={(e) => { e.stopPropagation(); onFeedback(rec.title, 'accept'); }}
              style={{
                flex: 1, padding: '12px', borderRadius: 8, fontWeight: 700, fontSize: 15, cursor: 'pointer',
                border: feedbackSent[acceptKey] ? '2px solid #10b981' : '2px solid #a7f3d0',
                background: feedbackSent[acceptKey] ? '#10b981' : '#d1fae5',
                color: feedbackSent[acceptKey] ? '#fff' : '#065f46',
              }}
            >
              {feedbackSent[acceptKey] ? 'Saved!' : 'This fits me'}
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onFeedback(rec.title, 'reject'); }}
              style={{
                flex: 1, padding: '12px', borderRadius: 8, fontWeight: 700, fontSize: 15, cursor: 'pointer',
                border: feedbackSent[rejectKey] ? '2px solid #ef4444' : '2px solid #fca5a5',
                background: feedbackSent[rejectKey] ? '#ef4444' : '#fee2e2',
                color: feedbackSent[rejectKey] ? '#fff' : '#991b1b',
              }}
            >
              {feedbackSent[rejectKey] ? 'Saved!' : 'Not for me'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const { data, profile } = location.state || {};
  const [expanded, setExpanded] = useState(0);
  const [feedbackSent, setFeedbackSent] = useState({});
  const [feedbackError, setFeedbackError] = useState('');

  if (!data) {
    return (
      <div className="page">
        <div className="container" style={{ textAlign: 'center', paddingTop: 80 }}>
          <p style={{ fontSize: 18, color: '#6b7280' }}>No results found.</p>
          <button className="btn-primary" style={{ marginTop: 20 }} onClick={() => navigate('/quiz')}>Take the quiz</button>
        </div>
      </div>
    );
  }

  async function handleFeedback(careerTitle, action) {
    const key = `${careerTitle}-${action}`;
    if (feedbackSent[key]) return;

    try {
      const payload = {
        user_name: profile.name || 'anonymous',
        career_title: careerTitle,
        action: action,
      };
      console.log('Sending feedback:', payload);
      const response = await axios.post('http://localhost:8000/feedback', payload, {
        headers: { 'Content-Type': 'application/json' }
      });
      console.log('Feedback response:', response.data);
      setFeedbackSent(prev => ({ ...prev, [key]: true }));
      setFeedbackError('');
    } catch (err) {
      console.error('Feedback error:', err.response?.data || err.message);
      setFeedbackError('Could not save feedback. Make sure the backend is running.');
    }
  }

  const topMatch = data.recommendations[0];

  return (
    <div className="page" style={{ background: '#f0f4f8' }}>
      <div className="container">
        <div style={{ background: 'linear-gradient(135deg, #4f46e5, #0ea5e9)', borderRadius: 16, padding: '28px 32px', marginBottom: 32, color: '#fff' }}>
          <p style={{ fontSize: 14, opacity: 0.8, marginBottom: 6 }}>Results for</p>
          <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>{data.student_name}</h1>
          <p style={{ opacity: 0.85, fontSize: 16, marginBottom: 16 }}>{profile.degree} in {profile.major}</p>
          <div style={{ background: 'rgba(255,255,255,0.2)', borderRadius: 10, padding: '12px 16px', display: 'inline-block' }}>
            <span style={{ fontSize: 15 }}>Top match: <strong>{topMatch.title}</strong> — {topMatch.match_score}% fit</span>
          </div>
        </div>

        {feedbackError && (
          <div className="error-box" style={{ marginBottom: 16 }}>{feedbackError}</div>
        )}

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1f2937' }}>Your top {data.recommendations.length} career matches</h2>
          <span style={{ fontSize: 13, color: '#6b7280' }}>Analyzed {data.total_careers_analyzed} careers</span>
        </div>

        {data.recommendations.map((rec, i) => (
          <CareerCard
            key={rec.career_id}
            rec={rec}
            isExpanded={expanded === i}
            onToggle={() => setExpanded(expanded === i ? -1 : i)}
            onFeedback={handleFeedback}
            feedbackSent={feedbackSent}
          />
        ))}

        <div style={{ textAlign: 'center', marginTop: 32, paddingBottom: 32 }}>
          <button className="btn-secondary" onClick={() => navigate('/quiz')}>Retake assessment</button>
          <p style={{ fontSize: 13, color: '#9ca3af', marginTop: 12 }}>
            Click "This fits me" or "Not for me" on each card to train the AI
          </p>
          <button
            onClick={() => navigate('/admin')}
            style={{ marginTop: 12, background: 'none', border: 'none', color: '#4f46e5', fontSize: 13, cursor: 'pointer', textDecoration: 'underline' }}
          >
            View Admin Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
