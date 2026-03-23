import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function Admin() {
  const navigate = useNavigate();
  const [weights, setWeights] = useState({});
  const [feedback, setFeedback] = useState({});
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchAll();
  }, []);

  async function fetchAll() {
    setLoading(true);
    try {
      const [rlRes, fbRes, sessRes] = await Promise.all([
        axios.get('http://localhost:8000/admin/rl-weights'),
        axios.get('http://localhost:8000/admin/feedback'),
        axios.get('http://localhost:8000/admin/sessions').catch(() => ({ data: { sessions: [] } })),
      ]);
      setWeights(rlRes.data.weights || {});
      setFeedback(fbRes.data.feedback_stats || {});
      setSessions(sessRes.data.sessions || []);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }

  const totalAccepts = Object.values(feedback).reduce((s, v) => s + (v.accept || 0), 0);
  const totalRejects = Object.values(feedback).reduce((s, v) => s + (v.reject || 0), 0);
  const topCareer = Object.entries(weights).sort((a, b) => b[1] - a[1])[0];
  const weightEntries = Object.entries(weights).sort((a, b) => b[1] - a[1]);

  const tabs = ['overview', 'rl weights', 'feedback', 'sessions'];

  if (loading) {
    return (
      <div className="page">
        <div className="loading-spinner">
          <div className="spinner" />
          <p style={{ color: '#4f46e5', fontWeight: 600 }}>Loading admin data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page" style={{ background: '#f0f4f8' }}>
      <div className="container" style={{ maxWidth: 900 }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
          <div>
            <h1 style={{ fontSize: 26, fontWeight: 800, color: '#1f2937' }}>
              Admin Dashboard
            </h1>
            <p style={{ color: '#6b7280', fontSize: 14, marginTop: 2 }}>
              AI Based Career Guidance System — real-time analytics
            </p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={fetchAll}
              style={{ padding: '9px 18px', borderRadius: 8, background: '#e0e7ff', color: '#3730a3', fontWeight: 600, fontSize: 14, border: 'none', cursor: 'pointer' }}
            >
              ↺ Refresh
            </button>
            <button
              onClick={() => navigate('/')}
              className="btn-secondary"
              style={{ padding: '9px 18px', fontSize: 14 }}
            >
              ← Back to app
            </button>
          </div>
        </div>

        {/* Stats cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 28 }}>
          {[
            { label: 'Total sessions', value: sessions.length || '—', color: '#4f46e5', bg: '#eef2ff' },
            { label: 'Total accepts', value: totalAccepts, color: '#10b981', bg: '#ecfdf5' },
            { label: 'Total rejects', value: totalRejects, color: '#ef4444', bg: '#fef2f2' },
            { label: 'Top RL career', value: topCareer ? topCareer[0].split(' ')[0] : '—', color: '#f59e0b', bg: '#fffbeb' },
          ].map((s, i) => (
            <div key={i} style={{ background: s.bg, borderRadius: 12, padding: '18px 16px', border: `1px solid ${s.color}22` }}>
              <div style={{ fontSize: 28, fontWeight: 800, color: s.color }}>{s.value}</div>
              <div style={{ fontSize: 13, color: '#6b7280', marginTop: 4 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 20, background: '#e5e7eb', borderRadius: 10, padding: 4 }}>
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                flex: 1, padding: '9px 0', borderRadius: 8, fontSize: 14, fontWeight: 600,
                border: 'none', cursor: 'pointer', textTransform: 'capitalize',
                background: activeTab === tab ? '#fff' : 'transparent',
                color: activeTab === tab ? '#4f46e5' : '#6b7280',
                boxShadow: activeTab === tab ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
                transition: 'all 0.15s',
              }}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Tab: Overview */}
        {activeTab === 'overview' && (
          <div className="card">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 20, color: '#1f2937' }}>System overview</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              <div style={{ background: '#f9fafb', borderRadius: 10, padding: 18 }}>
                <h3 style={{ fontSize: 15, fontWeight: 600, color: '#374151', marginBottom: 12 }}>
                  RL learning status
                </h3>
                {weightEntries.length === 0 ? (
                  <p style={{ color: '#9ca3af', fontSize: 14 }}>No feedback collected yet. Use the app and click "This fits me / Not for me" buttons.</p>
                ) : (
                  weightEntries.slice(0, 5).map(([career, weight]) => (
                    <div key={career} style={{ marginBottom: 10 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
                        <span style={{ color: '#374151', fontWeight: 500 }}>{career}</span>
                        <span style={{ color: weight >= 0 ? '#10b981' : '#ef4444', fontWeight: 700 }}>
                          {weight >= 0 ? '+' : ''}{(weight * 100).toFixed(1)}
                        </span>
                      </div>
                      <div style={{ height: 6, background: '#e5e7eb', borderRadius: 99, overflow: 'hidden' }}>
                        <div style={{
                          height: '100%', borderRadius: 99,
                          width: `${Math.abs(weight) * 100}%`,
                          background: weight >= 0 ? '#10b981' : '#ef4444',
                          transition: 'width 0.6s ease',
                        }} />
                      </div>
                    </div>
                  ))
                )}
              </div>

              <div style={{ background: '#f9fafb', borderRadius: 10, padding: 18 }}>
                <h3 style={{ fontSize: 15, fontWeight: 600, color: '#374151', marginBottom: 12 }}>
                  Feedback summary
                </h3>
                {Object.keys(feedback).length === 0 ? (
                  <p style={{ color: '#9ca3af', fontSize: 14 }}>No feedback yet. Complete the quiz and rate your results.</p>
                ) : (
                  Object.entries(feedback).slice(0, 5).map(([career, counts]) => (
                    <div key={career} style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 6 }}>{career}</div>
                      <div style={{ display: 'flex', gap: 8 }}>
                        <span style={{ background: '#d1fae5', color: '#065f46', padding: '2px 10px', borderRadius: 99, fontSize: 12, fontWeight: 600 }}>
                          ✓ {counts.accept || 0} accepts
                        </span>
                        <span style={{ background: '#fee2e2', color: '#991b1b', padding: '2px 10px', borderRadius: 99, fontSize: 12, fontWeight: 600 }}>
                          ✕ {counts.reject || 0} rejects
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab: RL Weights */}
        {activeTab === 'rl weights' && (
          <div className="card">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8, color: '#1f2937' }}>RL Q-table weights</h2>
            <p style={{ fontSize: 14, color: '#6b7280', marginBottom: 20 }}>
              Positive = users accepted this career. Negative = users rejected it. These values boost or reduce future scores.
            </p>
            {weightEntries.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#9ca3af' }}>
                <p style={{ fontSize: 16 }}>No RL data yet.</p>
                <p style={{ fontSize: 13, marginTop: 8 }}>Complete the quiz and click "This fits me / Not for me" to start training the RL agent.</p>
              </div>
            ) : (
              weightEntries.map(([career, weight]) => (
                <div key={career} style={{ marginBottom: 14, padding: '14px 16px', background: '#f9fafb', borderRadius: 10, border: `1px solid ${weight >= 0 ? '#d1fae5' : '#fee2e2'}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 600, fontSize: 15, color: '#1f2937' }}>{career}</span>
                    <span style={{ fontWeight: 800, fontSize: 18, color: weight >= 0 ? '#10b981' : '#ef4444' }}>
                      {weight >= 0 ? '+' : ''}{weight.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ height: 8, background: '#e5e7eb', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', borderRadius: 99,
                      width: `${Math.min(Math.abs(weight) * 100, 100)}%`,
                      background: weight >= 0 ? '#10b981' : '#ef4444',
                    }} />
                  </div>
                  <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 6 }}>
                    {weight > 0.05 ? 'Boosted — users consistently liked this' :
                     weight < -0.05 ? 'Penalized — users consistently rejected this' :
                     'Neutral — not enough feedback yet'}
                  </p>
                </div>
              ))
            )}
          </div>
        )}

        {/* Tab: Feedback */}
        {activeTab === 'feedback' && (
          <div className="card">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 20, color: '#1f2937' }}>Feedback per career</h2>
            {Object.keys(feedback).length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#9ca3af' }}>
                <p style={{ fontSize: 16 }}>No feedback recorded yet.</p>
                <p style={{ fontSize: 13, marginTop: 8 }}>Use the "This fits me / Not for me" buttons on results page.</p>
              </div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#f3f4f6' }}>
                    {['Career', 'Accepts', 'Rejects', 'Total', 'Satisfaction'].map(h => (
                      <th key={h} style={{ padding: '12px 16px', textAlign: 'left', fontSize: 13, fontWeight: 600, color: '#374151' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(feedback).map(([career, counts]) => {
                    const total = (counts.accept || 0) + (counts.reject || 0);
                    const satisfaction = total > 0 ? Math.round((counts.accept || 0) / total * 100) : 0;
                    return (
                      <tr key={career} style={{ borderTop: '1px solid #f3f4f6' }}>
                        <td style={{ padding: '12px 16px', fontSize: 14, fontWeight: 500, color: '#1f2937' }}>{career}</td>
                        <td style={{ padding: '12px 16px' }}>
                          <span style={{ background: '#d1fae5', color: '#065f46', padding: '3px 10px', borderRadius: 99, fontSize: 13, fontWeight: 600 }}>
                            ✓ {counts.accept || 0}
                          </span>
                        </td>
                        <td style={{ padding: '12px 16px' }}>
                          <span style={{ background: '#fee2e2', color: '#991b1b', padding: '3px 10px', borderRadius: 99, fontSize: 13, fontWeight: 600 }}>
                            ✕ {counts.reject || 0}
                          </span>
                        </td>
                        <td style={{ padding: '12px 16px', fontSize: 14, color: '#374151' }}>{total}</td>
                        <td style={{ padding: '12px 16px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ flex: 1, height: 8, background: '#e5e7eb', borderRadius: 99, overflow: 'hidden' }}>
                              <div style={{ height: '100%', width: `${satisfaction}%`, background: satisfaction >= 60 ? '#10b981' : '#ef4444', borderRadius: 99 }} />
                            </div>
                            <span style={{ fontSize: 13, fontWeight: 600, color: satisfaction >= 60 ? '#10b981' : '#ef4444', minWidth: 36 }}>
                              {satisfaction}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        )}

        {/* Tab: Sessions */}
        {activeTab === 'sessions' && (
          <div className="card">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8, color: '#1f2937' }}>User sessions</h2>
            <p style={{ fontSize: 14, color: '#6b7280', marginBottom: 20 }}>
              {sessions.length === 0
                ? 'Sessions are saved when PostgreSQL database is connected.'
                : `${sessions.length} sessions recorded`}
            </p>
            {sessions.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#9ca3af' }}>
                <p style={{ fontSize: 16 }}>No sessions in database yet.</p>
                <p style={{ fontSize: 13, marginTop: 8 }}>Set up PostgreSQL to enable session tracking.</p>
              </div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#f3f4f6' }}>
                    {['#', 'Name', 'Degree', 'Major', 'Top Career', 'Score', 'Date'].map(h => (
                      <th key={h} style={{ padding: '12px 16px', textAlign: 'left', fontSize: 13, fontWeight: 600, color: '#374151' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sessions.map(s => (
                    <tr key={s.id} style={{ borderTop: '1px solid #f3f4f6' }}>
                      <td style={{ padding: '12px 16px', fontSize: 13, color: '#9ca3af' }}>{s.id}</td>
                      <td style={{ padding: '12px 16px', fontSize: 14, fontWeight: 600, color: '#1f2937' }}>{s.name}</td>
                      <td style={{ padding: '12px 16px', fontSize: 13, color: '#374151' }}>{s.degree}</td>
                      <td style={{ padding: '12px 16px', fontSize: 13, color: '#374151' }}>{s.major}</td>
                      <td style={{ padding: '12px 16px' }}>
                        <span style={{ background: '#e0e7ff', color: '#3730a3', padding: '3px 10px', borderRadius: 99, fontSize: 12, fontWeight: 600 }}>
                          {s.top_career}
                        </span>
                      </td>
                      <td style={{ padding: '12px 16px', fontSize: 14, fontWeight: 700, color: '#4f46e5' }}>{s.top_score}%</td>
                      <td style={{ padding: '12px 16px', fontSize: 12, color: '#9ca3af' }}>
                        {new Date(s.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
