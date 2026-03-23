import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const STEPS = [
  { id: 'name',       label: 'Your name',           type: 'text',     placeholder: 'e.g. Suhas' },
  { id: 'degree',     label: 'Your degree',          type: 'select',   options: ['B.Tech', 'B.E.', 'BCA', 'BBA', 'B.Sc', 'MBA', 'MCA', 'M.Tech', 'Other'] },
  { id: 'major',      label: 'Your major / branch',  type: 'text',     placeholder: 'e.g. Computer Science' },
  { id: 'interests',  label: 'Your interests',       type: 'tags',     placeholder: 'e.g. coding, design, data' },
  { id: 'skills',     label: 'Your current skills',  type: 'tags',     placeholder: 'e.g. Python, Excel, Figma' },
  { id: 'personality_traits', label: 'Your personality traits', type: 'chips',
    options: ['Analytical','Creative','Logical','Communicative','Curious','Detail-oriented',
               'Leadership','Problem solver','Empathetic','Strategic','Organized','Patient'] },
  { id: 'work_preference', label: 'Preferred work style', type: 'chips',
    options: ['Remote', 'Office', 'Hybrid'] },
  { id: 'goals',      label: 'Your career goals',    type: 'textarea', placeholder: 'e.g. I want to build AI products that help people...' },
];

function TagInput({ value, onChange, placeholder }) {
  const [input, setInput] = useState('');
  const tags = value || [];

  function add() {
    const trimmed = input.trim();
    if (trimmed && !tags.includes(trimmed)) {
      onChange([...tags, trimmed]);
    }
    setInput('');
  }

  function remove(tag) {
    onChange(tags.filter(t => t !== tag));
  }

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
        {tags.map(tag => (
          <span key={tag} style={{ background: '#e0e7ff', color: '#3730a3', padding: '5px 12px', borderRadius: 99, fontSize: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
            {tag}
            <button onClick={() => remove(tag)} style={{ background: 'none', color: '#6366f1', fontSize: 16, lineHeight: 1, padding: 0 }}>×</button>
          </span>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); add(); }}}
          placeholder={placeholder}
          style={{ flex: 1, padding: '12px 16px', borderRadius: 8, border: '2px solid #e5e7eb', fontSize: 15, transition: 'border 0.2s' }}
          onFocus={e => e.target.style.borderColor = '#6366f1'}
          onBlur={e => e.target.style.borderColor = '#e5e7eb'}
        />
        <button onClick={add} style={{ background: '#4f46e5', color: '#fff', padding: '12px 20px', borderRadius: 8, fontSize: 15, fontWeight: 600 }}>
          Add
        </button>
      </div>
      <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 6 }}>Press Enter or comma to add each item</p>
    </div>
  );
}

function ChipsInput({ value, onChange, options, multi = true }) {
  const selected = value || (multi ? [] : '');

  function toggle(option) {
    if (!multi) { onChange(option); return; }
    if (selected.includes(option)) {
      onChange(selected.filter(o => o !== option));
    } else {
      onChange([...selected, option]);
    }
  }

  function isSelected(option) {
    return multi ? selected.includes(option) : selected === option;
  }

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
      {options.map(option => (
        <button
          key={option}
          onClick={() => toggle(option)}
          style={{
            padding: '10px 20px', borderRadius: 99, fontSize: 14, fontWeight: 500,
            border: '2px solid',
            borderColor: isSelected(option) ? '#4f46e5' : '#e5e7eb',
            background: isSelected(option) ? '#4f46e5' : '#fff',
            color: isSelected(option) ? '#fff' : '#4b5563',
            transition: 'all 0.15s',
          }}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

export default function Quiz() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const current = STEPS[step];
  const progress = ((step) / STEPS.length) * 100;

  function setValue(val) {
    setAnswers(prev => ({ ...prev, [current.id]: val }));
  }

  function getValue() {
    return answers[current.id] || (current.type === 'tags' || current.type === 'chips' ? [] : '');
  }

  function isValid() {
    const val = getValue();
    if (current.type === 'tags' || (current.type === 'chips' && current.id === 'personality_traits')) {
      return Array.isArray(val) && val.length > 0;
    }
    return val && val.toString().trim().length > 0;
  }

  async function handleSubmit() {
    setLoading(true);
    setError('');
    try {
      const payload = {
        name: answers.name || '',
        degree: answers.degree || '',
        major: answers.major || '',
        interests: answers.interests || [],
        skills: answers.skills || [],
        personality_traits: answers.personality_traits || [],
        work_preference: Array.isArray(answers.work_preference) ? answers.work_preference[0] : (answers.work_preference || 'Remote'),
        goals: answers.goals || '',
      };
      const res = await axios.post('http://localhost:8000/recommend', payload);
      navigate('/results', { state: { data: res.data, profile: payload } });
    } catch (err) {
      setError('Could not connect to the server. Make sure your backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  }

  function next() {
    if (step < STEPS.length - 1) setStep(s => s + 1);
    else handleSubmit();
  }

  function back() {
    if (step > 0) setStep(s => s - 1);
  }

  if (loading) {
    return (
      <div className="page">
        <div className="loading-spinner">
          <div className="spinner" />
          <p style={{ color: '#4f46e5', fontWeight: 600, fontSize: 18 }}>Analyzing your profile with AI...</p>
          <p style={{ color: '#9ca3af', fontSize: 14 }}>This takes a few seconds</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="container" style={{ maxWidth: 620 }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#1f2937', marginBottom: 4 }}>
            🎯 AI Career Guidance System
          </h1>
          <p style={{ color: '#9ca3af', fontSize: 14 }}>
            Step {step + 1} of {STEPS.length}
          </p>
        </div>

        {/* Progress */}
        <div className="progress-bar" style={{ marginBottom: 32 }}>
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>

        {/* Question Card */}
        <div className="card">
          <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1f2937', marginBottom: 24 }}>
            {current.label}
          </h2>

          {error && <div className="error-box">{error}</div>}

          {/* Render the right input type */}
          {current.type === 'text' && (
            <input
              value={getValue()}
              onChange={e => setValue(e.target.value)}
              placeholder={current.placeholder}
              onKeyDown={e => e.key === 'Enter' && isValid() && next()}
              style={{ width: '100%', padding: '14px 16px', borderRadius: 8, border: '2px solid #e5e7eb', fontSize: 16, transition: 'border 0.2s' }}
              onFocus={e => e.target.style.borderColor = '#6366f1'}
              onBlur={e => e.target.style.borderColor = '#e5e7eb'}
              autoFocus
            />
          )}

          {current.type === 'select' && (
            <select
              value={getValue()}
              onChange={e => setValue(e.target.value)}
              style={{ width: '100%', padding: '14px 16px', borderRadius: 8, border: '2px solid #e5e7eb', fontSize: 16, background: '#fff' }}
            >
              <option value="">Select your degree...</option>
              {current.options.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
          )}

          {current.type === 'tags' && (
            <TagInput value={getValue()} onChange={setValue} placeholder={current.placeholder} />
          )}

          {current.type === 'chips' && (
            <ChipsInput
              value={getValue()}
              onChange={setValue}
              options={current.options}
              multi={current.id !== 'work_preference'}
            />
          )}

          {current.type === 'textarea' && (
            <textarea
              value={getValue()}
              onChange={e => setValue(e.target.value)}
              placeholder={current.placeholder}
              rows={4}
              style={{ width: '100%', padding: '14px 16px', borderRadius: 8, border: '2px solid #e5e7eb', fontSize: 15, resize: 'vertical', transition: 'border 0.2s' }}
              onFocus={e => e.target.style.borderColor = '#6366f1'}
              onBlur={e => e.target.style.borderColor = '#e5e7eb'}
            />
          )}

          {/* Navigation */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 32, gap: 12 }}>
            {step > 0 ? (
              <button className="btn-secondary" onClick={back}>← Back</button>
            ) : <div />}
            <button
              className="btn-primary"
              onClick={next}
              disabled={!isValid()}
              style={{ minWidth: 140 }}
            >
              {step === STEPS.length - 1 ? '🚀 Get My Results' : 'Next →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
