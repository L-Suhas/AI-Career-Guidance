import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function Admin() {
  const navigate = useNavigate();
  const [weights, setWeights] = useState({});
  const [feedback, setFeedback] = useState({});
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => { fetchAll(); }, []);

  async function fetchAll() {
    setLoading(true);
    try {
      const [rlRes, fbRes] = await Promise.all([
        axios.get("http://localhost:8000/admin/rl-weights"),
        axios.get("http://localhost:8000/admin/feedback"),
      ]);
      setWeights(rlRes.data.weights || {});
      setFeedback(fbRes.data.feedback_stats || {});
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  const totalAccepts = Object.values(feedback).reduce((s, v) => s + (v.accept || 0), 0);
  const totalRejects = Object.values(feedback).reduce((s, v) => s + (v.reject || 0), 0);
  const weightEntries = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  const tabs = ["overview", "rl weights", "feedback"];

  if (loading) return (
    <div className="page">
      <div className="loading-spinner">
        <div className="spinner" />
        <p style={{ color: "#4f46e5", fontWeight: 600 }}>Loading...</p>
      </div>
    </div>
  );

  return (
    <div className="page" style={{ background: "#f0f4f8" }}>
      <div className="container" style={{ maxWidth: 900 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
          <div>
            <h1 style={{ fontSize: 26, fontWeight: 800, color: "#1f2937" }}>Admin Dashboard</h1>
            <p style={{ color: "#6b7280", fontSize: 14 }}>AI Based Career Guidance System</p>
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button onClick={fetchAll} style={{ padding: "9px 18px", borderRadius: 8, background: "#e0e7ff", color: "#3730a3", fontWeight: 600, fontSize: 14, border: "none", cursor: "pointer" }}>Refresh</button>
            <button onClick={() => navigate("/")} className="btn-secondary" style={{ padding: "9px 18px", fontSize: 14 }}>Back to app</button>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 28 }}>
          {[
            { label: "Total accepts", value: totalAccepts, color: "#10b981", bg: "#ecfdf5" },
            { label: "Total rejects", value: totalRejects, color: "#ef4444", bg: "#fef2f2" },
            { label: "Careers tracked", value: weightEntries.length, color: "#4f46e5", bg: "#eef2ff" },
          ].map((s, i) => (
            <div key={i} style={{ background: s.bg, borderRadius: 12, padding: "18px 16px" }}>
              <div style={{ fontSize: 32, fontWeight: 800, color: s.color }}>{s.value}</div>
              <div style={{ fontSize: 13, color: "#6b7280", marginTop: 4 }}>{s.label}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: 4, marginBottom: 20, background: "#e5e7eb", borderRadius: 10, padding: 4 }}>
          {tabs.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              flex: 1, padding: "9px 0", borderRadius: 8, fontSize: 14, fontWeight: 600,
              border: "none", cursor: "pointer", textTransform: "capitalize",
              background: activeTab === tab ? "#fff" : "transparent",
              color: activeTab === tab ? "#4f46e5" : "#6b7280",
            }}>{tab}</button>
          ))}
        </div>

        {activeTab === "overview" && (
          <div className="card">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 20, color: "#1f2937" }}>RL learning status</h2>
            {weightEntries.length === 0 ? (
              <p style={{ color: "#9ca3af", textAlign: "center", padding: "40px 0" }}>No feedback yet. Use the quiz and click This fits me or Not for me buttons.</p>
            ) : (
              weightEntries.map(([career, weight]) => (
                <div key={career} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                    <span style={{ fontWeight: 500, color: "#374151" }}>{career}</span>
                    <span style={{ fontWeight: 700, color: weight >= 0 ? "#10b981" : "#ef4444" }}>
                      {weight >= 0 ? "+" : ""}{(weight * 100).toFixed(1)}
                    </span>
                  </div>
                  <div style={{ height: 8, background: "#e5e7eb", borderRadius: 99, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: Math.min(Math.abs(weight) * 100, 100) + "%", background: weight >= 0 ? "#10b981" : "#ef4444", borderRadius: 99 }} />
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === "rl weights" && (
          <div className="card">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8, color: "#1f2937" }}>Q-table weights</h2>
            <p style={{ fontSize: 14, color: "#6b7280", marginBottom: 20 }}>Positive means users liked this career. Negative means users rejected it.</p>
            {weightEntries.length === 0 ? (
              <p style={{ color: "#9ca3af", textAlign: "center", padding: "40px 0" }}>No RL data yet.</p>
            ) : (
              weightEntries.map(([career, weight]) => (
                <div key={career} style={{ marginBottom: 14, padding: 16, background: "#f9fafb", borderRadius: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontWeight: 600, color: "#1f2937" }}>{career}</span>
                    <span style={{ fontWeight: 800, fontSize: 18, color: weight >= 0 ? "#10b981" : "#ef4444" }}>
                      {weight >= 0 ? "+" : ""}{weight.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ height: 8, background: "#e5e7eb", borderRadius: 99, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: Math.min(Math.abs(weight) * 100, 100) + "%", background: weight >= 0 ? "#10b981" : "#ef4444", borderRadius: 99 }} />
                  </div>
                  <p style={{ fontSize: 12, color: "#9ca3af", marginTop: 6 }}>
                    {weight > 0.05 ? "Boosted" : weight < -0.05 ? "Penalized" : "Neutral"}
                  </p>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === "feedback" && (
          <div className="card">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 20, color: "#1f2937" }}>Feedback per career</h2>
            {Object.keys(feedback).length === 0 ? (
              <p style={{ color: "#9ca3af", textAlign: "center", padding: "40px 0" }}>No feedback recorded yet.</p>
            ) : (
              Object.entries(feedback).map(([career, counts]) => {
                const total = (counts.accept || 0) + (counts.reject || 0);
                const sat = total > 0 ? Math.round((counts.accept || 0) / total * 100) : 0;
                return (
                  <div key={career} style={{ marginBottom: 16, padding: 16, background: "#f9fafb", borderRadius: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                      <span style={{ fontWeight: 600, color: "#1f2937" }}>{career}</span>
                      <div style={{ display: "flex", gap: 8 }}>
                        <span style={{ background: "#d1fae5", color: "#065f46", padding: "2px 10px", borderRadius: 99, fontSize: 12, fontWeight: 600 }}>{counts.accept || 0} accepts</span>
                        <span style={{ background: "#fee2e2", color: "#991b1b", padding: "2px 10px", borderRadius: 99, fontSize: 12, fontWeight: 600 }}>{counts.reject || 0} rejects</span>
                      </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ flex: 1, height: 8, background: "#e5e7eb", borderRadius: 99, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: sat + "%", background: sat >= 60 ? "#10b981" : "#ef4444", borderRadius: 99 }} />
                      </div>
                      <span style={{ fontSize: 13, fontWeight: 700, color: sat >= 60 ? "#10b981" : "#ef4444" }}>{sat}%</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
}
