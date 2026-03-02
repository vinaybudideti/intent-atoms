import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, PieChart, Pie, Area, AreaChart } from "recharts";
import { queryEngine, fetchStats, fetchAtoms, clearCache, healthCheck } from "./api";

// —— Simulation Engine (fallback when backend is offline) ——
const SAMPLE_QUERIES = [
  "How do I deploy a React app with Docker on AWS?",
  "What's the best way to implement JWT authentication in Node.js?",
  "Explain the difference between SQL and NoSQL databases",
  "How to set up CI/CD pipeline with GitHub Actions?",
  "What are React hooks and how do useState and useEffect work?",
  "How to containerize a Python Flask application with Docker?",
  "Best practices for REST API design and versioning",
  "How to implement rate limiting in Express.js?",
  "Explain microservices architecture vs monolithic",
  "How to deploy a Node.js app to AWS EC2?",
  "What is Docker and how does containerization work?",
  "How to build a REST API with FastAPI in Python?",
  "React state management: Redux vs Context API vs Zustand",
  "How to set up PostgreSQL with TypeORM in Node.js?",
  "Explain OAuth 2.0 authentication flow",
  "How to implement WebSocket real-time communication?",
  "Best practices for MongoDB schema design",
  "How to use GitHub Actions for automated testing?",
  "Deploying containerized applications to AWS ECS",
  "How to implement caching with Redis in Node.js?",
];

const ATOM_LIBRARY = {
  react_build: { label: "react_production_build", tags: ["react", "frontend"], tokens: 280 },
  docker_container: { label: "docker_containerization", tags: ["docker", "devops"], tokens: 310 },
  aws_deploy: { label: "aws_deployment", tags: ["aws", "cloud"], tokens: 350 },
  jwt_auth: { label: "jwt_authentication", tags: ["auth", "security"], tokens: 290 },
  nodejs_backend: { label: "nodejs_backend_setup", tags: ["node", "backend"], tokens: 260 },
  sql_nosql: { label: "sql_vs_nosql_comparison", tags: ["database"], tokens: 320 },
  cicd_pipeline: { label: "cicd_pipeline_setup", tags: ["devops", "automation"], tokens: 340 },
  github_actions: { label: "github_actions_config", tags: ["devops", "github"], tokens: 270 },
  react_hooks: { label: "react_hooks_explanation", tags: ["react", "frontend"], tokens: 300 },
  usestate_useeffect: { label: "usestate_useeffect_usage", tags: ["react", "frontend"], tokens: 250 },
  python_flask: { label: "flask_app_setup", tags: ["python", "backend"], tokens: 280 },
  rest_api_design: { label: "rest_api_best_practices", tags: ["api", "backend"], tokens: 330 },
  rate_limiting: { label: "rate_limiting_implementation", tags: ["security", "backend"], tokens: 240 },
  microservices: { label: "microservices_architecture", tags: ["architecture"], tokens: 360 },
  monolithic: { label: "monolithic_vs_micro", tags: ["architecture"], tokens: 290 },
  express_middleware: { label: "express_middleware_setup", tags: ["node", "backend"], tokens: 230 },
  oauth_flow: { label: "oauth2_flow_explanation", tags: ["auth", "security"], tokens: 310 },
  websocket: { label: "websocket_implementation", tags: ["realtime", "backend"], tokens: 280 },
  mongodb_schema: { label: "mongodb_schema_design", tags: ["database", "mongodb"], tokens: 300 },
  redis_caching: { label: "redis_caching_setup", tags: ["caching", "backend"], tokens: 260 },
  state_management: { label: "react_state_management", tags: ["react", "frontend"], tokens: 340 },
  postgresql: { label: "postgresql_typeorm_setup", tags: ["database", "node"], tokens: 290 },
  ecs_deploy: { label: "aws_ecs_deployment", tags: ["aws", "devops"], tokens: 320 },
  fastapi: { label: "fastapi_setup", tags: ["python", "api"], tokens: 270 },
};

const QUERY_ATOMS = {
  0: ["react_build", "docker_container", "aws_deploy"],
  1: ["jwt_auth", "nodejs_backend"],
  2: ["sql_nosql"],
  3: ["cicd_pipeline", "github_actions"],
  4: ["react_hooks", "usestate_useeffect"],
  5: ["python_flask", "docker_container"],
  6: ["rest_api_design"],
  7: ["rate_limiting", "express_middleware"],
  8: ["microservices", "monolithic"],
  9: ["nodejs_backend", "aws_deploy"],
  10: ["docker_container"],
  11: ["rest_api_design", "fastapi"],
  12: ["state_management", "react_hooks"],
  13: ["postgresql", "nodejs_backend"],
  14: ["oauth_flow"],
  15: ["websocket", "nodejs_backend"],
  16: ["mongodb_schema"],
  17: ["github_actions", "cicd_pipeline"],
  18: ["docker_container", "ecs_deploy"],
  19: ["redis_caching", "nodejs_backend"],
};

const HAIKU_COST_PER_1K = { input: 0.0008, output: 0.004 };
const SONNET_COST_PER_1K = { input: 0.003, output: 0.015 };

function simulateQuery(queryIndex, atomCache) {
  const atoms = QUERY_ATOMS[queryIndex] || ["react_build"];
  let hits = 0, misses = 0, tokensSaved = 0, tokensUsed = 0;
  const atomDetails = [];

  for (const atomKey of atoms) {
    const atom = ATOM_LIBRARY[atomKey];
    if (atomCache.has(atomKey)) {
      hits++;
      tokensSaved += atom.tokens;
      atomDetails.push({ key: atomKey, ...atom, status: "hit" });
    } else {
      misses++;
      tokensUsed += atom.tokens + 100;
      atomCache.add(atomKey);
      atomDetails.push({ key: atomKey, ...atom, status: "miss" });
    }
  }

  const decomposeCost = 50 * HAIKU_COST_PER_1K.input + 80 * HAIKU_COST_PER_1K.output;
  const genCost = misses > 0 ? (misses * 100 * SONNET_COST_PER_1K.input + tokensUsed * SONNET_COST_PER_1K.output) : 0;
  const composeCost = atoms.length > 1 ? (200 * HAIKU_COST_PER_1K.input + 150 * HAIKU_COST_PER_1K.output) : 0;
  const actualCost = decomposeCost + genCost + composeCost;
  const fullCost = atoms.length * 100 * SONNET_COST_PER_1K.input + atoms.reduce((s, k) => s + (ATOM_LIBRARY[k]?.tokens || 300), 0) * SONNET_COST_PER_1K.output;

  return {
    query: SAMPLE_QUERIES[queryIndex],
    totalAtoms: atoms.length,
    hits, misses, tokensSaved, tokensUsed,
    actualCost: actualCost * 1000,
    fullCost: fullCost * 1000,
    savings: Math.max(0, ((1 - actualCost / fullCost) * 100)),
    atomDetails,
    time: 200 + Math.random() * 800 - hits * 120,
  };
}

// Convert live API response to same format as simulation
function liveResultToLogEntry(query, apiResult) {
  const atomDetails = [];
  // Build atom details from what the API gives us
  const totalAtoms = apiResult.total_atoms || 0;
  const cacheHits = apiResult.cache_hits || 0;
  const cacheMisses = apiResult.cache_misses || 0;
  // We don't have per-atom breakdown from the API, create synthetic entries
  for (let i = 0; i < cacheHits; i++) {
    atomDetails.push({ label: `cached_atom_${i + 1}`, tags: [], tokens: 0, status: "hit" });
  }
  for (let i = 0; i < cacheMisses; i++) {
    atomDetails.push({ label: `generated_atom_${i + 1}`, tags: [], tokens: 0, status: "miss" });
  }

  return {
    query,
    totalAtoms,
    hits: cacheHits,
    misses: cacheMisses,
    tokensSaved: apiResult.tokens_saved || 0,
    tokensUsed: apiResult.total_tokens_used || 0,
    actualCost: (apiResult.estimated_cost || 0) * 1000,
    fullCost: (apiResult.estimated_cost_without_cache || 0) * 1000,
    savings: apiResult.cost_savings_pct || 0,
    atomDetails,
    time: apiResult.total_time_ms || 0,
    response: apiResult.response || "",
    timings: {
      decomposition: apiResult.decomposition_time_ms || 0,
      matching: apiResult.matching_time_ms || 0,
      generation: apiResult.generation_time_ms || 0,
      composition: apiResult.composition_time_ms || 0,
    },
  };
}

// —— Components ——

const COLORS = {
  bg: "#0a0a0f",
  surface: "#12121a",
  surfaceHover: "#1a1a26",
  border: "#1e1e2e",
  borderLight: "#2a2a3e",
  accent: "#22d3ee",
  accentDim: "rgba(34,211,238,0.15)",
  green: "#34d399",
  greenDim: "rgba(52,211,153,0.15)",
  red: "#f87171",
  redDim: "rgba(248,113,113,0.15)",
  amber: "#fbbf24",
  amberDim: "rgba(251,191,36,0.15)",
  purple: "#a78bfa",
  purpleDim: "rgba(167,139,250,0.15)",
  text: "#e2e8f0",
  textDim: "#64748b",
  textMuted: "#475569",
};

function MetricCard({ label, value, subvalue, color = COLORS.accent, icon }) {
  return (
    <div style={{
      background: COLORS.surface, border: `1px solid ${COLORS.border}`,
      borderRadius: 12, padding: "20px 22px", flex: 1, minWidth: 180,
      transition: "border-color 0.2s",
    }}
    onMouseEnter={e => e.currentTarget.style.borderColor = color}
    onMouseLeave={e => e.currentTarget.style.borderColor = COLORS.border}
    >
      <div style={{ fontSize: 12, color: COLORS.textDim, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 8 }}>
        {icon && <span style={{ marginRight: 6 }}>{icon}</span>}{label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color, fontFamily: "'JetBrains Mono', 'SF Mono', monospace", lineHeight: 1 }}>
        {value}
      </div>
      {subvalue && <div style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 6 }}>{subvalue}</div>}
    </div>
  );
}

function AtomPill({ atom, index }) {
  const isHit = atom.status === "hit";
  const bg = isHit ? COLORS.greenDim : COLORS.redDim;
  const color = isHit ? COLORS.green : COLORS.red;
  const borderColor = isHit ? "rgba(52,211,153,0.3)" : "rgba(248,113,113,0.3)";

  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      background: bg, border: `1px solid ${borderColor}`,
      borderRadius: 20, padding: "5px 12px", fontSize: 12,
      fontFamily: "'JetBrains Mono', monospace",
      animation: `fadeSlideIn 0.3s ease ${index * 0.1}s both`,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, flexShrink: 0 }} />
      <span style={{ color }}>{atom.label}</span>
      <span style={{ color: COLORS.textMuted, fontSize: 10 }}>{isHit ? "cached" : "generated"}</span>
    </div>
  );
}

function QueryLogItem({ result, index, isLatest }) {
  const [expanded, setExpanded] = useState(isLatest);

  return (
    <div style={{
      background: isLatest ? COLORS.surfaceHover : COLORS.surface,
      border: `1px solid ${isLatest ? COLORS.borderLight : COLORS.border}`,
      borderRadius: 10, overflow: "hidden", marginBottom: 8,
      transition: "all 0.3s ease",
      animation: isLatest ? "fadeSlideIn 0.4s ease" : "none",
    }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          padding: "12px 16px", cursor: "pointer", display: "flex",
          alignItems: "center", justifyContent: "space-between",
          gap: 12,
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13, color: COLORS.text, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {result.query}
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexShrink: 0 }}>
          <span style={{
            fontSize: 11, fontFamily: "monospace", padding: "2px 8px",
            borderRadius: 6, fontWeight: 600,
            background: result.savings > 30 ? COLORS.greenDim : result.savings > 0 ? COLORS.amberDim : COLORS.redDim,
            color: result.savings > 30 ? COLORS.green : result.savings > 0 ? COLORS.amber : COLORS.red,
          }}>
            {result.savings > 0 ? `↓${result.savings.toFixed(0)}%` : "NEW"}
          </span>
          <span style={{
            fontSize: 11, fontFamily: "monospace",
            color: COLORS.textDim,
          }}>
            {result.hits}/{result.totalAtoms} hit
          </span>
          <span style={{ color: COLORS.textMuted, fontSize: 16, transform: expanded ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}>⌄</span>
        </div>
      </div>

      {expanded && (
        <div style={{ padding: "0 16px 14px", borderTop: `1px solid ${COLORS.border}` }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 12 }}>
            {result.atomDetails.map((a, i) => <AtomPill key={i} atom={a} index={i} />)}
          </div>
          <div style={{ display: "flex", gap: 16, marginTop: 12, fontSize: 11, color: COLORS.textDim, fontFamily: "monospace", flexWrap: "wrap" }}>
            <span>Cost: ${result.actualCost.toFixed(4)}</span>
            <span style={{ color: COLORS.textMuted }}>vs ${result.fullCost.toFixed(4)} full</span>
            <span>Time: {result.time.toFixed(0)}ms</span>
            <span>Tokens saved: {result.tokensSaved}</span>
          </div>
          {result.response && (
            <div style={{
              marginTop: 12, padding: "10px 14px", borderRadius: 8,
              background: COLORS.bg, border: `1px solid ${COLORS.border}`,
              fontSize: 12, color: COLORS.textDim, lineHeight: 1.6,
              maxHeight: 120, overflow: "auto",
            }}>
              {result.response.slice(0, 500)}{result.response.length > 500 ? "..." : ""}
            </div>
          )}
          {result.timings && (
            <div style={{ display: "flex", gap: 12, marginTop: 8, fontSize: 10, color: COLORS.textMuted, fontFamily: "monospace" }}>
              <span>Decompose: {result.timings.decomposition.toFixed(0)}ms</span>
              <span>Match: {result.timings.matching.toFixed(0)}ms</span>
              <span>Generate: {result.timings.generation.toFixed(0)}ms</span>
              <span>Compose: {result.timings.composition.toFixed(0)}ms</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PipelineDiagram({ currentStep }) {
  const steps = [
    { id: "decompose", label: "Decompose", icon: "◇", desc: "Break into atoms" },
    { id: "match", label: "Match", icon: "⊕", desc: "Search cache" },
    { id: "generate", label: "Generate", icon: "⚡", desc: "Fill misses" },
    { id: "compose", label: "Compose", icon: "◈", desc: "Stitch response" },
  ];

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "14px 0" }}>
      {steps.map((step, i) => {
        const isActive = step.id === currentStep;
        const isPast = steps.findIndex(s => s.id === currentStep) > i;
        const color = isActive ? COLORS.accent : isPast ? COLORS.green : COLORS.textMuted;
        return (
          <div key={step.id} style={{ display: "flex", alignItems: "center", flex: 1 }}>
            <div style={{
              textAlign: "center", flex: 1,
              opacity: isActive ? 1 : isPast ? 0.7 : 0.4,
              transition: "all 0.3s ease",
            }}>
              <div style={{ fontSize: 20, marginBottom: 4, color, filter: isActive ? `drop-shadow(0 0 8px ${color})` : "none" }}>{step.icon}</div>
              <div style={{ fontSize: 11, fontWeight: 600, color, letterSpacing: "0.03em" }}>{step.label}</div>
              <div style={{ fontSize: 9, color: COLORS.textMuted, marginTop: 2 }}>{step.desc}</div>
            </div>
            {i < steps.length - 1 && (
              <div style={{
                width: 30, height: 1, flexShrink: 0,
                background: isPast ? COLORS.green : COLORS.border,
                transition: "background 0.3s ease",
              }} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// —— Main App ——
export default function IntentAtomsDashboard() {
  const [queryLog, setQueryLog] = useState([]);
  const [atomCache, setAtomCache] = useState(new Set());
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(null);
  const [queryIndex, setQueryIndex] = useState(0);
  const [customQuery, setCustomQuery] = useState("");
  const [totalSaved, setTotalSaved] = useState(0);
  const [totalCost, setTotalCost] = useState(0);
  const [totalFullCost, setTotalFullCost] = useState(0);
  const [view, setView] = useState("dashboard");
  const [mode, setMode] = useState("simulation"); // "simulation" | "live"
  const [backendOnline, setBackendOnline] = useState(false);
  const [liveAtoms, setLiveAtoms] = useState([]);
  const [liveStats, setLiveStats] = useState(null);
  const [error, setError] = useState(null);
  const atomCacheRef = useRef(atomCache);
  const intervalRef = useRef(null);

  useEffect(() => { atomCacheRef.current = atomCache; }, [atomCache]);

  // Check backend health on mount and when switching to live mode
  useEffect(() => {
    healthCheck().then(ok => setBackendOnline(ok));
    const interval = setInterval(() => healthCheck().then(ok => setBackendOnline(ok)), 10000);
    return () => clearInterval(interval);
  }, []);

  // Fetch live atoms + stats when in live mode
  useEffect(() => {
    if (mode !== "live" || !backendOnline) return;
    const load = async () => {
      try {
        const [atoms, stats] = await Promise.all([fetchAtoms(0, 100), fetchStats()]);
        setLiveAtoms(atoms.atoms || []);
        setLiveStats(stats);
      } catch (e) {
        console.error("Failed to load live data:", e);
      }
    };
    load();
  }, [mode, backendOnline, queryLog.length]);

  // Run a simulated query
  const runSimulatedQuery = useCallback((idx) => {
    setIsRunning(true);
    setError(null);
    const steps = ["decompose", "match", "generate", "compose"];
    let step = 0;

    const advanceStep = () => {
      if (step < steps.length) {
        setCurrentStep(steps[step]);
        step++;
        setTimeout(advanceStep, 300 + Math.random() * 200);
      } else {
        const newCache = new Set(atomCacheRef.current);
        const result = simulateQuery(idx % SAMPLE_QUERIES.length, newCache);
        setAtomCache(newCache);
        setQueryLog(prev => [result, ...prev].slice(0, 50));
        setTotalSaved(prev => prev + result.tokensSaved);
        setTotalCost(prev => prev + result.actualCost);
        setTotalFullCost(prev => prev + result.fullCost);
        setCurrentStep(null);
        setIsRunning(false);
        setQueryIndex(prev => prev + 1);
      }
    };
    advanceStep();
  }, []);

  // Run a live query against the backend
  const runLiveQuery = useCallback(async (queryText) => {
    setIsRunning(true);
    setError(null);
    setCurrentStep("decompose");

    try {
      // Start pipeline animation
      const stepTimer = setInterval(() => {
        setCurrentStep(prev => {
          const steps = ["decompose", "match", "generate", "compose"];
          const idx = steps.indexOf(prev);
          if (idx < steps.length - 1) return steps[idx + 1];
          return prev;
        });
      }, 800);

      const apiResult = await queryEngine(queryText);
      clearInterval(stepTimer);
      setCurrentStep("compose");

      const result = liveResultToLogEntry(queryText, apiResult);
      setQueryLog(prev => [result, ...prev].slice(0, 50));
      setTotalSaved(prev => prev + result.tokensSaved);
      setTotalCost(prev => prev + result.actualCost);
      setTotalFullCost(prev => prev + result.fullCost);

      setTimeout(() => {
        setCurrentStep(null);
        setIsRunning(false);
        setQueryIndex(prev => prev + 1);
      }, 300);
    } catch (e) {
      setError(e.message);
      setCurrentStep(null);
      setIsRunning(false);
    }
  }, []);

  const handleRunQuery = () => {
    if (isRunning) return;
    if (mode === "live") {
      const q = customQuery.trim() || SAMPLE_QUERIES[queryIndex % SAMPLE_QUERIES.length];
      runLiveQuery(q);
    } else {
      runSimulatedQuery(queryIndex % SAMPLE_QUERIES.length);
    }
  };

  const handleCustomSubmit = (e) => {
    e.preventDefault();
    if (isRunning || !customQuery.trim()) return;
    if (mode === "live") {
      runLiveQuery(customQuery.trim());
    } else {
      // In simulation, just run next sample query
      runSimulatedQuery(queryIndex % SAMPLE_QUERIES.length);
    }
    setCustomQuery("");
  };

  const runAuto = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      return;
    }
    let idx = queryIndex;
    intervalRef.current = setInterval(() => {
      if (!isRunning) {
        if (mode === "simulation") {
          runSimulatedQuery(idx % SAMPLE_QUERIES.length);
          idx++;
        } else {
          runLiveQuery(SAMPLE_QUERIES[idx % SAMPLE_QUERIES.length]);
          idx++;
        }
      }
    }, mode === "live" ? 6000 : 2500);
  };

  const handleReset = async () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setQueryLog([]);
    setAtomCache(new Set());
    setTotalSaved(0);
    setTotalCost(0);
    setTotalFullCost(0);
    setQueryIndex(0);
    setError(null);
    if (mode === "live" && backendOnline) {
      try { await clearCache(); } catch (e) { console.error(e); }
    }
  };

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  const cacheHitRate = queryLog.length > 0
    ? (queryLog.reduce((s, r) => s + r.hits, 0) / Math.max(queryLog.reduce((s, r) => s + r.totalAtoms, 0), 1) * 100)
    : 0;

  const overallSavings = totalFullCost > 0 ? ((1 - totalCost / totalFullCost) * 100) : 0;

  const costOverTime = queryLog.slice().reverse().map((r, i) => ({
    query: i + 1,
    actual: +r.actualCost.toFixed(4),
    full: +r.fullCost.toFixed(4),
    savings: +r.savings.toFixed(1),
  }));

  const hitRateOverTime = queryLog.slice().reverse().map((r, i) => {
    const slice = queryLog.slice(0, queryLog.length - i);
    const totalA = slice.reduce((s, x) => s + x.totalAtoms, 0);
    const totalH = slice.reduce((s, x) => s + x.hits, 0);
    return { query: i + 1, hitRate: totalA > 0 ? +(totalH / totalA * 100).toFixed(1) : 0 };
  });

  // Domain data — from simulation or live
  let domainData = [];
  if (mode === "simulation") {
    const domainCounts = {};
    for (const key of atomCache) {
      const atom = ATOM_LIBRARY[key];
      if (atom) atom.tags.forEach(t => { domainCounts[t] = (domainCounts[t] || 0) + 1; });
    }
    domainData = Object.entries(domainCounts).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
  } else if (liveStats?.domain_distribution) {
    domainData = Object.entries(liveStats.domain_distribution).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
  }

  const PIE_COLORS = [COLORS.accent, COLORS.green, COLORS.purple, COLORS.amber, COLORS.red, "#818cf8", "#fb923c", "#38bdf8"];

  const cachedAtomCount = mode === "live" ? (liveStats?.total_atoms_stored || 0) : atomCache.size;

  return (
    <div style={{
      minHeight: "100vh", background: COLORS.bg, color: COLORS.text,
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${COLORS.bg}; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 3px; }
      `}</style>

      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${COLORS.border}`, padding: "16px 28px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "linear-gradient(180deg, rgba(34,211,238,0.03) 0%, transparent 100%)",
        flexWrap: "wrap", gap: 12,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.purple})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, fontWeight: 700, color: COLORS.bg,
          }}>⚛</div>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.02em" }}>
              Intent Atoms
            </div>
            <div style={{ fontSize: 11, color: COLORS.textDim }}>
              Sub-query intelligent caching for LLMs
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {/* Mode toggle */}
          <div style={{
            display: "flex", gap: 2, background: COLORS.surface, borderRadius: 6,
            border: `1px solid ${COLORS.border}`, padding: 2, marginRight: 8,
          }}>
            {["simulation", "live"].map(m => (
              <button key={m} onClick={() => { setMode(m); if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }} style={{
                padding: "5px 12px", borderRadius: 4, border: "none", cursor: "pointer",
                fontSize: 11, fontWeight: 600, textTransform: "capitalize",
                background: mode === m ? (m === "live" ? COLORS.greenDim : COLORS.accentDim) : "transparent",
                color: mode === m ? (m === "live" ? COLORS.green : COLORS.accent) : COLORS.textMuted,
                transition: "all 0.2s",
              }}>
                {m === "live" && <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: backendOnline ? COLORS.green : COLORS.red, marginRight: 5, verticalAlign: "middle" }} />}
                {m}
              </button>
            ))}
          </div>

          {/* View tabs */}
          {["dashboard", "atoms", "analytics"].map(v => (
            <button key={v} onClick={() => setView(v)} style={{
              padding: "6px 14px", borderRadius: 6, border: "none", cursor: "pointer",
              fontSize: 12, fontWeight: 500, letterSpacing: "0.02em", textTransform: "capitalize",
              background: view === v ? COLORS.accentDim : "transparent",
              color: view === v ? COLORS.accent : COLORS.textDim,
              transition: "all 0.2s",
            }}>{v}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: "20px 28px", maxWidth: 1200, margin: "0 auto" }}>

        {/* Backend status banner for live mode */}
        {mode === "live" && !backendOnline && (
          <div style={{
            background: COLORS.redDim, border: `1px solid rgba(248,113,113,0.3)`,
            borderRadius: 10, padding: "12px 18px", marginBottom: 16,
            display: "flex", alignItems: "center", gap: 10,
          }}>
            <span style={{ color: COLORS.red, fontSize: 16 }}>!</span>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.red }}>Backend Offline</div>
              <div style={{ fontSize: 12, color: COLORS.textDim, marginTop: 2 }}>
                Start the server: <code style={{ background: COLORS.surface, padding: "2px 8px", borderRadius: 4, fontSize: 11 }}>source venv/bin/activate && uvicorn api.server:app --port 8000</code>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div style={{
            background: COLORS.redDim, border: `1px solid rgba(248,113,113,0.3)`,
            borderRadius: 10, padding: "10px 16px", marginBottom: 16,
            fontSize: 13, color: COLORS.red,
          }}>
            Error: {error}
          </div>
        )}

        {/* Metrics */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
          <MetricCard label="Queries" value={queryLog.length} icon="▸" color={COLORS.accent} subvalue="processed through engine" />
          <MetricCard label="Cache Hit Rate" value={`${cacheHitRate.toFixed(1)}%`} icon="◎" color={COLORS.green} subvalue={`${cachedAtomCount} atoms cached`} />
          <MetricCard label="Cost Savings" value={`${overallSavings.toFixed(1)}%`} icon="↓" color={overallSavings > 30 ? COLORS.green : COLORS.amber} subvalue={`$${((totalFullCost - totalCost) / 1000).toFixed(6)} saved`} />
          <MetricCard label="Tokens Saved" value={totalSaved.toLocaleString()} icon="⚡" color={COLORS.purple} subvalue="not regenerated" />
        </div>

        {/* Pipeline + Controls */}
        <div style={{
          background: COLORS.surface, border: `1px solid ${COLORS.border}`,
          borderRadius: 12, padding: "16px 22px", marginBottom: 20,
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDim }}>
              Pipeline
              <span style={{
                marginLeft: 8, fontSize: 10, padding: "2px 8px", borderRadius: 10,
                background: mode === "live" ? COLORS.greenDim : COLORS.accentDim,
                color: mode === "live" ? COLORS.green : COLORS.accent,
              }}>{mode === "live" ? "LIVE" : "SIM"}</span>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={handleRunQuery}
                disabled={isRunning || (mode === "live" && !backendOnline)}
                style={{
                  padding: "7px 16px", borderRadius: 6, border: "none", cursor: isRunning || (mode === "live" && !backendOnline) ? "not-allowed" : "pointer",
                  background: COLORS.accent, color: COLORS.bg, fontSize: 12, fontWeight: 600,
                  opacity: isRunning || (mode === "live" && !backendOnline) ? 0.5 : 1, transition: "opacity 0.2s",
                }}
              >
                {isRunning ? "Processing..." : "▶ Run Query"}
              </button>
              <button
                onClick={runAuto}
                disabled={mode === "live" && !backendOnline}
                style={{
                  padding: "7px 16px", borderRadius: 6, border: `1px solid ${COLORS.border}`,
                  cursor: (mode === "live" && !backendOnline) ? "not-allowed" : "pointer",
                  background: intervalRef.current ? COLORS.redDim : "transparent",
                  color: intervalRef.current ? COLORS.red : COLORS.textDim, fontSize: 12, fontWeight: 500,
                  opacity: (mode === "live" && !backendOnline) ? 0.5 : 1,
                }}
              >
                {intervalRef.current ? "■ Stop" : "⟳ Auto-run"}
              </button>
              <button
                onClick={handleReset}
                style={{
                  padding: "7px 12px", borderRadius: 6, border: `1px solid ${COLORS.border}`,
                  cursor: "pointer", background: "transparent", color: COLORS.textDim, fontSize: 12,
                }}
              >
                Reset
              </button>
            </div>
          </div>

          {/* Custom query input (live mode) */}
          {mode === "live" && (
            <form onSubmit={handleCustomSubmit} style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <input
                type="text"
                value={customQuery}
                onChange={e => setCustomQuery(e.target.value)}
                placeholder="Type a custom query..."
                style={{
                  flex: 1, padding: "8px 14px", borderRadius: 6,
                  border: `1px solid ${COLORS.border}`, background: COLORS.bg,
                  color: COLORS.text, fontSize: 13, outline: "none",
                  fontFamily: "inherit",
                }}
              />
              <button
                type="submit"
                disabled={isRunning || !customQuery.trim() || !backendOnline}
                style={{
                  padding: "8px 16px", borderRadius: 6, border: "none",
                  background: COLORS.accentDim, color: COLORS.accent,
                  fontSize: 12, fontWeight: 600, cursor: "pointer",
                  opacity: isRunning || !customQuery.trim() || !backendOnline ? 0.5 : 1,
                }}
              >
                Send
              </button>
            </form>
          )}

          <div style={{ fontSize: 12, color: COLORS.textMuted, marginBottom: 4 }}>
            Next: <span style={{ color: COLORS.text }}>{SAMPLE_QUERIES[queryIndex % SAMPLE_QUERIES.length]}</span>
          </div>

          <PipelineDiagram currentStep={currentStep} />
        </div>

        {view === "dashboard" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Cost Comparison Chart */}
            <div style={{
              background: COLORS.surface, border: `1px solid ${COLORS.border}`,
              borderRadius: 12, padding: "18px 20px",
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDim, marginBottom: 14 }}>Cost per Query (mUSD)</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={costOverTime}>
                  <defs>
                    <linearGradient id="fullGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={COLORS.red} stopOpacity={0.2} />
                      <stop offset="100%" stopColor={COLORS.red} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="actualGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={COLORS.green} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={COLORS.green} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="query" stroke={COLORS.textMuted} tick={{ fontSize: 10 }} />
                  <YAxis stroke={COLORS.textMuted} tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12, color: COLORS.text }}
                    formatter={(v, name) => [`$${v.toFixed(4)}`, name === "full" ? "Without Atoms" : "With Atoms"]}
                  />
                  <Area type="monotone" dataKey="full" stroke={COLORS.red} fill="url(#fullGrad)" strokeWidth={1.5} dot={false} />
                  <Area type="monotone" dataKey="actual" stroke={COLORS.green} fill="url(#actualGrad)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Hit Rate Over Time */}
            <div style={{
              background: COLORS.surface, border: `1px solid ${COLORS.border}`,
              borderRadius: 12, padding: "18px 20px",
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDim, marginBottom: 14 }}>Cumulative Cache Hit Rate (%)</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={hitRateOverTime}>
                  <defs>
                    <linearGradient id="hitGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={COLORS.accent} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={COLORS.accent} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="query" stroke={COLORS.textMuted} tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} stroke={COLORS.textMuted} tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12, color: COLORS.text }}
                    formatter={(v) => [`${v}%`, "Hit Rate"]}
                  />
                  <Area type="monotone" dataKey="hitRate" stroke={COLORS.accent} fill="url(#hitGrad)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Query Log */}
            <div style={{
              background: COLORS.surface, border: `1px solid ${COLORS.border}`,
              borderRadius: 12, padding: "18px 20px", gridColumn: "1 / -1",
              maxHeight: 400, overflow: "auto",
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDim, marginBottom: 12 }}>Query Log</div>
              {queryLog.length === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: COLORS.textMuted, fontSize: 13 }}>
                  {mode === "live"
                    ? 'Click "Run Query" or type a custom query to send it to the backend engine.'
                    : 'Click "Run Query" or "Auto-run" to start processing queries and see the cache build up over time.'
                  }
                </div>
              ) : (
                queryLog.map((r, i) => <QueryLogItem key={queryLog.length - i} result={r} index={i} isLatest={i === 0} />)
              )}
            </div>
          </div>
        )}

        {view === "atoms" && (
          <div style={{
            background: COLORS.surface, border: `1px solid ${COLORS.border}`,
            borderRadius: 12, padding: "18px 20px",
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDim, marginBottom: 14 }}>
              Cached Atoms ({cachedAtomCount})
            </div>
            {mode === "live" ? (
              liveAtoms.length === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: COLORS.textMuted, fontSize: 13 }}>
                  No atoms cached yet. Run some queries in Live mode to populate the cache.
                </div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 10 }}>
                  {liveAtoms.map(atom => (
                    <div key={atom.id} style={{
                      background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                      borderRadius: 10, padding: "14px 16px",
                    }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.accent, fontFamily: "monospace", marginBottom: 6 }}>
                        {atom.intent_label || atom.id}
                      </div>
                      <div style={{ fontSize: 12, color: COLORS.text, marginBottom: 8, lineHeight: 1.4 }}>
                        {atom.intent_text}
                      </div>
                      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
                        {(atom.domain_tags || []).map(t => (
                          <span key={t} style={{
                            fontSize: 10, padding: "2px 8px", borderRadius: 10,
                            background: COLORS.purpleDim, color: COLORS.purple,
                          }}>{t}</span>
                        ))}
                      </div>
                      <div style={{ display: "flex", gap: 12, fontSize: 11, color: COLORS.textDim, fontFamily: "monospace" }}>
                        <span>{atom.token_count} tokens</span>
                        <span>Used {atom.usage_count}x</span>
                      </div>
                    </div>
                  ))}
                </div>
              )
            ) : (
              atomCache.size === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: COLORS.textMuted, fontSize: 13 }}>
                  No atoms cached yet. Run some queries to populate the cache.
                </div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 10 }}>
                  {[...atomCache].map(key => {
                    const atom = ATOM_LIBRARY[key];
                    if (!atom) return null;
                    return (
                      <div key={key} style={{
                        background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                        borderRadius: 10, padding: "14px 16px",
                      }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.accent, fontFamily: "monospace", marginBottom: 6 }}>
                          {atom.label}
                        </div>
                        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
                          {atom.tags.map(t => (
                            <span key={t} style={{
                              fontSize: 10, padding: "2px 8px", borderRadius: 10,
                              background: COLORS.purpleDim, color: COLORS.purple,
                            }}>{t}</span>
                          ))}
                        </div>
                        <div style={{ fontSize: 11, color: COLORS.textDim, fontFamily: "monospace" }}>
                          {atom.tokens} tokens cached
                        </div>
                      </div>
                    );
                  })}
                </div>
              )
            )}
          </div>
        )}

        {view === "analytics" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Domain Distribution */}
            <div style={{
              background: COLORS.surface, border: `1px solid ${COLORS.border}`,
              borderRadius: 12, padding: "18px 20px",
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDim, marginBottom: 14 }}>Domain Distribution</div>
              {domainData.length === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: COLORS.textMuted, fontSize: 13 }}>Run queries to see domain analytics.</div>
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={domainData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false} fontSize={10}>
                      {domainData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                    </Pie>
                    <Tooltip contentStyle={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12, color: COLORS.text }} />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* Savings per query bar chart */}
            <div style={{
              background: COLORS.surface, border: `1px solid ${COLORS.border}`,
              borderRadius: 12, padding: "18px 20px",
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDim, marginBottom: 14 }}>Savings % per Query</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={costOverTime}>
                  <XAxis dataKey="query" stroke={COLORS.textMuted} tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} stroke={COLORS.textMuted} tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12, color: COLORS.text }} formatter={(v) => [`${v}%`, "Savings"]} />
                  <Bar dataKey="savings" radius={[4, 4, 0, 0]}>
                    {costOverTime.map((entry, i) => (
                      <Cell key={i} fill={entry.savings > 40 ? COLORS.green : entry.savings > 10 ? COLORS.amber : COLORS.red} fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Live Stats Panel (only in live mode) */}
            {mode === "live" && liveStats && (
              <div style={{
                background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                borderRadius: 12, padding: "18px 20px", gridColumn: "1 / -1",
              }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDim, marginBottom: 14 }}>Backend Stats (Live)</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.accent, fontFamily: "monospace" }}>{liveStats.total_atoms_stored}</div>
                    <div style={{ fontSize: 11, color: COLORS.textDim }}>Atoms Stored</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.green, fontFamily: "monospace" }}>{liveStats.total_queries_processed}</div>
                    <div style={{ fontSize: 11, color: COLORS.textDim }}>Total Queries</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.purple, fontFamily: "monospace" }}>{(liveStats.overall_hit_rate * 100).toFixed(1)}%</div>
                    <div style={{ fontSize: 11, color: COLORS.textDim }}>Overall Hit Rate</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.amber, fontFamily: "monospace" }}>{liveStats.total_tokens_saved?.toLocaleString()}</div>
                    <div style={{ fontSize: 11, color: COLORS.textDim }}>Total Tokens Saved</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.green, fontFamily: "monospace" }}>${liveStats.total_cost_saved?.toFixed(6)}</div>
                    <div style={{ fontSize: 11, color: COLORS.textDim }}>Total Cost Saved</div>
                  </div>
                </div>
                {liveStats.most_reused_atoms?.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <div style={{ fontSize: 12, color: COLORS.textDim, marginBottom: 8 }}>Most Reused Atoms:</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {liveStats.most_reused_atoms.slice(0, 8).map((a, i) => (
                        <span key={i} style={{
                          fontSize: 11, padding: "4px 10px", borderRadius: 15,
                          background: COLORS.accentDim, color: COLORS.accent,
                          fontFamily: "monospace",
                        }}>
                          {a.label} ({a.uses}x)
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Key Insight */}
            <div style={{
              gridColumn: "1 / -1",
              background: `linear-gradient(135deg, ${COLORS.accentDim}, ${COLORS.purpleDim})`,
              border: `1px solid ${COLORS.borderLight}`,
              borderRadius: 12, padding: "22px 26px",
            }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.accent, marginBottom: 8 }}>Key Insight</div>
              <div style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.7 }}>
                Notice how the savings increase over time as the atom cache builds up. Early queries have low cache hit rates
                because everything is new. But as more atomic intents get cached, subsequent queries that share overlapping
                concepts (e.g., "Docker deployment" appears in multiple different questions) get progressively cheaper.
                <br /><br />
                In production with thousands of users asking overlapping questions, the cache hit rate would be significantly higher,
                potentially reaching <span style={{ color: COLORS.green, fontWeight: 600 }}>60-70% savings</span> on API costs.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
