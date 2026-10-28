import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api.js';
import CaseBar from './components/CaseBar.jsx';
import UploadCard from './components/UploadCard.jsx';
import ImageResult from './components/ImageResult.jsx';
import VideoResult from './components/VideoResult.jsx';
import AudioResult from './components/AudioResult.jsx';
import Assessment from './components/Assessment.jsx';
import ChatPanel from './components/ChatPanel.jsx';
import IntegrityPanel from './components/IntegrityPanel.jsx';
import { Card, Empty, ErrorBox, Spinner } from './components/common.jsx';

const EMPTY = { images: [], videos: [], audio: [], timeline: [], analysis: null };

export default function App() {
  const [cases, setCases] = useState([]);
  const [caseId, setCaseId] = useState('');
  const [data, setData] = useState(EMPTY);
  const [cfg, setCfg] = useState({});
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState({});          // image | video | audio | classify | reanalyze | report
  const [errors, setErrors] = useState({});
  const seq = useRef(0);

  const setB = (k, v) => setBusy((b) => ({ ...b, [k]: v }));
  const setE = (k, v) => setErrors((e) => ({ ...e, [k]: v }));

  const refreshCases = useCallback(async () => {
    try { setCases(await api.listCases()); setE('cases', ''); } catch (e) { setE('cases', e.message); }
  }, []);

  const loadCase = useCallback(async (cid) => {
    const mine = ++seq.current;                    // ignore stale responses when the user switches case quickly
    if (!cid) { setData(EMPTY); return; }
    setLoading(true); setE('load', '');
    try {
      const t = await api.timeline(cid);
      if (mine === seq.current) setData({ images: t.images || [], videos: t.videos || [], audio: t.audio || [], timeline: t.timeline || [], analysis: t.analysis || null });
    } catch (e) { if (mine === seq.current) { setE('load', e.message); setData(EMPTY); } }
    finally { if (mine === seq.current) setLoading(false); }
  }, []);

  useEffect(() => { refreshCases(); api.config().then(setCfg).catch(() => {}); }, [refreshCases]);
  useEffect(() => { loadCase(caseId); }, [caseId, loadCase]);

  const createCase = async (name) => {
    setE('create', '');
    try { const r = await api.createCase(name); await refreshCases(); setCaseId(r.case_id); return true; }
    catch (e) { setE('create', e.message); return false; }
  };

  const analyze = (kind) => async (file) => {
    setB(kind, true); setE(kind, '');
    try {
      const r = await api.analyze(kind, file, caseId || undefined);
      if (r.case_id !== caseId) { await refreshCases(); setCaseId(r.case_id); } else { await loadCase(caseId); }
      return true;
    } catch (e) { setE(kind, e.message); if (caseId) loadCase(caseId); return false; }   // evidence may be stored even when analysis failed
    finally { setB(kind, false); }
  };

  const reanalyze = async (eid) => {
    setB('reanalyze', true); setE('reanalyze', '');
    try { await api.reanalyze(caseId, eid); await loadCase(caseId); }
    catch (e) { setE('reanalyze', e.message); }
    finally { setB('reanalyze', false); }
  };

  const classify = async () => {
    setB('classify', true); setE('classify', '');
    try { const a = await api.classify(caseId); setData((d) => ({ ...d, analysis: a })); }
    catch (e) { setE('classify', e.message); }
    finally { setB('classify', false); }
  };

  const report = async () => {
    setB('report', true); setE('report', '');
    try { await api.downloadReport(caseId); } catch (e) { setE('report', e.message); }
    finally { setB('report', false); }
  };

  const hasEvidence = data.images.length + data.videos.length + data.audio.length > 0;
  const busyAny = Object.values(busy).some(Boolean);
  const eventsFor = (eid) => data.timeline.filter((e) => e.source === 'video' && e.evidence_id === eid);
  const maxMb = cfg.max_upload_mb;

  return (
    <div className="app">
      <header className="top">
        <h1>CrimeVision AI</h1>
        <p className="muted">Preliminary AI-assisted evidence analysis. Outputs are model predictions that need human review; they are not findings of fact or guilt.</p>
      </header>

      <CaseBar cases={cases} caseId={caseId} onSelect={setCaseId} onCreate={createCase} onRefresh={() => { refreshCases(); loadCase(caseId); }}
               busy={busyAny} onReport={report} reporting={busy.report} />
      <ErrorBox error={errors.cases || errors.create || errors.report || errors.load} />

      <Card title="Add evidence" className="uploads-card">
        {!caseId && <p className="muted small">No case selected: the first upload will create a new case automatically.</p>}
        <div className="uploads">
          <UploadCard kind="image" title="Image" maxMb={maxMb} busy={busy.image} error={errors.image} onGo={analyze('image')} />
          <UploadCard kind="video" title="Video" maxMb={maxMb} busy={busy.video} error={errors.video} onGo={analyze('video')} />
          <UploadCard kind="audio" title="Audio" maxMb={maxMb} busy={busy.audio} error={errors.audio} onGo={analyze('audio')} />
        </div>
      </Card>

      {loading && <p className="center"><Spinner label="Loading case…" /></p>}
      <ErrorBox error={errors.reanalyze} />

      {caseId && !loading && !hasEvidence && !errors.load && <Empty>This case has no analysed evidence yet. Upload an image, video or audio file above.</Empty>}

      {data.images.length > 0 && <h2 className="section">Images</h2>}
      {[...data.images].reverse().map((im) => <ImageResult key={im.evidence_id} im={im} onReanalyze={reanalyze} busy={busy.reanalyze} />)}

      {data.videos.length > 0 && <h2 className="section">Videos</h2>}
      {[...data.videos].reverse().map((v) => <VideoResult key={v.evidence_id} v={v} events={eventsFor(v.evidence_id)} caseId={caseId} onReanalyze={reanalyze} busy={busy.reanalyze} />)}

      {data.audio.length > 0 && <h2 className="section">Audio</h2>}
      {[...data.audio].reverse().map((au) => <AudioResult key={au.evidence_id} au={au} onReanalyze={reanalyze} busy={busy.reanalyze} />)}

      {caseId && (
        <div className="bottom">
          <Card title="Case-level incident assessment">
            <Assessment a={data.analysis} busy={busy.classify} error={errors.classify} onClassify={classify} canClassify={hasEvidence} />
          </Card>
          <div className="stack">
            <ChatPanel caseId={caseId} disabled={!hasEvidence} />
            <IntegrityPanel caseId={caseId} />
          </div>
        </div>
      )}
    </div>
  );
}
