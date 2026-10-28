import { useRef, useState } from 'react';
import { api, KINDS } from '../api/client.js';

function UploadRow({ kind, caseId, maxMb, locked, onBusy, onDone, onError }) {
  const meta = KINDS[kind];
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [phase, setPhase] = useState('idle'); // idle | uploading | analyzing
  const [progress, setProgress] = useState(0);
  const [note, setNote] = useState(null); // {type, text}
  const busy = phase !== 'idle';

  function validate(f) {
    if (f.type && !f.type.startsWith(meta.mime)) {
      return `This is not a ${kind} file (detected type: ${f.type}).`;
    }
    if (f.size === 0) return 'The selected file is empty.';
    if (f.size > maxMb * 1024 * 1024) {
      return `File is ${(f.size / 1048576).toFixed(1)} MB. The server limit is ${maxMb} MB.`;
    }
    return null;
  }

  function onPick(e) {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setNote(null);
    if (f) {
      const problem = validate(f);
      if (problem) setNote({ type: 'error', text: problem });
    }
  }

  async function go() {
    if (!file) return setNote({ type: 'error', text: `Choose a ${kind} file first.` });
    const problem = validate(file);
    if (problem) return setNote({ type: 'error', text: problem });
    setNote(null);
    setProgress(0);
    setPhase('uploading');
    onBusy(true);
    try {
      const result = await api.upload(kind, file, caseId, {
        onProgress: setProgress,
        onUploaded: () => setPhase('analyzing'),
      });
      setNote({ type: 'success', text: `${meta.label} uploaded and analyzed.` });
      setFile(null);
      if (inputRef.current) inputRef.current.value = '';
      await onDone(kind, result);
    } catch (e) {
      setNote({ type: 'error', text: e.message });
      onError?.(e.message);
    } finally {
      setPhase('idle');
      onBusy(false);
    }
  }

  return (
    <div className="upload-row">
      <label className="upload-label" htmlFor={`file-${kind}`}>{meta.label}</label>
      <div className="upload-controls">
        <input
          ref={inputRef}
          id={`file-${kind}`}
          type="file"
          accept={meta.accept}
          onChange={onPick}
          disabled={busy || locked}
        />
        <button type="button" className="btn primary go" onClick={go} disabled={busy || locked}>
          {phase === 'uploading' ? `${progress}%` : phase === 'analyzing' ? 'Analyzing…' : 'Go'}
        </button>
      </div>
      {busy && (
        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100}
          aria-valuenow={phase === 'analyzing' ? 100 : progress}>
          <div className={phase === 'analyzing' ? 'bar indeterminate' : 'bar'}
            style={{ width: phase === 'analyzing' ? '100%' : `${progress}%` }} />
        </div>
      )}
      {note && <p className={`note ${note.type}`} role="status">{note.text}</p>}
    </div>
  );
}

export default function UploadPanel({ caseId, maxMb, lastEvidenceId, onUploaded, onError }) {
  const [activeCount, setActiveCount] = useState(0);
  const locked = activeCount > 0; // one upload at a time keeps a blank Case ID from splitting into several cases

  return (
    <section className="panel">
      <h2>Add evidence</h2>
      <p className="hint">
        {caseId ? <>Uploads go to case <b>{caseId}</b>.</> : 'No case loaded: the first upload creates a new case.'}
        {' '}Max {maxMb} MB per file.
      </p>
      {Object.keys(KINDS).map((kind) => (
        <UploadRow
          key={kind}
          kind={kind}
          caseId={caseId}
          maxMb={maxMb}
          locked={locked}
          onBusy={(b) => setActiveCount((n) => n + (b ? 1 : -1))}
          onDone={onUploaded}
          onError={onError}
        />
      ))}
      {lastEvidenceId && (
        <p className="hint">Last evidence ID: <code>{lastEvidenceId}</code></p>
      )}
    </section>
  );
}
