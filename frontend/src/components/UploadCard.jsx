import React, { useRef, useState } from 'react';
import { Spinner, ErrorBox } from './common.jsx';

const ACCEPT = { image: 'image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif', video: 'video/*,.mp4,.avi,.mov,.mkv,.webm', audio: 'audio/*,.wav,.mp3,.m4a,.ogg,.flac' };

// The Go button is a normal flex item that wraps below the file picker on narrow screens: never clipped or overlapped.
export default function UploadCard({ kind, title, maxMb, busy, error, onGo }) {
  const [file, setFile] = useState(null);
  const [localErr, setLocalErr] = useState('');
  const input = useRef(null);
  const pick = (f) => {
    setLocalErr('');
    if (f && maxMb && f.size > maxMb * 1024 * 1024) { setLocalErr(`File is ${(f.size / 1048576).toFixed(0)} MB; the limit is ${maxMb} MB.`); setFile(null); return; }
    setFile(f || null);
  };
  const go = async () => {
    if (!file) return;
    const ok = await onGo(file);
    if (ok) { setFile(null); if (input.current) input.current.value = ''; }
  };
  return (
    <div className="upload">
      <h3>{title}</h3>
      <div className="upload-row">
        <label className="file-pick">
          <input ref={input} type="file" accept={ACCEPT[kind]} disabled={busy} onChange={(e) => pick(e.target.files[0])} />
          <span className="file-name">{file ? file.name : `Choose ${kind} file`}</span>
        </label>
        <button className="btn primary go" onClick={go} disabled={!file || busy} aria-label={`Analyse ${kind}`}>
          {busy ? <Spinner label="Analysing…" /> : 'Go'}
        </button>
      </div>
      {busy && kind !== 'image' && <p className="muted small">{kind === 'video' ? 'Video analysis can take several minutes.' : 'Transcription can take a while on CPU.'} Keep this tab open.</p>}
      <ErrorBox error={localErr || error} />
    </div>
  );
}
