import { useEffect, useState } from 'react';

function colorFor(label) {
  let h = 0;
  for (const ch of label) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return `hsl(${h} 85% 60%)`;
}

export default function ImageViewer({ images, selectedId, onSelect }) {
  const [zoom, setZoom] = useState(1);
  const [showBoxes, setShowBoxes] = useState(true);
  const [hover, setHover] = useState(-1);
  const [loadFailed, setLoadFailed] = useState(false);

  const img = images.find((i) => i.evidence_id === selectedId) || null;

  useEffect(() => {
    setZoom(1);
    setHover(-1);
    setLoadFailed(false);
  }, [selectedId]);

  if (!img) {
    return (
      <section className="panel">
        <h2>Evidence image &amp; object detection</h2>
        <div className="empty">
          <p><b>No image yet.</b></p>
          <p>Upload an image with the Image “Go” button. The original picture will appear here
            with the objects the detector found.</p>
        </div>
      </section>
    );
  }

  const det = img.object_boxes; // null/undefined = detection did not run
  const boxes = det?.boxes || [];

  return (
    <section className="panel">
      <h2>Evidence image &amp; object detection</h2>

      <div className="viewer-toolbar">
        {images.length > 1 && (
          <select value={selectedId} onChange={(e) => onSelect(e.target.value)} aria-label="Select image">
            {images.map((i, n) => (
              <option key={i.evidence_id} value={i.evidence_id}>
                {n + 1}. {i.file} ({i.evidence_id})
              </option>
            ))}
          </select>
        )}
        <div className="zoom">
          <button type="button" className="btn" onClick={() => setZoom((z) => Math.max(1, +(z - 0.5).toFixed(1)))}
            disabled={zoom <= 1} aria-label="Zoom out">−</button>
          <span aria-live="polite">{Math.round(zoom * 100)}%</span>
          <button type="button" className="btn" onClick={() => setZoom((z) => Math.min(4, +(z + 0.5).toFixed(1)))}
            disabled={zoom >= 4} aria-label="Zoom in">+</button>
          <button type="button" className="btn" onClick={() => setZoom(1)} disabled={zoom === 1}>Fit</button>
          <a className="btn" href={img.image_url} target="_blank" rel="noopener noreferrer">Full size ↗</a>
        </div>
        <label className="check">
          <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
          Show boxes
        </label>
      </div>

      <div className="viewer-scroll">
        <div className="stage" style={{ width: `${zoom * 100}%` }}>
          {loadFailed ? (
            <div className="empty"><p>The original image could not be loaded from the server.</p></div>
          ) : (
            <img src={img.image_url} alt={`Evidence ${img.evidence_id}: ${img.file}`}
              onError={() => setLoadFailed(true)} draggable={false} />
          )}
          {showBoxes && !loadFailed && boxes.map((b, i) => {
            const c = colorFor(b.label);
            const { x1, y1, x2, y2 } = b.box;
            const nearTop = y1 < 0.07;
            return (
              <div key={i} className={`bbox${hover === i ? ' hot' : ''}`}
                style={{ left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`,
                  height: `${(y2 - y1) * 100}%`, borderColor: c }}>
                <span className={`bbox-label${nearTop ? ' inside' : ''}`} style={{ background: c }}>
                  {b.label} {(b.confidence * 100).toFixed(0)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <h3>Detected objects</h3>
      {det == null ? (
        <p className="note warn">
          {img.errors?.boxes
            ? `Object detection failed for this image: ${img.errors.boxes}.`
            : 'No bounding-box data exists for this image (it was uploaded before object detection was added, or detection did not run).'}
        </p>
      ) : boxes.length === 0 ? (
        <p className="note">
          No objects detected above {Math.round(det.threshold * 100)}% confidence ({det.model}).
        </p>
      ) : (
        <ul className="objlist">
          {boxes.map((b, i) => (
            <li key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(-1)}>
              <span className="swatch" style={{ background: colorFor(b.label) }} />
              <span>{b.label}</span>
              <span className="mute">{(b.confidence * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ul>
      )}
      {det && <p className="hint">Detector: {det.model}, threshold {det.threshold}. Boxes are model output, not verified findings.</p>}

      <h3>AI-generated caption</h3>
      <p>{img.caption ? img.caption : <span className="mute">No caption available{img.errors?.caption ? ` (${img.errors.caption})` : ''}.</span>}</p>

      <h3>Scene classification (CLIP zero-shot)</h3>
      {img.scene?.length ? (
        <ul className="objlist">
          {img.scene.map((s) => (
            <li key={s.label}><span>{s.label}</span><span className="mute">{(s.score * 100).toFixed(1)}%</span></li>
          ))}
        </ul>
      ) : (
        <p className="mute">No scene scores available{img.errors?.scene ? ` (${img.errors.scene})` : ''}.</p>
      )}
    </section>
  );
}
