import React from 'react';
import { labelColor, pct } from './common.jsx';

// The stage is exactly as large as the displayed <img>, and boxes are positioned in percent of the stage using
// the normalised 0..1 coordinates from the backend, so they stay aligned at any display size.
export default function BoxStage({ src, boxes = [], alt, onError }) {
  return (
    <div className="stage">
      <img src={src} alt={alt} onError={onError} />
      <div className="boxes" aria-hidden="true">
        {boxes.map((b, i) => {
          const { x1, y1, x2, y2 } = b.box;
          const p = (v) => `${+(v * 100).toFixed(3)}%`;
          const color = labelColor(b.label);
          return (
            <div key={i} className={`box${b.source === 'firearm_model' ? ' firearm' : ''}`}
                 style={{ left: p(x1), top: p(y1), width: p(x2 - x1), height: p(y2 - y1), borderColor: color }}>
              <span className="tag" style={{ background: color }}>{b.label} {pct(b.confidence)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
