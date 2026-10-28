import React, { useState } from 'react';
import BoxStage from './BoxStage.jsx';
import Assessment from './Assessment.jsx';
import { Card, Empty, ErrorBox, FirearmNote, pct } from './common.jsx';

export default function ImageResult({ im, onReanalyze, busy }) {
  const [showAll, setShowAll] = useState(false);
  const [imgErr, setImgErr] = useState(false);
  const dets = im.detections || [];
  const source = im.object_boxes?.boxes || [];                      // same detector output, overlay payload
  const boxes = dets.length ? dets.filter((d) => showAll || d.relevant !== false).map((d) => ({ label: d.label, confidence: d.confidence ?? d.score, box: d.box, source: d.source })) : source;
  const hidden = dets.length - dets.filter((d) => d.relevant !== false).length;
  return (
    <Card title={im.file} className="result" actions={<button className="btn small" disabled={busy} onClick={() => onReanalyze(im.evidence_id)}>Re-run analysis</button>}>
      <p className="muted small">Evidence {im.evidence_id}{im.width ? ` · analysed at ${im.width}×${im.height}px` : ''}</p>
      <ErrorBox error={im.errors && `Some steps failed, so their results are missing (not negative): ${Object.keys(im.errors).join(', ')}.`} />
      <div className="cols">
        <div>
          {imgErr ? <div className="alert error">The original image could not be loaded.</div>
            : <BoxStage src={im.image_url} boxes={boxes} alt={`Evidence ${im.file}`} onError={() => setImgErr(true)} />}
          {hidden > 0 && <label className="check"><input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> Show {hidden} other COCO class detection(s)</label>}
          <p className="muted small">Boxes are real model predictions: {im.model_info?.detector?.model || 'RT-DETR'} (COCO classes, no firearm class){im.firearm_detector?.status === 'ok' ? ' plus the separate firearm model (dashed boxes)' : ''}. An empty overlay does not mean no weapon.</p>
          <FirearmNote info={im.firearm_detector} />
        </div>
        <div>
          <h4>Detected objects</h4>
          {dets.length ? (
            <table className="tbl"><thead><tr><th>Class</th><th>Confidence</th><th>Source</th></tr></thead>
              <tbody>{dets.map((d, i) => <tr key={i}><td>{d.label}{d.overlaps_coco && <span className="tagline">overlaps COCO: {d.overlaps_coco.join(', ')}</span>}</td><td>{pct(d.confidence ?? d.score)}</td><td>{d.source === 'firearm_model' ? 'firearm model' : 'RT-DETR'}</td></tr>)}</tbody></table>
          ) : <Empty>{im.errors?.detector ? 'Detector failed.' : 'Nothing detected above the threshold.'}</Empty>}
          <h4>Caption <span className="tagline">BLIP · model output, may be wrong</span></h4>
          {im.caption ? <p>{im.caption}</p> : <Empty>{im.errors?.caption ? 'Captioning failed.' : 'No caption.'}</Empty>}
          <h4>Scene scores <span className="tagline">CLIP · relative, sum to 100%</span></h4>
          {im.scene?.length ? im.scene.map((s) => (
            <div className="bar" key={s.label}><span>{s.label}</span><div className="track"><div className="fill" style={{ width: pct(s.score) }} /></div><b>{pct(s.score)}</b></div>
          )) : <Empty>{im.errors?.scene ? 'Scene classification failed.' : 'No scene scores.'}</Empty>}
        </div>
      </div>
      <h4>Preliminary incident assessment</h4>
      <Assessment a={im.assessment} />
    </Card>
  );
}
