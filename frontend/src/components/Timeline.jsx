export default function Timeline({ items }) {
  return (
    <section className="panel">
      <h2>Forensic timeline</h2>
      {items.length === 0 ? (
        <p className="hint">No timeline events yet. Upload evidence or load a case.</p>
      ) : (
        <ul className="timeline">
          {items.map((e, i) => (
            <li key={`${e.event_id ?? 'e'}-${i}`}>
              <b className="time">{e.t == null ? 'untimed' : `${Number(e.t).toFixed(1)}s`}</b>
              <span className="mute">
                {' '}{e.source || ''} {e.event_id || ''}
              </span>
              {e.review_status && <span className="badge">{e.review_status}</span>}
              <br />
              <span className={e.type === 'weapon' ? 'weapon' : ''}>{e.text || ''}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
