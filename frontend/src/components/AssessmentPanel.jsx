export default function AssessmentPanel({ analysis, running, onClassify }) {
  return (
    <section className="panel">
      <h2>Crime assessment</h2>
      <button type="button" className="btn primary" onClick={onClassify} disabled={running}>
        {running ? 'Classifying…' : 'Classify crime'}
      </button>
      {analysis ? (
        <div className="assessment">
          <p className="verdict">
            <b>{analysis.crime_type || 'Unclassified'}</b>
            <span className="badge">confidence: {String(analysis.confidence ?? 'N/A')}</span>
          </p>
          {analysis.reasoning && <p>{analysis.reasoning}</p>}
          {analysis.scenario && <p className="mute">{analysis.scenario}</p>}
          <p className="hint">AI-generated assessment based on the case evidence. It requires human review.</p>
        </div>
      ) : (
        <p className="hint">No assessment yet. Load a case with evidence, then run “Classify crime”.</p>
      )}
    </section>
  );
}
