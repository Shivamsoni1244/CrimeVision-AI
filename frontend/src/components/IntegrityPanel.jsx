import React, { useState } from 'react';
import { Card, ErrorBox, Spinner } from './common.jsx';
import { api } from '../api.js';

// The shape of /api/integrity and /api/audit is defined in storage.py, so it is shown as returned.
export default function IntegrityPanel({ caseId }) {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const load = async () => {
    setBusy(true); setError('');
    try { const [integrity, audit] = await Promise.all([api.integrity(caseId), api.audit(caseId)]); setData({ integrity, audit }); }
    catch (e) { setError(e.message); setData(null); }
    finally { setBusy(false); }
  };
  return (
    <Card title="Integrity & audit" actions={<button className="btn small" onClick={load} disabled={!caseId || busy}>{busy ? <Spinner label="Loading…" /> : 'Verify'}</button>}>
      <ErrorBox error={error} />
      {!data ? <p className="empty">Press Verify to check stored evidence hashes and view the audit log.</p> : (
        <>
          <h4>Integrity check</h4><pre className="data">{JSON.stringify(data.integrity, null, 2)}</pre>
          <h4>Audit log</h4><pre className="data">{JSON.stringify(data.audit, null, 2)}</pre>
        </>
      )}
    </Card>
  );
}
