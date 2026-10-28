// Every call goes to a route that exists in app.py. Errors are always turned into ApiError with a readable message,
// including non-JSON error pages (proxy failures, crashes).
const BASE = globalThis.__API_BASE__ || '';

export class ApiError extends Error {
  constructor(message, status = 0, body = null) { super(message); this.status = status; this.body = body; }
}

async function request(path, opts = {}) {
  let res;
  try { res = await fetch(BASE + path, opts); }
  catch { throw new ApiError('Cannot reach the server. Is Flask running on port 5000?'); }
  const isJson = (res.headers.get('content-type') || '').includes('application/json');
  let body = null;
  if (isJson) { try { body = await res.json(); } catch { body = null; } }
  if (!res.ok) {
    const msg = body?.error
      || (res.status === 413 ? 'File too large.' : null)
      || (res.status >= 500 ? `Server error (${res.status}). See the Flask terminal.` : `Request failed (${res.status}).`);
    throw new ApiError(msg, res.status, body);
  }
  return body;
}

const json = (method, data) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data ?? {}) });
const enc = encodeURIComponent;

export const api = {
  config: () => request('/api/config'),
  listCases: () => request('/api/cases'),
  createCase: (case_id) => request('/api/cases', json('POST', case_id ? { case_id } : {})),
  timeline: (cid) => request(`/api/timeline/${enc(cid)}`),
  analyze(kind, file, caseId) {            // kind: image | video | audio  -> POST /api/analyze/<kind>
    const fd = new FormData();
    if (caseId) fd.append('case_id', caseId);
    fd.append('file', file);
    return request(`/api/analyze/${kind}`, { method: 'POST', body: fd });
  },
  reanalyze: (cid, eid) => request(`/api/reanalyze/${enc(cid)}/${enc(eid)}`, { method: 'POST' }),
  classify: (cid) => request(`/api/classify/${enc(cid)}`, { method: 'POST' }),
  chat: (cid, question) => request(`/api/chat/${enc(cid)}`, json('POST', { question })),
  integrity: (cid) => request(`/api/integrity/${enc(cid)}`),
  audit: (cid) => request(`/api/audit/${enc(cid)}`),
  evidenceFileUrl: (cid, eid) => `${BASE}/api/cases/${enc(cid)}/evidence/${enc(eid)}/file`,
  frameUrl: (cid, eid, ref) => `${BASE}/api/cases/${enc(cid)}/evidence/${enc(eid)}/frames/${enc(ref)}`,
  async downloadReport(cid) {              // GET /api/report/<cid> -> PDF; errors arrive as JSON
    let res;
    try { res = await fetch(`${BASE}/api/report/${enc(cid)}`); }
    catch { throw new ApiError('Cannot reach the server. Is Flask running on port 5000?'); }
    if (!res.ok) {
      let msg = `Report failed (${res.status}).`;
      try { msg = (await res.json()).error || msg; } catch { /* non-JSON error page */ }
      throw new ApiError(msg, res.status);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), { href: url, download: `forensic_${cid}.pdf` });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  },
};
