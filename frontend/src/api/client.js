// Reusable Flask API client. Never throws raw parse errors: every failure becomes an ApiError
// with a readable message, even when the server answers with an HTML error page.

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

export const KINDS = {
  image: { label: 'Image', accept: 'image/*', mime: 'image/' },
  video: { label: 'Video', accept: 'video/*', mime: 'video/' },
  audio: { label: 'Audio', accept: 'audio/*', mime: 'audio/' },
};

function messageFor(status, data) {
  if (data && typeof data.error === 'string') return data.error;
  if (status === 413) return 'File is too large for the server.';
  if (status === 404) return 'Not found.';
  if (status >= 500) return `Server error (HTTP ${status}). Check the Flask terminal.`;
  return `Request failed (HTTP ${status}).`;
}

async function readJson(res) {
  const text = await res.text();
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    return null;
  }
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new ApiError('Cannot reach the server. Check that Flask is running on port 5000.');
  }
  const data = await readJson(res);
  if (!res.ok) throw new ApiError(messageFor(res.status, data), res.status);
  if (data === null) {
    throw new ApiError(`Server returned a non-JSON response (HTTP ${res.status}).`, res.status);
  }
  return data;
}

const enc = encodeURIComponent;

export const api = {
  config: () => request('/api/config'),
  listCases: () => request('/api/cases'),
  createCase: (caseId) =>
    request('/api/cases', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(caseId ? { case_id: caseId } : {}),
    }),
  timeline: (cid) => request(`/api/timeline/${enc(cid)}`),
  evidence: (cid) => request(`/api/cases/${enc(cid)}/evidence`),
  integrity: (cid) => request(`/api/integrity/${enc(cid)}`),
  audit: (cid) => request(`/api/audit/${enc(cid)}`),
  classify: (cid) => request(`/api/classify/${enc(cid)}`, { method: 'POST' }),
  chat: (cid, question) =>
    request(`/api/chat/${enc(cid)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    }),

  // Returns a Blob for the PDF; errors (JSON) are converted to ApiError.
  async downloadReport(cid) {
    let res;
    try {
      res = await fetch(`/api/report/${enc(cid)}`);
    } catch {
      throw new ApiError('Cannot reach the server. Check that Flask is running on port 5000.');
    }
    if (!res.ok) {
      const data = await readJson(res);
      throw new ApiError(messageFor(res.status, data), res.status);
    }
    return res.blob();
  },

  // FormData is sent without a manual Content-Type so the browser adds the multipart boundary.
  // XMLHttpRequest is used (instead of fetch) because it reports upload progress.
  upload(kind, file, caseId, { onProgress, onUploaded } = {}) {
    return new Promise((resolve, reject) => {
      const fd = new FormData();
      fd.append('file', file);
      if (caseId) fd.append('case_id', caseId);
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `/api/analyze/${kind}`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress?.(Math.round((e.loaded / e.total) * 100));
      };
      xhr.upload.onload = () => onUploaded?.();
      xhr.onerror = () =>
        reject(new ApiError('Network error during upload. Check that Flask is running.'));
      xhr.onload = () => {
        let data = null;
        try {
          data = JSON.parse(xhr.responseText);
        } catch {
          /* non-JSON body, handled below */
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          if (data && data.case_id) resolve(data);
          else reject(new ApiError('Upload response did not include a case ID.', xhr.status));
        } else {
          reject(new ApiError(messageFor(xhr.status, data), xhr.status));
        }
      };
      xhr.send(fd);
    });
  },
};
