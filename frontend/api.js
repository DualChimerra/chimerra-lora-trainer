// Thin REST client.

async function req(method, url, body) {
    const init = { method, headers: {} };
    if (body !== undefined) {
        init.headers['Content-Type'] = 'application/json';
        init.body = JSON.stringify(body);
    }
    const r = await fetch(url, init);
    if (!r.ok) {
        const text = await r.text();
        throw new Error(`${r.status} ${r.statusText}: ${text}`);
    }
    const ct = r.headers.get('content-type') || '';
    return ct.includes('application/json') ? r.json() : r.text();
}

export const api = {
    getConfig:        () => req('GET',  '/api/config'),
    putConfig:  (cfg) => req('PUT',  '/api/config', cfg),

    fsList:    (path) => req('GET',  `/api/fs/list?path=${encodeURIComponent(path)}`),
    fsScanDataset: (path) => req('GET', `/api/fs/scan_dataset?path=${encodeURIComponent(path)}`),
    fsModels:  (path) => req('GET',  `/api/fs/models?path=${encodeURIComponent(path)}`),
    fsOutputs: (path = '') => req('GET', `/api/fs/outputs?path=${encodeURIComponent(path)}`),
    fsSamples: (path = '') => req('GET', `/api/fs/samples?path=${encodeURIComponent(path)}`),
    fileUrl:   (path) => `/api/fs/file?path=${encodeURIComponent(path)}`,
    thumbUrl:  (path, size = 256) => `/api/fs/thumb?path=${encodeURIComponent(path)}&size=${size}`,

    calcSteps: () => req('GET', '/api/calc/total_steps'),
    previewCommand: (cfg) => req('POST', '/api/calc/preview_command', cfg),

    presetsList:   () => req('GET', '/api/presets'),
    presetsSave:   (p) => req('POST', '/api/presets', p),
    presetsDelete: (name) => req('DELETE', `/api/presets/${encodeURIComponent(name)}`),

    trainStart:  () => req('POST', '/api/train/start'),
    trainStop:   () => req('POST', '/api/train/stop'),
    trainStatus: () => req('GET',  '/api/train/status'),
    clearLogs:   () => req('POST', '/api/train/clear_logs'),
};

export function openSocket(onEvent) {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws`;
    let ws;
    let alive = true;
    let backoff = 600;

    const connect = () => {
        ws = new WebSocket(url);
        ws.onopen = () => { backoff = 600; onEvent({ type: '_open' }); };
        ws.onmessage = (ev) => {
            try { onEvent(JSON.parse(ev.data)); }
            catch (e) { /* ignore malformed */ }
        };
        ws.onclose = () => {
            onEvent({ type: '_close' });
            if (alive) setTimeout(connect, backoff);
            backoff = Math.min(8000, backoff * 1.6);
        };
        ws.onerror = () => { try { ws.close(); } catch {} };
    };
    connect();
    return () => { alive = false; try { ws.close(); } catch {} };
}
