// Exercise the actual form handler against legacy and upgraded capability responses.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const source = fs.readFileSync(path.join(__dirname, '../webapp/static/app.js'), 'utf8');

async function submit(native, mode = 'all') {
  const values = {strength: '3', seconds: '300', 'max-side': '1920', fps: '30', crf: '12', steps: '6',
    'frequency-mode': 'dct', band: 'mid_high', 'message-bits': '', 'compute-device': 'auto', encoder: 'auto',
    'vae-mode': 'auto', 'denoise-device': 'auto', 'frame-mode': mode, 'export-mode': 'full',
    'random-frames': '100', 'random-seed': '17'};
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {value: values[id] || '', files: [], style: {}, dataset: {},
      classList: {add() {}, remove() {}, toggle() {}}, addEventListener() {},
      querySelector() { return {}; }, scrollIntoView() {}});
    return elements.get(id);
  }
  class FormData {
    constructor() { this.items = new Map(); }
    append(key, value) { this.items.set(key, value); }
    get(key) { return this.items.get(key); }
  }
  const jobs = [], requests = [], storage = new Map();
  const methods = ['dct','carrier','spectral','denoise','vae','geometry','guided','temporal'].map(id => ({id, name: id, ready: true}));
  const context = vm.createContext({document: {getElementById: element, querySelectorAll: () => []},
    localStorage: {getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value)},
    crypto, FormData, setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1,
    fetch: async (url, options = {}) => {
      let body;
      if (url === '/api/capabilities') body = {methods, adaptive_vae: true, fast_vae: true, ffmpeg: true,
        device: 'test', acceleration: {cuda: true}, max_pending_jobs: 50,
        max_processing_side: 1920, ...(native ? {max_processing_frames: 9000} : {})};
      else if (url === '/api/jobs' && options.method === 'POST') {
        const config = JSON.parse(options.body.get('config'));
        requests.push(config);
        if (!native && config.frame_mode === 'all') return {ok: false, json: async () => ({detail: '超过当前模式 2400 帧上限'})};
        body = {id: 'a'.repeat(32), name: 'long.mp4', status: 'queued', phase: '等待处理', progress: 0,
          logs: [], created_at: new Date().toISOString(), config, request_id: 'b'.repeat(32)};
        jobs.push(body);
      } else body = jobs;
      return {ok: true, json: async () => body};
    }});
  vm.runInContext(source, context);
  await new Promise(setImmediate);
  vm.runInContext("selected=['dct','spectral','geometry'];pendingFiles=[{file:{name:'long.mp4',size:1000},requestId:'b'.repeat(32)}];", context);
  await element('run').onclick();
  assert.equal(requests.length, 1);
  const cfg = requests[0];
  assert.equal(cfg.seconds, 300);
  assert.equal(cfg.fps, 30);
  assert.equal(cfg.max_side, 1920);
  assert.equal(cfg.strength, 3);
  assert.deepEqual(cfg.methods, ['dct','spectral','geometry']);
  assert.equal(cfg.export_mode, 'full');
  assert.equal(cfg.frame_mode, mode === 'all' && native ? 'all' : 'random');
  if (mode === 'all' && !native) assert.equal(cfg.random_frames, 9000);
  if (mode === 'random') {
    assert.equal(cfg.random_frames, 100);
    assert.equal(cfg.random_seed, 17);
  }
  assert.equal(jobs.length, 1);
  assert.equal(vm.runInContext('pendingFiles.length', context), 0);
}

(async () => {
  await submit(false);
  await submit(true);
  await submit(false, 'random');
  console.log('Form submission verified: legacy full coverage, native all mode, random selection and quality settings preserved.');
})().catch(error => { console.error(error); process.exitCode = 1; });
