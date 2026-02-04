const CACHE_NAME = 'cw-decoder-v1';
const ASSETS = [
    './',
    './index.html',
    './style.css',
    './main.js',
    './manifest.json',
    './icon.svg',
    '../cw_decoder_quantized.onnx',
    '../cw_decoder.onnx',
    '../dsp.js',
    '../inference.js',
    '../visualization.js',
    '../noise-node.js',
    '../stream-inference.js',
    '../data_gen.js',
    '../audio-processor.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.js',
    'https://unpkg.com/vue@3/dist/vue.esm-browser.js'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(ASSETS);
        })
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            return response || fetch(event.request);
        })
    );
});