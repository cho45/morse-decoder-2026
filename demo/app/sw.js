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
    '../multi-stream-manager.js',
    '../peak-detector.js',
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
    // Network First strategy
    event.respondWith(
        fetch(event.request)
            .then((response) => {
                // Check if we received a valid response
                if (!response || response.status !== 200 || response.type === 'error') {
                    return response;
                }

                // Clone the response
                const responseToCache = response.clone();

                caches.open(CACHE_NAME)
                    .then((cache) => {
                        cache.put(event.request, responseToCache);
                    });

                return response;
            })
            .catch(() => {
                // If fetch fails (offline), try cache
                return caches.match(event.request);
            })
    );
});