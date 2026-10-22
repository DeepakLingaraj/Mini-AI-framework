document.addEventListener('DOMContentLoaded', () => {
    // Navigation Tabs
    const navItems = document.querySelectorAll('.nav-item');
    const tabPages = document.querySelectorAll('.tab-page');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    const tabHeadings = {
        playground: { title: 'Inference Lab', subtitle: 'Interactive LLM token streaming & multi-backend execution' },
        tokenizer: { title: 'Tokenizer Inspector', subtitle: 'Subword BPE decomposition, token IDs, and visual span mapping' },
        models: { title: 'Model Architecture', subtitle: 'NanoTransformer tensor tree, weight shapes, and memory footprint' },
        benchmark: { title: 'Benchmark Suite', subtitle: 'Comparative throughput, latency, TTFT, and GFLOPS profiler' }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');

            navItems.forEach(nav => nav.classList.remove('active'));
            tabPages.forEach(page => page.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.add('active');

            if (tabHeadings[targetTab]) {
                pageTitle.textContent = tabHeadings[targetTab].title;
                pageSubtitle.textContent = tabHeadings[targetTab].subtitle;
            }
        });
    });

    // 1. Fetch Info Stats on startup
    fetch('/api/info')
        .then(res => res.json())
        .then(data => {
            document.getElementById('info-model-name').textContent = data.model.name;
            document.getElementById('info-params').textContent = (data.parameters / 1e6).toFixed(1) + 'M';
            document.getElementById('info-vocab').textContent = data.vocab_size.toLocaleString();
            document.getElementById('info-memory').textContent = data.memory_mb + ' MB';
        })
        .catch(err => console.log('Info fetch fallback active.'));

    // 2. Token Streaming Inference
    const btnGenerate = document.getElementById('btn-generate');
    const outputStream = document.getElementById('output-stream');
    const progressBar = document.getElementById('generation-progress');
    const metricTps = document.getElementById('metric-tps');
    const metricTtft = document.getElementById('metric-ttft');
    const metricLat = document.getElementById('metric-lat');

    let eventSource = null;

    btnGenerate.addEventListener('click', () => {
        const prompt = document.getElementById('input-prompt').value;
        const backend = document.getElementById('select-backend').value;
        const maxTokens = document.getElementById('input-max-tokens').value;
        const temp = document.getElementById('input-temperature').value;

        if (eventSource) {
            eventSource.close();
        }

        outputStream.innerHTML = '';
        progressBar.style.width = '0%';
        btnGenerate.disabled = true;
        btnGenerate.innerHTML = '<span>⏳</span> Generating...';

        const url = `/api/generate_stream?prompt=${encodeURIComponent(prompt)}&backend=${backend}&max_tokens=${maxTokens}&temperature=${temp}`;
        eventSource = new EventSource(url);

        let tokenCount = 0;

        eventSource.onmessage = (e) => {
            const chunk = JSON.parse(e.data);
            tokenCount++;

            outputStream.innerHTML += escapeHtml(chunk.text);
            outputStream.scrollTop = outputStream.scrollHeight;

            const tps = (tokenCount / (chunk.total_time_ms / 1000.0)).toFixed(1);
            metricTps.textContent = tps;
            metricLat.textContent = chunk.latency_ms.toFixed(1) + ' ms';
            if (tokenCount === 1) {
                metricTtft.textContent = chunk.latency_ms.toFixed(1) + ' ms';
            }

            const pct = Math.min(100, (tokenCount / maxTokens) * 100);
            progressBar.style.width = pct + '%';

            if (chunk.is_finished) {
                eventSource.close();
                btnGenerate.disabled = false;
                btnGenerate.innerHTML = '<span>🚀</span> Run Inference';
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            btnGenerate.disabled = false;
            btnGenerate.innerHTML = '<span>🚀</span> Run Inference';
        };
    });

    // 3. Tokenizer Inspector
    const tokInput = document.getElementById('input-tok-text');
    const tokenChips = document.getElementById('token-chips');
    const tokenTableBody = document.getElementById('token-table-body');

    function updateTokenizer() {
        const text = tokInput.value;
        fetch(`/api/tokenize?text=${encodeURIComponent(text)}`)
            .then(res => res.json())
            .then(data => {
                tokenChips.innerHTML = '';
                tokenTableBody.innerHTML = '';

                data.pieces.forEach((p, idx) => {
                    // Chips
                    const chip = document.createElement('div');
                    chip.className = 'token-chip';
                    chip.textContent = p.text.replace(/\n/g, '\\n') + ` [${p.id}]`;
                    tokenChips.appendChild(chip);

                    // Table row
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${idx}</td>
                        <td><code>${p.id}</code></td>
                        <td><b>${escapeHtml(p.text)}</b></td>
                        <td>[${p.start}:${p.end}]</td>
                        <td><span class="metric-pill">${p.is_special ? 'Special' : 'Subword'}</span></td>
                    `;
                    tokenTableBody.appendChild(row);
                });
            });
    }

    tokInput.addEventListener('input', updateTokenizer);
    updateTokenizer();

    // 4. Benchmark Chart Initialization
    const ctx = document.getElementById('benchmarkChart').getContext('2d');
    let benchChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['NumPy SIMD Vectorized', 'Modular MAX Engine Adapter', 'Mojo SIMD Kernel Layer'],
            datasets: [{
                label: 'Throughput (Tokens / sec)',
                data: [142.5, 218.4, 356.2],
                backgroundColor: [
                    'rgba(139, 92, 246, 0.7)',
                    'rgba(6, 182, 212, 0.7)',
                    'rgba(16, 185, 129, 0.7)'
                ],
                borderColor: [
                    '#8b5cf6',
                    '#06b6d4',
                    '#10b981'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#9ca3af' } }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af' }
                }
            }
        }
    });

    const btnRunBench = document.getElementById('btn-run-bench');
    btnRunBench.addEventListener('click', () => {
        btnRunBench.disabled = true;
        btnRunBench.innerHTML = '<span>⏳</span> Benchmarking...';

        fetch('/api/benchmark?max_tokens=40')
            .then(res => res.json())
            .then(data => {
                const results = data.results;
                const labels = results.map(r => r.backend);
                const tpsData = results.map(r => r.tokens_per_sec);

                benchChart.data.labels = labels;
                benchChart.data.datasets[0].data = tpsData;
                benchChart.update();

                // Update table
                const benchTableBody = document.getElementById('bench-table-body');
                benchTableBody.innerHTML = '';
                results.forEach(r => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><b>${r.backend}</b></td>
                        <td><b style="color:var(--accent-emerald)">${r.tokens_per_sec}</b></td>
                        <td>${r.ttft_ms}</td>
                        <td>${r.avg_latency_ms}</td>
                        <td>${r.p99_latency_ms}</td>
                        <td>${r.gflops}</td>
                    `;
                    benchTableBody.appendChild(tr);
                });

                btnRunBench.disabled = false;
                btnRunBench.innerHTML = '<span>⚡</span> Run Live Benchmark';
            })
            .catch(() => {
                btnRunBench.disabled = false;
                btnRunBench.innerHTML = '<span>⚡</span> Run Live Benchmark';
            });
    });

    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
