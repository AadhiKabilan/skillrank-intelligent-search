async function submitSearch() {
    const query = document.getElementById('queryInput').value.trim();
    const k = parseInt(document.getElementById('topK').value);

    if (!query) return;

    // UI States
    const btn = document.getElementById('searchBtn');
    const loadDiv = document.getElementById('loading');
    const resDiv = document.getElementById('results');
    const errDiv = document.getElementById('error');

    btn.disabled = true;
    loadDiv.classList.remove('hidden');
    resDiv.classList.add('hidden');
    errDiv.classList.add('hidden');

    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ q: query, k: k })
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(errText || `Server error: ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (e) {
        errDiv.textContent = e.message;
        errDiv.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        loadDiv.classList.add('hidden');
    }
}

function renderResults(data) {
    const resDiv = document.getElementById('results');
    const answerDiv = document.getElementById('llmAnswer');
    const hitsDiv = document.getElementById('hitsList');

    // Formatting answer (simple markdown-like replacement)
    let answerHtml = data.answer
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    answerDiv.innerHTML = answerHtml;

    hitsDiv.innerHTML = '';
    data.hits.forEach(hit => {
        const card = document.createElement('div');
        card.className = 'paper-card';

        const scorePct = (hit.score * 100).toFixed(1);

        card.innerHTML = `
            <div class="paper-header">
                <h3><a href="${hit.url}" target="_blank">${hit.title}</a></h3>
                <span class="score">${scorePct}% Match</span>
            </div>
            <p class="paper-summary">${hit.text}...</p>
            <div class="paper-meta">ID: ${hit.id}</div>
        `;
        hitsDiv.appendChild(card);
    });

    resDiv.classList.remove('hidden');
}