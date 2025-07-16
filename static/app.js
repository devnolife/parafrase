document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('paraphrase-form');
  const resultSection = document.getElementById('result-section');
  const resultsDiv = document.getElementById('results');
  const loadExampleBtn = document.getElementById('loadExample');
  const inputText = document.getElementById('inputText');

  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    resultsDiv.innerHTML = '<div class="text-center my-3"><div class="spinner-border" role="status"></div> Memproses...</div>';
    resultSection.style.display = 'block';
    const payload = {
      text: inputText.value,
      method: document.getElementById('method').value,
      num_variations: parseInt(document.getElementById('numVariations').value),
      synonym_rate: parseFloat(document.getElementById('synonymRate').value),
      temperature: parseFloat(document.getElementById('temperature').value),
      top_p: parseFloat(document.getElementById('topP').value)
    };
    try {
      const res = await fetch('/api/paraphrase', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.success) {
        showResults(data.results, data.processing_time);
      } else {
        resultsDiv.innerHTML = `<div class="alert alert-danger">${data.error || 'Terjadi error.'}</div>`;
      }
    } catch (err) {
      resultsDiv.innerHTML = `<div class="alert alert-danger">Gagal memproses: ${err}</div>`;
    }
  });

  function showResults(results, processingTime) {
    if (!results || results.length === 0) {
      resultsDiv.innerHTML = '<div class="alert alert-warning">Tidak ada parafrase valid yang dihasilkan.</div>';
      return;
    }
    let html = `<div class="mb-2"><strong>Waktu proses:</strong> ${processingTime.toFixed(2)} detik</div>`;
    results.forEach((r, i) => {
      html += `<div class="card mb-3">
                <div class="card-body">
                    <h5 class="card-title">Parafrase ${i + 1}</h5>
                    <p class="card-text">${r.text}</p>
                    <ul class="list-unstyled mb-0">
                        <li><strong>Similarity:</strong> ${(r.similarity_score * 100).toFixed(1)}%</li>
                        <li><strong>Method:</strong> ${r.method || '-'}${r.model ? ' (' + r.model + ')' : ''}</li>
                        <li><strong>Confidence:</strong> ${r.confidence !== undefined ? (r.confidence * 100).toFixed(1) + '%' : '-'}</li>
                        <li><strong>Transformasi:</strong> ${r.transformations ? r.transformations.join(', ') : '-'}</li>
                    </ul>
                </div>
            </div>`;
    });
    resultsDiv.innerHTML = html;
  }

  loadExampleBtn.addEventListener('click', async function () {
    try {
      const res = await fetch('/api/examples');
      const data = await res.json();
      if (data.examples && data.examples.length > 0) {
        inputText.value = data.examples[Math.floor(Math.random() * data.examples.length)];
      }
    } catch (err) {
      alert('Gagal memuat contoh.');
    }
  });
}); 
