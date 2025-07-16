/**
 * Enhanced JavaScript for Indonesian Paraphraser
 * Supports optimized paraphrasing with better UI/UX
 */

// Global variables
let isProcessing = false;
let currentResults = [];
let systemStatus = {};

// DOM elements
const form = document.getElementById('paraphrase-form');
const resultSection = document.getElementById('result-section');
const resultsDiv = document.getElementById('results');
const loadExampleBtn = document.getElementById('loadExample');
const clearTextBtn = document.getElementById('clearText');
const inputText = document.getElementById('inputText');

// Statistics elements
const statusIndicator = document.getElementById('statusIndicator');
const synonymCount = document.getElementById('synonymCount');
const methodCount = document.getElementById('methodCount');
const processingTime = document.getElementById('processingTime');
const totalGenerated = document.getElementById('totalGenerated');
const highQualityCount = document.getElementById('highQualityCount');
const methodUsed = document.getElementById('methodUsed');

// Configuration
const CONFIG = {
  API_ENDPOINTS: {
    PARAPHRASE: '/api/paraphrase',
    EXAMPLES: '/api/examples',
    STATUS: '/api/status'
  },
  QUALITY_THRESHOLDS: {
    HIGH: 0.7,
    MEDIUM: 0.5,
    LOW: 0.3
  },
  DEBOUNCE_DELAY: 300,
  MAX_RETRIES: 3
};

// Initialize application
document.addEventListener('DOMContentLoaded', function () {
  initializeApp();
});

/**
 * Initialize the application
 */
async function initializeApp() {
  try {
    // Load system status
    await loadSystemStatus();

    // Setup event listeners
    setupEventListeners();

    // Setup auto-resize for textarea
    setupTextareaAutoResize();

    // Setup keyboard shortcuts
    setupKeyboardShortcuts();

    // Show initial state
    showInitialState();

    console.log('Application initialized successfully');
  } catch (error) {
    console.error('Error initializing application:', error);
    showAlert('Gagal menginisialisasi aplikasi. Silakan refresh halaman.', 'danger');
  }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Form submission
  form.addEventListener('submit', handleFormSubmit);

  // Button events
  loadExampleBtn.addEventListener('click', loadRandomExample);
  clearTextBtn.addEventListener('click', clearAllText);

  // Input events
  inputText.addEventListener('input', debounce(handleTextInput, CONFIG.DEBOUNCE_DELAY));
  inputText.addEventListener('paste', handleTextPaste);

  // Parameter change events
  document.getElementById('method').addEventListener('change', handleMethodChange);
  document.getElementById('synonymRate').addEventListener('input', handleParameterChange);
  document.getElementById('numVariations').addEventListener('input', handleParameterChange);
  document.getElementById('qualityThreshold').addEventListener('input', handleParameterChange);

  // Advanced parameters
  document.getElementById('temperature').addEventListener('input', handleParameterChange);
  document.getElementById('topP').addEventListener('input', handleParameterChange);
  document.getElementById('repetitionPenalty').addEventListener('input', handleParameterChange);

  // Checkbox events
  document.getElementById('smartReplacement').addEventListener('change', handleParameterChange);
  document.getElementById('preserveKeywords').addEventListener('change', handleParameterChange);

  // Window events
  window.addEventListener('beforeunload', handleBeforeUnload);
  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);
}

/**
 * Setup textarea auto-resize functionality
 */
function setupTextareaAutoResize() {
  inputText.addEventListener('input', autoResizeTextarea);
  // Initial resize
  autoResizeTextarea();
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
  document.addEventListener('keydown', function (e) {
    // Ctrl/Cmd + Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!isProcessing) {
        form.dispatchEvent(new Event('submit'));
      }
    }

    // Ctrl/Cmd + L to load example
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
      e.preventDefault();
      loadRandomExample();
    }

    // Ctrl/Cmd + K to clear text
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      clearAllText();
    }

    // Escape to cancel processing (if implemented)
    if (e.key === 'Escape' && isProcessing) {
      // TODO: Implement cancellation
      console.log('Processing cancellation requested');
    }
  });
}

/**
 * Load system status from API
 */
async function loadSystemStatus() {
  try {
    const response = await fetch(CONFIG.API_ENDPOINTS.STATUS);
    const data = await response.json();

    systemStatus = data;
    updateStatusUI(data);

    return data;
  } catch (error) {
    console.error('Error loading system status:', error);
    updateStatusUI({
      paraphraser_loaded: false,
      synonym_count: 0,
      methods_available: {}
    });
    throw error;
  }
}

/**
 * Update status UI elements
 */
function updateStatusUI(status) {
  statusIndicator.textContent = status.paraphraser_loaded ? '✅' : '❌';
  synonymCount.textContent = status.synonym_count || '0';
  methodCount.textContent = Object.keys(status.methods_available || {}).length;

  // Update status indicator tooltip
  statusIndicator.title = status.paraphraser_loaded ?
    'Sistem siap digunakan' :
    'Sistem belum siap';
}

/**
 * Handle form submission
 */
async function handleFormSubmit(e) {
  e.preventDefault();

  if (isProcessing) {
    return;
  }

  const textValue = inputText.value.trim();
  if (!textValue) {
    showAlert('Mohon masukkan teks yang ingin diparafrase.', 'warning');
    inputText.focus();
    return;
  }

  // Validate text length
  if (textValue.length < 10) {
    showAlert('Teks terlalu pendek. Minimal 10 karakter.', 'warning');
    return;
  }

  if (textValue.length > 1000) {
    showAlert('Teks terlalu panjang. Maksimal 1000 karakter.', 'warning');
    return;
  }

  // Check system status
  if (!systemStatus.paraphraser_loaded) {
    showAlert('Sistem parafrase belum siap. Silakan tunggu atau refresh halaman.', 'danger');
    return;
  }

  await processParaphrase(textValue);
}

/**
 * Process paraphrase request
 */
async function processParaphrase(text) {
  isProcessing = true;
  showProcessingState();

  const payload = buildPayload(text);
  let retryCount = 0;

  while (retryCount < CONFIG.MAX_RETRIES) {
    try {
      const response = await fetch(CONFIG.API_ENDPOINTS.PARAPHRASE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.success) {
        currentResults = data.results;
        showResults(data);

        // Analytics (if implemented)
        trackParaphraseSuccess(payload.method, data.results.length);

        break;
      } else {
        throw new Error(data.error || 'Terjadi kesalahan saat memproses.');
      }

    } catch (error) {
      console.error(`Attempt ${retryCount + 1} failed:`, error);
      retryCount++;

      if (retryCount >= CONFIG.MAX_RETRIES) {
        handleProcessingError(error);
        break;
      }

      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
    }
  }

  isProcessing = false;
  hideProcessingState();
}

/**
 * Build payload for API request
 */
function buildPayload(text) {
  return {
    text: text,
    method: document.getElementById('method').value,
    num_variations: parseInt(document.getElementById('numVariations').value),
    synonym_rate: parseFloat(document.getElementById('synonymRate').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    top_p: parseFloat(document.getElementById('topP').value),
    repetition_penalty: parseFloat(document.getElementById('repetitionPenalty').value),
    smart_replacement: document.getElementById('smartReplacement').checked,
    preserve_keywords: document.getElementById('preserveKeywords').checked,
    quality_threshold: parseFloat(document.getElementById('qualityThreshold').value)
  };
}

/**
 * Show processing state
 */
function showProcessingState() {
  const submitBtn = form.querySelector('button[type="submit"]');
  submitBtn.innerHTML = '<span class="loading-spinner"></span> Memproses...';
  submitBtn.disabled = true;

  // Disable form inputs
  const inputs = form.querySelectorAll('input, select, textarea');
  inputs.forEach(input => input.disabled = true);

  // Show processing in results
  resultsDiv.innerHTML = `
        <div class="text-center my-4">
            <div class="loading-spinner" style="width: 40px; height: 40px; border-width: 4px;"></div>
            <p class="mt-3">Sedang memproses parafrase...</p>
            <small class="text-muted">Mohon tunggu, proses ini mungkin membutuhkan beberapa detik</small>
        </div>
    `;
  resultSection.style.display = 'block';

  // Auto-scroll to results
  setTimeout(() => {
    resultSection.scrollIntoView({ behavior: 'smooth' });
  }, 100);
}

/**
 * Hide processing state
 */
function hideProcessingState() {
  const submitBtn = form.querySelector('button[type="submit"]');
  submitBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Parafrase';
  submitBtn.disabled = false;

  // Enable form inputs
  const inputs = form.querySelectorAll('input, select, textarea');
  inputs.forEach(input => input.disabled = false);
}

/**
 * Show results
 */
function showResults(data) {
  // Update statistics
  processingTime.textContent = data.processing_time.toFixed(2);
  totalGenerated.textContent = data.total_generated;
  highQualityCount.textContent = data.high_quality_count;
  methodUsed.textContent = data.method_used;

  if (!data.results || data.results.length === 0) {
    showEmptyResults();
    return;
  }

  // Generate results HTML
  let html = '';
  data.results.forEach((result, index) => {
    html += generateResultCard(result, index);
  });

  resultsDiv.innerHTML = html;
  resultSection.style.display = 'block';

  // Add animations
  setTimeout(() => {
    const cards = resultsDiv.querySelectorAll('.result-card');
    cards.forEach((card, index) => {
      setTimeout(() => {
        card.classList.add('fade-in');
      }, index * 100);
    });
  }, 100);

  // Auto-scroll to results
  setTimeout(() => {
    resultSection.scrollIntoView({ behavior: 'smooth' });
  }, 200);
}

/**
 * Generate result card HTML
 */
function generateResultCard(result, index) {
  const qualityClass = getQualityClass(result.quality_score);
  const qualityText = getQualityText(result.quality_score);
  const cardId = `result-${index}`;

  return `
        <div class="card result-card" id="${cardId}">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-start mb-3">
                    <h5 class="card-title mb-0">
                        <i class="fas fa-quote-left"></i> Parafrase ${index + 1}
                    </h5>
                    <div>
                        <span class="quality-badge ${qualityClass}">${qualityText}</span>
                        <span class="method-info ms-2">${result.method || 'Unknown'}</span>
                    </div>
                </div>
                
                <p class="card-text fs-5 mb-3">${escapeHtml(result.text)}</p>
                
                ${generateProgressBars(result)}
                
                ${generateTransformationBadges(result)}
                
                ${generateReplacementInfo(result)}
                
                <div class="mt-3">
                    <button class="btn btn-outline-primary btn-sm me-2" onclick="copyToClipboard('${escapeForJs(result.text)}', '${cardId}')">
                        <i class="fas fa-copy"></i> Salin
                    </button>
                    <button class="btn btn-outline-secondary btn-sm me-2" onclick="useAsInput('${escapeForJs(result.text)}')">
                        <i class="fas fa-redo"></i> Parafrase Ulang
                    </button>
                    <button class="btn btn-outline-info btn-sm" onclick="showResultDetails('${cardId}', ${index})">
                        <i class="fas fa-info-circle"></i> Detail
                    </button>
                </div>
            </div>
        </div>
    `;
}

/**
 * Generate progress bars for result metrics
 */
function generateProgressBars(result) {
  return `
        <div class="row mb-3">
            <div class="col-md-3">
                <small class="text-muted">
                    <i class="fas fa-percentage"></i> Similarity: 
                    <strong>${(result.similarity_score * 100).toFixed(1)}%</strong>
                </small>
                <div class="progress-custom">
                    <div class="progress-bar-custom" style="width: ${result.similarity_score * 100}%"></div>
                </div>
            </div>
            <div class="col-md-3">
                <small class="text-muted">
                    <i class="fas fa-star"></i> Quality: 
                    <strong>${(result.quality_score * 100).toFixed(1)}%</strong>
                </small>
                <div class="progress-custom">
                    <div class="progress-bar-custom" style="width: ${result.quality_score * 100}%"></div>
                </div>
            </div>
            <div class="col-md-3">
                <small class="text-muted">
                    <i class="fas fa-thumbs-up"></i> Confidence: 
                    <strong>${(result.confidence * 100).toFixed(1)}%</strong>
                </small>
                <div class="progress-custom">
                    <div class="progress-bar-custom" style="width: ${result.confidence * 100}%"></div>
                </div>
            </div>
            <div class="col-md-3">
                <small class="text-muted">
                    <i class="fas fa-exchange-alt"></i> Transformasi: 
                    <strong>${result.transformations ? result.transformations.length : 0}</strong>
                </small>
            </div>
        </div>
    `;
}

/**
 * Generate transformation badges
 */
function generateTransformationBadges(result) {
  if (!result.transformations || result.transformations.length === 0) {
    return '';
  }

  return `
        <div class="mt-3">
            <small class="text-muted">
                <i class="fas fa-tools"></i> Transformasi yang diterapkan:
            </small>
            <div class="mt-1">
                ${result.transformations.map(t =>
    `<span class="badge bg-secondary me-1">${t}</span>`
  ).join('')}
            </div>
        </div>
    `;
}

/**
 * Generate replacement info
 */
function generateReplacementInfo(result) {
  if (!result.replacement_details || result.replacement_details.length === 0) {
    return '';
  }

  return `
        <div class="mt-3">
            <small class="text-muted">
                <i class="fas fa-sync-alt"></i> Penggantian sinonim: 
                <strong>${result.replacement_details.length}</strong>
            </small>
        </div>
    `;
}

/**
 * Show empty results message
 */
function showEmptyResults() {
  resultsDiv.innerHTML = `
        <div class="alert alert-warning text-center">
            <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
            <h5>Tidak ada parafrase berkualitas tinggi yang dihasilkan</h5>
            <p class="mb-3">Coba lakukan hal berikut:</p>
            <ul class="list-unstyled">
                <li>• Turunkan threshold kualitas</li>
                <li>• Ubah metode parafrase</li>
                <li>• Sesuaikan parameter sinonim</li>
                <li>• Gunakan teks yang lebih panjang</li>
            </ul>
            <button class="btn btn-primary mt-2" onclick="toggleAdvanced()">
                <i class="fas fa-cogs"></i> Ubah Parameter
            </button>
        </div>
    `;
}

/**
 * Handle processing error
 */
function handleProcessingError(error) {
  console.error('Processing error:', error);

  let errorMessage = 'Terjadi kesalahan saat memproses parafrase.';

  if (error.message.includes('HTTP 400')) {
    errorMessage = 'Data yang dikirim tidak valid. Silakan periksa input Anda.';
  } else if (error.message.includes('HTTP 500')) {
    errorMessage = 'Terjadi kesalahan server. Silakan coba lagi nanti.';
  } else if (error.message.includes('Failed to fetch')) {
    errorMessage = 'Gagal terhubung ke server. Periksa koneksi internet Anda.';
  }

  resultsDiv.innerHTML = `
        <div class="alert alert-danger text-center">
            <i class="fas fa-exclamation-circle fa-2x mb-3"></i>
            <h5>Gagal Memproses Parafrase</h5>
            <p>${errorMessage}</p>
            <button class="btn btn-outline-danger mt-2" onclick="location.reload()">
                <i class="fas fa-refresh"></i> Refresh Halaman
            </button>
        </div>
    `;

  resultSection.style.display = 'block';
}

/**
 * Load random example text
 */
async function loadRandomExample() {
  try {
    const response = await fetch(CONFIG.API_ENDPOINTS.EXAMPLES);
    const data = await response.json();

    if (data.examples && data.examples.length > 0) {
      const randomExample = data.examples[Math.floor(Math.random() * data.examples.length)];
      inputText.value = randomExample;
      autoResizeTextarea();

      // Focus on textarea
      inputText.focus();

      // Show success message
      showAlert('Contoh teks berhasil dimuat!', 'success');
    }
  } catch (error) {
    console.error('Error loading example:', error);
    showAlert('Gagal memuat contoh teks. Silakan coba lagi.', 'warning');
  }
}

/**
 * Clear all text and results
 */
function clearAllText() {
  inputText.value = '';
  autoResizeTextarea();
  resultSection.style.display = 'none';
  currentResults = [];

  // Focus on textarea
  inputText.focus();

  showAlert('Teks dan hasil telah dibersihkan.', 'info');
}

/**
 * Handle text input changes
 */
function handleTextInput() {
  const textLength = inputText.value.length;

  // Update character count (if element exists)
  const charCount = document.getElementById('charCount');
  if (charCount) {
    charCount.textContent = `${textLength}/1000`;
    charCount.className = textLength > 1000 ? 'text-danger' : 'text-muted';
  }

  // Auto-resize textarea
  autoResizeTextarea();
}

/**
 * Handle text paste
 */
function handleTextPaste(e) {
  setTimeout(() => {
    const text = inputText.value;
    if (text.length > 1000) {
      inputText.value = text.substring(0, 1000);
      showAlert('Teks telah dipotong menjadi 1000 karakter.', 'warning');
    }
    autoResizeTextarea();
  }, 0);
}

/**
 * Handle method change
 */
function handleMethodChange() {
  const method = document.getElementById('method').value;

  // Show/hide relevant parameters based on method
  const t5Params = document.querySelectorAll('[data-method="t5"]');
  const hybridParams = document.querySelectorAll('[data-method="hybrid"]');

  t5Params.forEach(param => {
    param.style.display = (method === 't5' || method === 'integrated' || method === 'best') ? 'block' : 'none';
  });

  hybridParams.forEach(param => {
    param.style.display = (method === 'hybrid' || method === 'integrated' || method === 'best') ? 'block' : 'none';
  });
}

/**
 * Handle parameter changes
 */
function handleParameterChange() {
  // Clear previous results to avoid confusion
  if (currentResults.length > 0) {
    const parameterChanged = document.createElement('div');
    parameterChanged.className = 'alert alert-info alert-dismissible fade show';
    parameterChanged.innerHTML = `
            <i class="fas fa-info-circle"></i> Parameter telah diubah. Klik <strong>Parafrase</strong> untuk hasil baru.
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

    resultSection.insertBefore(parameterChanged, resultsDiv);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      if (parameterChanged.parentNode) {
        parameterChanged.remove();
      }
    }, 5000);
  }
}

/**
 * Auto-resize textarea
 */
function autoResizeTextarea() {
  inputText.style.height = 'auto';
  inputText.style.height = Math.min(inputText.scrollHeight, 300) + 'px';
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text, cardId) {
  try {
    await navigator.clipboard.writeText(text);

    // Show success feedback
    const button = document.querySelector(`#${cardId} .btn-outline-primary`);
    if (button) {
      const originalText = button.innerHTML;
      button.innerHTML = '<i class="fas fa-check"></i> Tersalin!';
      button.classList.remove('btn-outline-primary');
      button.classList.add('btn-success');

      setTimeout(() => {
        button.innerHTML = originalText;
        button.classList.remove('btn-success');
        button.classList.add('btn-outline-primary');
      }, 2000);
    }

    showAlert('Teks berhasil disalin ke clipboard!', 'success');
  } catch (err) {
    console.error('Error copying text:', err);

    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();

    try {
      document.execCommand('copy');
      showAlert('Teks berhasil disalin ke clipboard!', 'success');
    } catch (fallbackErr) {
      showAlert('Gagal menyalin teks. Silakan salin secara manual.', 'danger');
    }

    document.body.removeChild(textArea);
  }
}

/**
 * Use result as input for new paraphrase
 */
function useAsInput(text) {
  inputText.value = text;
  autoResizeTextarea();

  // Scroll to input
  inputText.scrollIntoView({ behavior: 'smooth' });
  inputText.focus();

  showAlert('Teks telah dimasukkan sebagai input baru.', 'info');
}

/**
 * Show detailed information about a result
 */
function showResultDetails(cardId, index) {
  const result = currentResults[index];
  if (!result) return;

  const modal = document.createElement('div');
  modal.className = 'modal fade';
  modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-info-circle"></i> Detail Parafrase ${index + 1}
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <h6><i class="fas fa-quote-left"></i> Teks Hasil</h6>
                            <p class="border p-3 rounded">${escapeHtml(result.text)}</p>
                        </div>
                        <div class="col-md-6">
                            <h6><i class="fas fa-chart-bar"></i> Metrik Kualitas</h6>
                            <ul class="list-unstyled">
                                <li><strong>Similarity:</strong> ${(result.similarity_score * 100).toFixed(1)}%</li>
                                <li><strong>Quality:</strong> ${(result.quality_score * 100).toFixed(1)}%</li>
                                <li><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</li>
                                <li><strong>Method:</strong> ${result.method}</li>
                            </ul>
                        </div>
                    </div>
                    
                    ${result.transformations && result.transformations.length > 0 ? `
                        <div class="mb-3">
                            <h6><i class="fas fa-tools"></i> Transformasi yang Diterapkan</h6>
                            <div class="d-flex flex-wrap gap-2">
                                ${result.transformations.map(t =>
    `<span class="badge bg-primary">${t}</span>`
  ).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${result.replacement_details && result.replacement_details.length > 0 ? `
                        <div class="mb-3">
                            <h6><i class="fas fa-sync-alt"></i> Penggantian Sinonim</h6>
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Posisi</th>
                                            <th>Kata Asli</th>
                                            <th>Pengganti</th>
                                            <th>Alternatif</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${result.replacement_details.map(detail => `
                                            <tr>
                                                <td>${detail.position + 1}</td>
                                                <td><code>${detail.original}</code></td>
                                                <td><code>${detail.replacement}</code></td>
                                                <td>${detail.available_synonyms ? detail.available_synonyms.length : 0} sinonim</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ` : ''}
                    
                    <div class="mb-3">
                        <h6><i class="fas fa-cogs"></i> Parameter yang Digunakan</h6>
                        <div class="row">
                            <div class="col-md-6">
                                <ul class="list-unstyled small">
                                    <li><strong>Synonym Rate:</strong> ${result.synonym_rate_used || 'N/A'}</li>
                                    <li><strong>Complexity Score:</strong> ${result.complexity_score ? result.complexity_score.toFixed(3) : 'N/A'}</li>
                                    <li><strong>Attempt Number:</strong> ${result.attempt_number || 'N/A'}</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <ul class="list-unstyled small">
                                    <li><strong>Generation Rank:</strong> ${result.generation_rank || 'N/A'}</li>
                                    <li><strong>Fluency Score:</strong> ${result.fluency_score ? result.fluency_score.toFixed(3) : 'N/A'}</li>
                                    <li><strong>Combined Score:</strong> ${result.combined_score ? result.combined_score.toFixed(3) : 'N/A'}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Tutup</button>
                    <button type="button" class="btn btn-primary" onclick="copyToClipboard('${escapeForJs(result.text)}', 'modal'); this.closest('.modal').querySelector('[data-bs-dismiss]').click();">
                        <i class="fas fa-copy"></i> Salin Teks
                    </button>
                </div>
            </div>
        </div>
    `;

  document.body.appendChild(modal);
  const modalInstance = new bootstrap.Modal(modal);
  modalInstance.show();

  // Remove modal from DOM after hiding
  modal.addEventListener('hidden.bs.modal', () => {
    document.body.removeChild(modal);
  });
}

/**
 * Toggle advanced parameters
 */
function toggleAdvanced() {
  const advancedParams = document.getElementById('advancedParams');
  const advancedIcon = document.getElementById('advancedIcon');

  if (advancedParams.style.display === 'none') {
    advancedParams.style.display = 'block';
    advancedIcon.classList.remove('fa-chevron-down');
    advancedIcon.classList.add('fa-chevron-up');

    // Smooth animation
    advancedParams.style.opacity = '0';
    advancedParams.style.transform = 'translateY(-10px)';

    setTimeout(() => {
      advancedParams.style.opacity = '1';
      advancedParams.style.transform = 'translateY(0)';
    }, 10);
  } else {
    advancedParams.style.opacity = '0';
    advancedParams.style.transform = 'translateY(-10px)';

    setTimeout(() => {
      advancedParams.style.display = 'none';
      advancedIcon.classList.remove('fa-chevron-up');
      advancedIcon.classList.add('fa-chevron-down');
    }, 300);
  }
}

/**
 * Show initial state
 */
function showInitialState() {
  // Add welcome message if needed
  const welcomeMessage = localStorage.getItem('paraphraser_welcome_shown');
  if (!welcomeMessage) {
    setTimeout(() => {
      showAlert('Selamat datang! Sistem parafrase siap digunakan. Coba masukkan teks atau klik "Contoh" untuk memulai.', 'info');
      localStorage.setItem('paraphraser_welcome_shown', 'true');
    }, 1000);
  }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info', duration = 5000) {
  const alertDiv = document.createElement('div');
  alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
  alertDiv.style.position = 'fixed';
  alertDiv.style.top = '20px';
  alertDiv.style.right = '20px';
  alertDiv.style.zIndex = '9999';
  alertDiv.style.minWidth = '300px';
  alertDiv.style.maxWidth = '500px';

  const icon = getAlertIcon(type);
  alertDiv.innerHTML = `
        <i class="fas fa-${icon}"></i> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

  document.body.appendChild(alertDiv);

  // Auto-dismiss
  setTimeout(() => {
    if (alertDiv.parentNode) {
      alertDiv.remove();
    }
  }, duration);
}

/**
 * Get alert icon based on type
 */
function getAlertIcon(type) {
  const icons = {
    success: 'check-circle',
    danger: 'exclamation-circle',
    warning: 'exclamation-triangle',
    info: 'info-circle'
  };
  return icons[type] || 'info-circle';
}

/**
 * Get quality class based on score
 */
function getQualityClass(score) {
  if (score >= CONFIG.QUALITY_THRESHOLDS.HIGH) return 'quality-high';
  if (score >= CONFIG.QUALITY_THRESHOLDS.MEDIUM) return 'quality-medium';
  return 'quality-low';
}

/**
 * Get quality text based on score
 */
function getQualityText(score) {
  if (score >= CONFIG.QUALITY_THRESHOLDS.HIGH) return 'Tinggi';
  if (score >= CONFIG.QUALITY_THRESHOLDS.MEDIUM) return 'Sedang';
  return 'Rendah';
}

/**
 * Escape HTML for safe insertion
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Escape text for JavaScript strings
 */
function escapeForJs(text) {
  return text.replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\n/g, '\\n');
}

/**
 * Debounce function to limit function calls
 */
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * Handle before unload
 */
function handleBeforeUnload(e) {
  if (isProcessing) {
    e.preventDefault();
    e.returnValue = 'Parafrase sedang diproses. Yakin ingin meninggalkan halaman?';
  }
}

/**
 * Handle online status
 */
function handleOnline() {
  showAlert('Koneksi internet tersambung kembali.', 'success');
}

/**
 * Handle offline status
 */
function handleOffline() {
  showAlert('Koneksi internet terputus. Beberapa fitur mungkin tidak berfungsi.', 'warning');
}

/**
 * Track paraphrase success (for analytics)
 */
function trackParaphraseSuccess(method, resultCount) {
  // TODO: Implement analytics tracking
  console.log(`Paraphrase successful: method=${method}, results=${resultCount}`);
}

/**
 * Export functions for global access
 */
window.paraphraser = {
  copyToClipboard,
  useAsInput,
  showResultDetails,
  toggleAdvanced,
  loadRandomExample,
  clearAllText
};
