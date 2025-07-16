#!/usr/bin/env python3
"""
Flask Web App for Indonesian Paraphraser
Enhanced with better accuracy and IndoT5 + Synonym integration
"""

from flask import Flask, render_template, request, jsonify
import json
import time
import os
import logging
from paraphraser import IntegratedParaphraser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global paraphraser instance
paraphraser = None

def initialize_paraphraser():
    """Initialize paraphraser with optimized settings"""
    global paraphraser
    try:
        paraphraser = IntegratedParaphraser(
            synonym_dict_path='sinonim.json',
            t5_model_name="LazarusNLP/IndoNanoT5-base",
            enable_hybrid=True,
            enable_t5=True
        )
        logger.info("Paraphraser initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize paraphraser: {e}")
        paraphraser = None

# Initialize paraphraser on startup
initialize_paraphraser()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/paraphrase', methods=['POST'])
def paraphrase_api():
    """API endpoint for paraphrasing"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'Text is required'})
        
        text = data['text'].strip()
        if not text:
            return jsonify({'success': False, 'error': 'Text cannot be empty'})
        
        # Extract parameters
        method = data.get('method', 'integrated')
        num_variations = min(int(data.get('num_variations', 3)), 5)
        synonym_rate = float(data.get('synonym_rate', 0.4))
        temperature = float(data.get('temperature', 0.8))
        top_p = float(data.get('top_p', 0.95))
        
        # Check if paraphraser is available
        if not paraphraser:
            return jsonify({'success': False, 'error': 'Paraphraser not initialized'})
        
        # Record start time
        start_time = time.time()
        
        # Prepare parameters for different methods
        hybrid_params = {
            'synonym_rate': synonym_rate,
            'use_smart_replacement': True,
            'preserve_keywords': True
        }
        
        t5_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_length': 128,
            'num_beams': 5,
            'do_sample': True,
            'early_stopping': True,
            'repetition_penalty': 1.3,
            'no_repeat_ngram_size': 3
        }
        
        # Generate paraphrases
        results = paraphraser.paraphrase(
            text=text,
            method=method,
            num_variations=num_variations,
            hybrid_params=hybrid_params,
            t5_params=t5_params
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Filter and enhance results
        enhanced_results = []
        for result in results:
            # Add quality score
            quality_score = calculate_quality_score(text, result)
            result['quality_score'] = quality_score
            
            # Only include high-quality results
            if quality_score > 0.3:
                enhanced_results.append(result)
        
        # Sort by quality score
        enhanced_results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'results': enhanced_results,
            'processing_time': processing_time,
            'original_text': text,
            'method_used': method,
            'total_generated': len(results),
            'high_quality_count': len(enhanced_results)
        })
        
    except Exception as e:
        logger.error(f"Error in paraphrase_api: {e}")
        return jsonify({'success': False, 'error': str(e)})

def calculate_quality_score(original: str, result: dict) -> float:
    """Calculate quality score for paraphrase result"""
    try:
        # Base score from similarity (inverse - lower similarity = higher quality for paraphrase)
        similarity = result.get('similarity_score', 0.5)
        base_score = 1.0 - similarity
        
        # Confidence score
        confidence = result.get('confidence', 0.5)
        
        # Length ratio score (penalize very short or very long paraphrases)
        original_len = len(original.split())
        result_len = len(result['text'].split())
        length_ratio = result_len / original_len if original_len > 0 else 0
        
        length_score = 1.0
        if length_ratio < 0.5 or length_ratio > 2.0:
            length_score = 0.3
        elif length_ratio < 0.7 or length_ratio > 1.5:
            length_score = 0.7
        
        # Transformation score (more transformations = better)
        transformations = result.get('transformations', [])
        transformation_score = min(len(transformations) / 3.0, 1.0)
        
        # Replacement score (synonym replacements)
        replacements = result.get('replacement_details', [])
        replacement_score = min(len(replacements) / 5.0, 1.0)
        
        # Combine scores with weights
        quality_score = (
            base_score * 0.3 +
            confidence * 0.25 +
            length_score * 0.2 +
            transformation_score * 0.15 +
            replacement_score * 0.1
        )
        
        return min(max(quality_score, 0.0), 1.0)
        
    except Exception as e:
        logger.error(f"Error calculating quality score: {e}")
        return 0.5

@app.route('/api/examples')
def examples_api():
    """API endpoint for example texts"""
    examples = [
        "Teknologi kecerdasan buatan sangat penting untuk kemajuan masa depan.",
        "Pendidikan adalah kunci utama dalam mengembangkan potensi manusia.",
        "Pemerintah memberikan perhatian khusus terhadap masalah kemiskinan.",
        "Mahasiswa harus belajar dengan tekun untuk mencapai prestasi yang baik.",
        "Perusahaan menggunakan strategi pemasaran yang inovatif untuk menarik pelanggan.",
        "Komunikasi yang efektif sangat diperlukan dalam era digital modern.",
        "Indonesia memiliki kekayaan alam yang melimpah dan beragam.",
        "Penelitian ilmiah membutuhkan metode yang sistematis dan objektif.",
        "Globalisasi membawa dampak positif dan negatif bagi negara berkembang.",
        "Kesehatan mental sama pentingnya dengan kesehatan fisik dalam kehidupan."
    ]
    
    return jsonify({'examples': examples})

@app.route('/api/status')
def status_api():
    """API endpoint for system status"""
    status = {
        'paraphraser_loaded': paraphraser is not None,
        'methods_available': {},
        'synonym_count': 0
    }
    
    if paraphraser:
        status['methods_available'] = paraphraser.methods_available
        if paraphraser.hybrid_paraphraser:
            status['synonym_count'] = len(paraphraser.hybrid_paraphraser.synonym_dict)
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
