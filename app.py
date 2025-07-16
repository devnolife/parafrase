#!/usr/bin/env python3
"""
Flask Web Interface untuk Sistem Parafrase Bahasa Indonesia
Pengganti Streamlit yang lebih lightweight dan kompatibel
Jalankan dengan: python app.py
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import time
import traceback
from typing import Dict, List, Optional

# Import paraphraser modules
try:
    from paraphraser import HybridParaphraser, IndoT5Paraphraser, IntegratedParaphraser
    PARAPHRASER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Paraphraser import failed: {e}")
    PARAPHRASER_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Global paraphraser instance
paraphraser = None
paraphraser_type = None
system_info = {}

def initialize_paraphraser():
    """Initialize the paraphraser system"""
    global paraphraser, paraphraser_type, system_info
    
    if not PARAPHRASER_AVAILABLE:
        return False
    
    try:
        # Try integrated paraphraser first
        paraphraser = IntegratedParaphraser()
        paraphraser_type = "integrated"
        system_info = {
            "methods_available": paraphraser.methods_available,
            "status": "integrated_loaded"
        }
        return True
    except Exception as e:
        print(f"Failed to initialize integrated paraphraser: {e}")
        
        # Fallback to hybrid only
        synonym_files = ['sinonim_extended.json', 'sinonim.json']
        for file in synonym_files:
            if os.path.exists(file):
                try:
                    paraphraser = HybridParaphraser(file)
                    paraphraser_type = "hybrid_only"
                    system_info = {
                        "methods_available": {"hybrid": True, "t5": False},
                        "status": "hybrid_only",
                        "synonym_file": file
                    }
                    return True
                except Exception as e2:
                    print(f"Failed to initialize hybrid paraphraser: {e2}")
        
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/system-info')
def get_system_info():
    """Get system information"""
    return jsonify({
        "paraphraser_available": PARAPHRASER_AVAILABLE,
        "paraphraser_type": paraphraser_type,
        "system_info": system_info
    })

@app.route('/api/paraphrase', methods=['POST'])
def paraphrase_text():
    """Process paraphrase request"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Get parameters
        method = data.get('method', 'hybrid')
        num_variations = data.get('num_variations', 3)
        synonym_rate = data.get('synonym_rate', 0.3)
        temperature = data.get('temperature', 0.8)
        top_p = data.get('top_p', 0.95)
        
        if not paraphraser:
            return jsonify({"error": "Paraphraser not initialized"}), 500
        
        start_time = time.time()
        
        # Generate paraphrases
        if paraphraser_type == "integrated":
            hybrid_params = {'synonym_rate': synonym_rate}
            t5_params = {'temperature': temperature, 'top_p': top_p}
            
            results = paraphraser.paraphrase(
                text,
                method=method,
                num_variations=num_variations,
                hybrid_params=hybrid_params,
                t5_params=t5_params
            )
        else:
            # Hybrid only
            results = paraphraser.paraphrase(
                text,
                num_variations=num_variations,
                synonym_rate=synonym_rate
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return jsonify({
            "success": True,
            "results": results,
            "processing_time": processing_time,
            "method_used": method,
            "total_results": len(results)
        })
        
    except Exception as e:
        print(f"Error in paraphrase_text: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/benchmark', methods=['POST'])
def benchmark_methods():
    """Run benchmark comparison"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        if paraphraser_type != "integrated":
            return jsonify({"error": "Benchmark only available for integrated paraphraser"}), 400
        
        # Run benchmark
        test_sentences = [text]
        benchmark_results = paraphraser.benchmark_comparison(test_sentences)
        
        return jsonify({
            "success": True,
            "benchmark_results": benchmark_results
        })
        
    except Exception as e:
        print(f"Error in benchmark_methods: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/examples')
def get_examples():
    """Get example texts"""
    examples = [
        "Pendidikan adalah proses pembelajaran yang sangat penting untuk mengembangkan potensi manusia.",
        "Teknologi informasi membantu mempercepat proses komunikasi di era modern.",
        "Pemerintah harus memberikan perhatian khusus terhadap masalah kemiskinan.",
        "Mahasiswa belajar dengan tekun untuk mencapai prestasi yang baik.",
        "Perusahaan menggunakan strategi pemasaran yang inovatif untuk menarik pelanggan.",
        "Artificial intelligence akan mengubah cara kita bekerja di masa depan.",
        "Perubahan iklim merupakan tantangan global yang memerlukan tindakan segera."
    ]
    
    return jsonify({"examples": examples})

if __name__ == '__main__':
    print("🚀 Starting Flask Paraphraser Server...")
    
    # Initialize paraphraser
    if initialize_paraphraser():
        print(f"✅ Paraphraser initialized successfully ({paraphraser_type})")
        print(f"📊 Available methods: {system_info.get('methods_available', {})}")
    else:
        print("❌ Failed to initialize paraphraser")
        print("⚠️  Server will run but paraphrasing will not work")
    
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n🌐 Server starting at http://localhost:5000")
    print("📝 Use Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 
