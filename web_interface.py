#!/usr/bin/env python3
"""
Streamlit Web Interface untuk Sistem Parafrase Hybrid + IndoT5
Jalankan dengan: streamlit run web_interface.py
"""

try:
    import streamlit as st
    from hybrid_paraphraser import HybridParaphraser
    from indot5_paraphraser import IndoT5Paraphraser
    from integrated_paraphraser import IntegratedParaphraser
    import json
    import os
    import time
    
    # Konfigurasi halaman
    st.set_page_config(
        page_title="Sistem Parafrase Bahasa Indonesia",
        page_icon="🔄",
        layout="wide"
    )
    
    # Inisialisasi session state
    if 'paraphraser' not in st.session_state:
        # Coba gunakan integrated paraphraser
        try:
            st.session_state.paraphraser = IntegratedParaphraser()
            st.session_state.paraphraser_type = "integrated"
            st.session_state.system_info = st.session_state.paraphraser.get_system_info()
        except Exception as e:
            st.error(f"Failed to initialize integrated paraphraser: {e}")
            # Fallback ke hybrid
            synonym_files = ['sinonim_extended.json', 'sinonim.json']
            synonym_file = None
            
            for file in synonym_files:
                if os.path.exists(file):
                    synonym_file = file
                    break
            
            if synonym_file:
                st.session_state.paraphraser = HybridParaphraser(synonym_file)
                st.session_state.paraphraser_type = "hybrid_only"
                st.session_state.system_info = {"methods_available": {"hybrid": True, "t5": False}}
            else:
                st.error("File kamus sinonim tidak ditemukan!")
                st.stop()
    
    # Header
    st.title("🔄 Sistem Parafrase Bahasa Indonesia")
    st.markdown("### Hybrid + IndoT5 + Integrated Methods")
    
    # Sidebar untuk konfigurasi
    st.sidebar.header("⚙️ Konfigurasi")
    
    # Tampilkan info sistem
    if st.sidebar.checkbox("Tampilkan Info Sistem"):
        st.sidebar.subheader("📊 System Information")
        if st.session_state.paraphraser_type == "integrated":
            methods_available = st.session_state.system_info.get('methods_available', {})
            st.sidebar.write(f"**Available Methods:**")
            for method, available in methods_available.items():
                icon = "✅" if available else "❌"
                st.sidebar.write(f"{icon} {method.capitalize()}")
            
            if st.session_state.system_info.get('hybrid_info'):
                hybrid_info = st.session_state.system_info['hybrid_info']
                st.sidebar.write(f"**Hybrid Info:**")
                st.sidebar.write(f"Synonyms: {hybrid_info.get('synonym_count', 'N/A')}")
            
            if st.session_state.system_info.get('t5_info'):
                t5_info = st.session_state.system_info['t5_info']
                st.sidebar.write(f"**T5 Info:**")
                st.sidebar.write(f"Model: {t5_info.get('model_name', 'N/A')}")
                st.sidebar.write(f"Device: {t5_info.get('device', 'N/A')}")
        else:
            st.sidebar.write("**Mode:** Hybrid Only")
    
    # Pilihan metode
    st.sidebar.subheader("🎯 Metode Parafrase")
    
    if st.session_state.paraphraser_type == "integrated":
        methods_available = st.session_state.system_info.get('methods_available', {})
        
        # Filter metode yang tersedia
        available_methods = []
        if methods_available.get('hybrid', False):
            available_methods.append("Hybrid")
        if methods_available.get('t5', False):
            available_methods.append("IndoT5")
        if len(available_methods) > 1:
            available_methods.extend(["Integrated", "Best"])
        
        selected_method = st.sidebar.selectbox(
            "Pilih Metode:",
            available_methods,
            index=len(available_methods)-1 if len(available_methods) > 1 else 0,
            help="Pilih metode parafrase yang ingin digunakan"
        )
        
        method_map = {
            "Hybrid": "hybrid",
            "IndoT5": "t5", 
            "Integrated": "integrated",
            "Best": "best"
        }
        
        selected_method_key = method_map.get(selected_method, "hybrid")
    else:
        selected_method = "Hybrid"
        selected_method_key = "hybrid"
        st.sidebar.write("**Available:** Hybrid Only")
    
    # Parameter konfigurasi
    st.sidebar.subheader("🔧 Parameter")
    
    num_variations = st.sidebar.slider(
        "Jumlah Variasi Parafrase",
        min_value=1,
        max_value=5,
        value=3,
        help="Jumlah variasi parafrase yang akan dihasilkan"
    )
    
    # Parameter khusus untuk hybrid
    if selected_method_key in ["hybrid", "integrated", "best"]:
        st.sidebar.write("**Hybrid Parameters:**")
        synonym_rate = st.sidebar.slider(
            "Rate Penggantian Sinonim",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Probabilitas penggantian kata dengan sinonim"
        )
    else:
        synonym_rate = 0.3
    
    # Parameter khusus untuk T5
    if selected_method_key in ["t5", "integrated", "best"]:
        st.sidebar.write("**T5 Parameters:**")
        t5_temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Mengontrol keragaman output (lebih tinggi = lebih kreatif)"
        )
        t5_top_p = st.sidebar.slider(
            "Top-p",
            min_value=0.1,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Nucleus sampling parameter"
        )
    else:
        t5_temperature = 0.8
        t5_top_p = 0.95
    
    # Area input
    st.header("📝 Input Teks")
    
    # Contoh teks
    example_texts = [
        "Pilih contoh...",
        "Pendidikan adalah proses pembelajaran yang sangat penting untuk mengembangkan potensi manusia.",
        "Teknologi informasi membantu mempercepat proses komunikasi di era modern.",
        "Pemerintah harus memberikan perhatian khusus terhadap masalah kemiskinan.",
        "Mahasiswa belajar dengan tekun untuk mencapai prestasi yang baik.",
        "Perusahaan menggunakan strategi pemasaran yang inovatif untuk menarik pelanggan.",
        "Artificial intelligence akan mengubah cara kita bekerja di masa depan.",
        "Perubahan iklim merupakan tantangan global yang memerlukan tindakan segera."
    ]
    
    selected_example = st.selectbox("Pilih contoh teks:", example_texts)
    
    # Input teks
    if selected_example != "Pilih contoh...":
        input_text = st.text_area(
            "Masukkan teks yang ingin diparafrase:",
            value=selected_example,
            height=100,
            help="Masukkan kalimat atau paragraf dalam bahasa Indonesia"
        )
    else:
        input_text = st.text_area(
            "Masukkan teks yang ingin diparafrase:",
            height=100,
            help="Masukkan kalimat atau paragraf dalam bahasa Indonesia"
        )
    
    # Tombol proses
    col1, col2 = st.columns([1, 4])
    
    with col1:
        process_button = st.button("🔄 Generate Parafrase", type="primary")
    
    with col2:
        if st.button("🔄 Benchmark Methods", help="Bandingkan performa semua metode"):
            if input_text.strip() and st.session_state.paraphraser_type == "integrated":
                st.session_state.run_benchmark = True
            else:
                st.warning("Masukkan teks dan pastikan integrated paraphraser tersedia!")
    
    # Proses parafrase
    if process_button:
        if input_text.strip():
            with st.spinner("Sedang memproses..."):
                try:
                    start_time = time.time()
                    
                    # Siapkan parameter
                    hybrid_params = {'synonym_rate': synonym_rate}
                    t5_params = {
                        'temperature': t5_temperature,
                        'top_p': t5_top_p
                    }
                    
                    # Generate parafrase
                    if st.session_state.paraphraser_type == "integrated":
                        results = st.session_state.paraphraser.paraphrase(
                            input_text,
                            method=selected_method_key,
                            num_variations=num_variations,
                            hybrid_params=hybrid_params,
                            t5_params=t5_params
                        )
                    else:
                        # Fallback ke hybrid
                        results = st.session_state.paraphraser.paraphrase(
                            input_text,
                            num_variations=num_variations,
                            synonym_rate=synonym_rate
                        )
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    if results:
                        st.success(f"✅ Berhasil menghasilkan {len(results)} parafrase dalam {processing_time:.2f} detik!")
                        
                        # Tampilkan hasil
                        st.header("📊 Hasil Parafrase")
                        
                        # Tampilkan teks asli
                        st.subheader("🔸 Teks Asli:")
                        st.info(input_text)
                        
                        # Tampilkan metode yang digunakan
                        st.subheader(f"🔸 Metode: {selected_method}")
                        
                        # Tampilkan parafrase
                        st.subheader("🔸 Hasil Parafrase:")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Parafrase {i} (Similarity: {result['similarity_score']:.2f})"):
                                st.write("**Teks:**")
                                st.write(result['text'])
                                
                                # Metadata
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Method:**")
                                    st.write(result.get('method', 'Unknown'))
                                    
                                    if 'model' in result:
                                        st.write("**Model:**")
                                        st.write(result['model'])
                                
                                with col2:
                                    st.write("**Similarity Score:**")
                                    st.progress(result['similarity_score'])
                                    
                                    st.write("**Confidence:**")
                                    st.progress(result.get('confidence', 0))
                                
                                # Transformasi untuk hybrid
                                if 'transformations' in result and result['transformations']:
                                    st.write("**Transformasi yang diterapkan:**")
                                    for transform in result['transformations']:
                                        st.write(f"• {transform}")
                                
                                # Posisi penggantian
                                if 'replaced_positions' in result and result['replaced_positions']:
                                    st.write(f"**Posisi kata yang diganti:** {result['replaced_positions']}")
                        
                        # Statistik
                        st.header("📈 Statistik")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_similarity = sum(r['similarity_score'] for r in results) / len(results)
                            st.metric("Rata-rata Similarity", f"{avg_similarity:.2f}")
                        
                        with col2:
                            avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
                            st.metric("Rata-rata Confidence", f"{avg_confidence:.2f}")
                        
                        with col3:
                            total_transformations = sum(len(r.get('transformations', [])) for r in results)
                            st.metric("Total Transformasi", total_transformations)
                        
                        with col4:
                            st.metric("Waktu Pemrosesan", f"{processing_time:.2f}s")
                        
                        # Download hasil
                        st.header("💾 Download Hasil")
                        
                        # Buat string hasil untuk download
                        download_content = f"TEKS ASLI:\n{input_text}\n\n"
                        download_content += f"METODE: {selected_method}\n"
                        download_content += f"WAKTU PEMROSESAN: {processing_time:.2f} detik\n\n"
                        download_content += "HASIL PARAFRASE:\n" + "="*50 + "\n"
                        
                        for i, result in enumerate(results, 1):
                            download_content += f"\n{i}. {result['text']}\n"
                            download_content += f"   Method: {result.get('method', 'Unknown')}\n"
                            download_content += f"   Similarity: {result['similarity_score']:.2f}\n"
                            download_content += f"   Confidence: {result.get('confidence', 0):.2f}\n"
                            
                            if 'transformations' in result and result['transformations']:
                                download_content += f"   Transformations: {', '.join(result['transformations'])}\n"
                            
                            if 'model' in result:
                                download_content += f"   Model: {result['model']}\n"
                        
                        st.download_button(
                            label="📥 Download Hasil",
                            data=download_content,
                            file_name=f"hasil_parafrase_{selected_method.lower()}.txt",
                            mime="text/plain"
                        )
                        
                    else:
                        st.warning("Tidak ada parafrase valid yang dihasilkan. Coba ubah konfigurasi atau gunakan teks yang berbeda.")
                
                except Exception as e:
                    st.error(f"Terjadi error: {str(e)}")
                    st.error("Pastikan dependencies telah diinstall dengan: pip install -r requirements.txt")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu!")
    
    # Benchmark mode
    if hasattr(st.session_state, 'run_benchmark') and st.session_state.run_benchmark:
        st.header("🏆 Benchmark Comparison")
        
        with st.spinner("Menjalankan benchmark untuk semua metode..."):
            try:
                # Siapkan test sentences
                test_sentences = [input_text] if input_text.strip() else [
                    "Pendidikan adalah proses pembelajaran yang sangat penting.",
                    "Teknologi informasi membantu mempercepat komunikasi."
                ]
                
                benchmark_results = st.session_state.paraphraser.benchmark_comparison(test_sentences)
                
                st.subheader("📊 Hasil Benchmark")
                
                # Tampilkan hasil untuk setiap metode
                methods = benchmark_results.get('methods', {})
                
                for method, data in methods.items():
                    st.write(f"**{method.upper()} Method:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Time", f"{data['total_time']:.2f}s")
                    
                    with col2:
                        st.metric("Avg Time/Sentence", f"{data['avg_time']:.2f}s")
                    
                    with col3:
                        st.metric("Total Paraphrases", data['total_paraphrases'])
                    
                    st.write("---")
                
                # Comparison
                if 'comparison' in benchmark_results and len(methods) > 1:
                    st.subheader("⚡ Perbandingan")
                    
                    comp = benchmark_results['comparison']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Speed Comparison:**")
                        if comp['speed_comparison']['hybrid_faster']:
                            st.success("Hybrid method is faster")
                        else:
                            st.info("T5 method is faster")
                        st.write(f"Speed ratio: {comp['speed_comparison']['speed_ratio']:.2f}x")
                    
                    with col2:
                        st.write("**Output Comparison:**")
                        st.write(f"Hybrid: {comp['output_comparison']['hybrid_count']} paraphrases")
                        st.write(f"T5: {comp['output_comparison']['t5_count']} paraphrases")
                
            except Exception as e:
                st.error(f"Error in benchmark: {e}")
        
        # Reset benchmark flag
        st.session_state.run_benchmark = False
    
    # Informasi tambahan di sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Informasi")
    st.sidebar.markdown("""
    **Metode yang tersedia:**
    - 🔄 **Hybrid**: Sinonim + transformasi sintaksis
    - 🤖 **IndoT5**: Model T5 untuk bahasa Indonesia
    - 🔗 **Integrated**: Gabungan Hybrid + IndoT5
    - 🏆 **Best**: Pilih hasil terbaik dari semua metode
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Tips:**")
    st.sidebar.markdown("""
    - Gunakan kalimat yang lengkap dan jelas
    - Rate sinonim rendah = perubahan minimal
    - Temperature tinggi = hasil lebih kreatif
    - Method 'Best' memberikan hasil terbaik
    - Coba benchmark untuk membandingkan metode
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model T5 yang didukung:**")
    st.sidebar.markdown("""
    - LazarusNLP/IndoNanoT5-base
    - ramsrigouthamg/t5_paraphraser
    - cahya/t5-base-indonesian-summarization-cased
    """)

except ImportError:
    print("Streamlit tidak terinstall. Install dengan: pip install streamlit")
    print("Atau jalankan demo.py untuk mode console.")
except Exception as e:
    print(f"Error: {e}")
    print("Pastikan semua dependencies telah diinstall: pip install -r requirements.txt")
