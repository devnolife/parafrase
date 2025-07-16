# Sistem Parafrase Bahasa Indonesia - Hybrid + IndoT5

Sistem parafrase Bahasa Indonesia berbasis kombinasi metode Hybrid (sinonim + transformasi sintaksis) dan model IndoT5 (transformer).

---

## 📁 Struktur Project Minimal

```
parafrase/
├── paraphraser.py            # Semua logika Hybrid, IndoT5, Integrated
├── web_interface.py          # Web UI (Streamlit)
├── test_paraphraser.py       # Semua unit test
├── sinonim_extended.json     # Kamus sinonim
├── requirements.txt          # Dependencies
└── README.md                 # Dokumentasi
```

---

## 🛠️ Instalasi

1. **Clone/download project**
2. **Buat dan aktifkan virtual environment (opsional tapi direkomendasikan):**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Opsional) Install sebagai package pip lokal:**
   - Buat file `setup.py` di root project (lihat contoh di bawah)
   - Install dengan:
     ```bash
     pip install .
     ```

---

## 💻 Penggunaan

### 1. Parafrase via Script (Python)

```python
from paraphraser import HybridParaphraser, IndoT5Paraphraser, IntegratedParaphraser

# Hybrid
hybrid = HybridParaphraser('sinonim_extended.json')
results = hybrid.paraphrase("Pendidikan sangat penting.")

# IndoT5
indot5 = IndoT5Paraphraser()
results = indot5.paraphrase("Pendidikan sangat penting.")

# Integrated (gabungan)
integrated = IntegratedParaphraser()
results = integrated.paraphrase("Pendidikan sangat penting.", method="integrated")
```

### 2. Web Interface (Streamlit)

```bash
streamlit run web_interface.py
```
Jika `streamlit` tidak dikenali, gunakan:
```bash
python -m streamlit run web_interface.py
```

### 3. Testing

```bash
python test_paraphraser.py
```

---

## ⚙️ Pilihan Metode Parafrase
- **hybrid**: Menggunakan kamus sinonim + transformasi sintaksis
- **t5**: Menggunakan model IndoT5 (transformer)
- **integrated**: Gabungan hybrid dan IndoT5
- **best**: Pilih hasil terbaik dari kedua metode

---

## 📊 Contoh Output

**Input:**
```
Pendidikan adalah proses pembelajaran yang sangat penting untuk mengembangkan potensi manusia.
```

**Output (integrated):**
```
1. Edukasi merupakan proses pembelajaran yang amat vital untuk membangun potensi manusia.
2. Proses pembelajaran yang sangat penting untuk mengembangkan potensi manusia adalah pendidikan.
3. Potensi manusia dikembangkan oleh proses pembelajaran yang sangat penting dalam pendidikan.
```

---

## 📝 Catatan
- File sinonim bisa dikustomisasi (`sinonim_extended.json`)
- Untuk model IndoT5, butuh koneksi internet saat pertama kali download model
- Semua pengujian ada di `test_paraphraser.py`

---

## 📦 Contoh setup.py

Buat file `setup.py` di root project dengan isi seperti berikut:
```python
from setuptools import setup, find_packages

setup(
    name='parafrase',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # dependencies utama, bisa juga otomatis dari requirements.txt
    ],
    include_package_data=True,
    description='Sistem Parafrase Bahasa Indonesia (Hybrid + IndoT5)',
    author='Nama Anda',
    author_email='email@domain.com',
    url='https://github.com/username/parafrase',
)
```

Setelah itu, jalankan:
```bash
pip install .
```

---

## 🤝 Kontribusi
- Fork, Pull Request, atau Issue sangat diterima!

---

**Happy Paraphrasing! 🎉**
