# ARIA — Asisten Risiko Akademik
Chatbot deteksi dini risiko keterlambatan kelulusan mahasiswa.

## Cara menjalankan

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Jalankan server
```
python app.py
```

### 3. Buka di browser
- **Laptop:** http://localhost:5000
- **HP (jaringan sama):** http://<IP_LAPTOP>:5000
  - Cari IP laptop: `ipconfig` (Windows) atau `ifconfig` (Mac/Linux)
  - Contoh: http://192.168.1.5:5000

## Struktur project
```
chatbot/
├── app.py              ← Backend Flask + logika inferensi
├── requirements.txt
├── README.md
└── templates/
    └── index.html      ← UI chatbot mobile-style
```

## Alur inferensi
1. Setup awal: mahasiswa input IPK, SKS, semester, matkul wajib sisa
2. Input harian: tugas tertunda, kualitas tidur, kondisi psikologis, faktor eksternal
3. Tahap 1 — Forward Chaining: cek kondisi pasti (IPK < 2.0, semester >= 14, dll)
4. Tahap 2 — Decision Tree (simulasi max_depth=4): traversal berdasarkan fitur
5. Output: label risiko + probabilitas + faktor dominan + rekomendasi
