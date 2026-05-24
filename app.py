from flask import Flask, render_template, request, jsonify, session
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = ""

# ── Load model yang sudah dilatih dari dataset.csv ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bundle   = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
MODEL    = bundle["model"]
FITUR    = bundle["fitur"]   # urutan fitur HARUS sama dengan saat training
LABELS   = bundle["label"]   # ['Tepat_Waktu', 'Terlambat_1Sem', 'Terlambat_Lebih']

# ── FORWARD CHAINING (rule-based, sesuai Tugas 3 hal. 2) ───────────────────
def forward_chaining(ipk, sks_lulus, semester, sisa_matkul):
    """
    Cek aturan pasti satu per satu.
    Kembalikan (label, alasan) jika ada rule cocok, atau (None, None).
    """
    if ipk < 2.0:
        return "risiko_tinggi", "IPK di bawah 2.0 — mendekati batas Drop Out"
    if semester >= 14:
        return "risiko_tinggi", "Semester sudah mencapai batas maksimal studi (14)"
    if sks_lulus >= 144 and sisa_matkul == 0:
        return "siap_lulus", "Semua syarat kelulusan terpenuhi (SKS ≥ 144 & tidak ada matkul wajib tersisa)"
    if sks_lulus >= 144 and sisa_matkul > 0:
        return "waspada", "SKS sudah cukup (≥ 144), tapi masih ada matkul wajib yang belum tuntas"
    if sks_lulus < 40 and semester >= 5:
        return "risiko_tinggi", "Progress SKS jauh di bawah ideal untuk semester ini (< 40 SKS di semester ≥ 5)"
    return None, None


# ── DECISION TREE via model.pkl ─────────────────────────────────────────────
def run_decision_tree(semester, ipk, sks_lulus, sks_diambil,
                      sisa_matkul, tugas_tertunda, skor_psiko,
                      tidur_jam, motivasi_num):
    """
    Jalankan model scikit-learn yang sudah dilatih dari 100 data dummy.
    Kembalikan (label, probabilitas_persen, dict_feature_importance).
    """
    # Susun vektor fitur SESUAI urutan FITUR dari model.pkl
    X = pd.DataFrame([[semester, ipk, sks_lulus, sks_diambil,
                        sisa_matkul, tugas_tertunda, skor_psiko,
                        tidur_jam, motivasi_num]], columns=FITUR)

    label_pred = MODEL.predict(X)[0]                       # string label
    proba      = MODEL.predict_proba(X)[0]                 # array probabilities
    prob_pct   = int(round(max(proba) * 100))              # persentase keyakinan

    # Faktor dominan dari feature_importances_ model
    importances = MODEL.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:4]        # 4 fitur terpenting
    top_factors = {FITUR[i]: round(importances[i], 3) for i in top_idx if importances[i] > 0}

    return label_pred, prob_pct, top_factors


# ── KONVERSI LABEL MODEL → INTERNAL KEY ─────────────────────────────────────
LABEL_MAP = {
    "Tepat_Waktu":     "tepat_waktu",
    "Terlambat_1Sem":  "terlambat_1sem",
    "Terlambat_Lebih": "terlambat_lebih",
}

# ── TEMPLATE OUTPUT ──────────────────────────────────────────────────────────
TEMPLATES = {
    "tepat_waktu": {
        "status": "Aman",
        "color": "green",
        "emoji": "✅",
        "judul": "Kondisimu terlihat baik!",
        "pesan": "Berdasarkan pola 100 data mahasiswa, kamu berada di jalur yang tepat. Pertahankan ritme belajar sekarang.",
        "rekomendasi": [
            "Pertahankan konsistensi mengerjakan tugas tepat waktu",
            "Jaga kualitas tidur minimal 7 jam per malam",
            "Lakukan review progress akademik tiap akhir semester",
        ],
    },
    "waspada": {
        "status": "Waspada",
        "color": "amber",
        "emoji": "⚠️",
        "judul": "Ada beberapa hal yang perlu diperhatikan",
        "pesan": "Kondisimu saat ini belum kritis, tapi ada sinyal yang perlu kamu sadari sebelum makin berat.",
        "rekomendasi": [
            "Kurangi tugas yang menumpuk — selesaikan minimal 1–2 tugas per hari",
            "Coba cerita ke teman atau dosen kalau merasa kewalahan",
            "Prioritaskan matkul wajib yang masih tersisa",
        ],
    },
    "terlambat_1sem": {
        "status": "Risiko Tinggi",
        "color": "coral",
        "emoji": "🔴",
        "judul": "Risiko keterlambatan 1 semester terdeteksi",
        "pesan": "Jika pola ini berlanjut 5–7 hari ke depan, ada potensi keterlambatan kelulusan hingga 1 semester.",
        "rekomendasi": [
            "Selesaikan minimal 3 tugas per hari selama 3 hari ke depan",
            "Konsultasi dengan dosen wali untuk menyusun ulang rencana studi",
            "Jika rekomendasi diikuti, estimasi risiko bisa turun ke ~45%",
        ],
    },
    "terlambat_lebih": {
        "status": "Risiko Sangat Tinggi",
        "color": "red",
        "emoji": "🚨",
        "judul": "Risiko keterlambatan lebih dari 1 semester",
        "pesan": "Kombinasi beban psikologis tinggi, tugas yang menumpuk, dan banyaknya matkul wajib sisa adalah sinyal serius.",
        "rekomendasi": [
            "Segera konsultasi dengan dosen wali atau bagian akademik",
            "Pertimbangkan untuk mengurangi beban SKS semester depan",
            "Prioritaskan kesehatan mental — hubungi konselor kampus jika perlu",
        ],
    },
    "risiko_tinggi": {
        "status": "Risiko Tinggi",
        "color": "coral",
        "emoji": "🔴",
        "judul": "Kondisi akademik memerlukan perhatian segera",
        "pesan": "Sistem mendeteksi kondisi akademik yang memerlukan tindakan segera.",
        "rekomendasi": [
            "Segera temui dosen wali atau bagian akademik",
            "Buat rencana studi ulang dengan bimbingan dosen",
            "Jangan tunda — semakin cepat ditangani semakin baik",
        ],
    },
    "siap_lulus": {
        "status": "Siap Lulus",
        "color": "green",
        "emoji": "🎓",
        "judul": "Selamat! Kamu hampir di garis finish",
        "pesan": "Semua syarat akademik sudah terpenuhi. Pastikan tidak ada administrasi yang terlewat.",
        "rekomendasi": [
            "Cek kelengkapan berkas wisuda",
            "Pastikan tidak ada nilai yang belum keluar",
            "Selesaikan tugas akhir / skripsi jika belum",
        ],
    },
}


def generate_output(label_key, prob, faktor_list, top_factors=None):
    result = dict(TEMPLATES.get(label_key, TEMPLATES["waspada"]))
    result["prob"]         = prob
    result["faktor"]       = faktor_list
    result["top_factors"]  = top_factors or {}
    result["metode"]       = "Forward Chaining" if label_key in ("risiko_tinggi", "siap_lulus", "waspada") else "Decision Tree (ML)"
    return result


# ── HELPER: parse input chatbot → nilai numerik ──────────────────────────────
def parse_value(key, raw):
    if key == "kualitas_tidur":
        return {"< 5 jam": 4.0, "5–7 jam": 6.0, "> 7 jam": 7.5}.get(raw, 6.0)
    if key == "tugas_tertunda":
        return {"0": 0.0, "1–2": 1.5, "3–5": 4.0, "> 5": 6.5}.get(raw, 0.0)
    if key == "kondisi_emoji":
        # Dipakai untuk menghitung skor psikologis sebagian
        return {"😊 Baik, siap gas!": 0, "😐 Biasa aja": 1, "😮‍💨 Agak lelah": 2, "😭 Kewalahan banget": 2}.get(raw, 1)
    if key == "eksternal":
        return {"Tidak ada": 0, "Sedikit": 1, "Banyak banget": 2}.get(raw, 0)
    if key == "motivasi":
        return {"Rendah": 0, "Sedang": 1, "Tinggi": 2}.get(raw, 1)
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def hitung_skor_psikologis(kondisi_raw, tidur_jam, eksternal_val, motivasi_raw):
    """
    Hitung skor psikologis (0–12) dari proxy input harian.
    Sesuai kuesioner 6 item skor 0–2 dari Tugas 2.
    """
    skor = 0
    # Item 1–2: kondisi hari ini (mewakili perasaan & kewalahan) → 0–4
    kondisi_item = {"😊 Baik, siap gas!": 0, "😐 Biasa aja": 2, "😮‍💨 Agak lelah": 3, "😭 Kewalahan banget": 4}.get(kondisi_raw, 2)
    skor += min(kondisi_item, 4)
    # Item 3: kualitas tidur → 0–2
    if tidur_jam < 5.0:
        skor += 2
    elif tidur_jam < 7.0:
        skor += 1
    # Item 4: motivasi (invers) → 0–2
    motivasi_skor = {"Rendah": 2, "Sedang": 1, "Tinggi": 0}.get(motivasi_raw, 1)
    skor += motivasi_skor
    # Item 5–6: beban eksternal → 0–2
    skor += min(int(eksternal_val), 2)
    return min(skor, 12)


# ── STATE MACHINE CHATBOT ────────────────────────────────────────────────────
FLOW = {
    "start": {
        "msg": (
            "Halo! Saya **CAKRA** — Chatbot Analisis Kondisi Risiko Akademik. 🎓\n\n"
            "Saya akan bantu kamu memantau risiko keterlambatan kelulusan menggunakan "
            "Machine Learning yang sudah dilatih dari data 100 mahasiswa.\n\n"
            "Kita mulai dengan data akademikmu ya!\n\n"
            "Kamu sekarang semester berapa?"
        ),
        "input_type": "number",
        "placeholder": "Contoh: 7",
        "key": "semester",
        "next": "tanya_ipk",
    },
    "tanya_ipk": {
        "msg": "Oke, semester {semester}. Berapa IPK terakhirmu?",
        "input_type": "number",
        "placeholder": "Contoh: 3.25",
        "key": "ipk",
        "next": "tanya_sks_lulus",
    },
    "tanya_sks_lulus": {
        "msg": "Sudah berapa SKS yang berhasil kamu lulus?",
        "input_type": "number",
        "placeholder": "Contoh: 95",
        "key": "sks_lulus",
        "next": "tanya_sks_diambil",
    },
    "tanya_sks_diambil": {
        "msg": "Semester ini kamu mengambil berapa SKS?",
        "input_type": "number",
        "placeholder": "Contoh: 20",
        "key": "sks_diambil",
        "next": "tanya_sisa_matkul",
    },
    "tanya_sisa_matkul": {
        "msg": "Ada berapa matkul wajib yang belum kamu selesaikan?",
        "input_type": "number",
        "placeholder": "Contoh: 5",
        "key": "sisa_matkul",
        "next": "tanya_harian_intro",
    },
    "tanya_harian_intro": {
        "msg": (
            "Mantap! Data akademikmu sudah tercatat. 📚\n\n"
            "Sekarang kita cek kondisi harianmu — 4 pertanyaan singkat saja.\n\n"
            "Semalam tidur berapa jam?"
        ),
        "input_type": "choices",
        "choices": ["< 5 jam", "5–7 jam", "> 7 jam"],
        "key": "kualitas_tidur",
        "next": "tanya_tugas",
    },
    "tanya_tugas": {
        "msg": "Hari ini ada berapa tugas yang belum selesai?",
        "input_type": "choices",
        "choices": ["0", "1–2", "3–5", "> 5"],
        "key": "tugas_tertunda",
        "next": "tanya_kondisi",
    },
    "tanya_kondisi": {
        "msg": "Kondisi kamu hari ini gimana?",
        "input_type": "choices",
        "choices": ["😊 Baik, siap gas!", "😐 Biasa aja", "😮‍💨 Agak lelah", "😭 Kewalahan banget"],
        "key": "kondisi_emoji",
        "next": "tanya_motivasi",
    },
    "tanya_motivasi": {
        "msg": "Seberapa tinggi motivasi belajarmu saat ini?",
        "input_type": "choices",
        "choices": ["Rendah", "Sedang", "Tinggi"],
        "key": "motivasi",
        "next": "tanya_eksternal",
    },
    "tanya_eksternal": {
        "msg": "Ada tanggung jawab di luar kuliah yang menyita waktu belajarmu minggu ini? (kerja, organisasi, dll)",
        "input_type": "choices",
        "choices": ["Tidak ada", "Sedikit", "Banyak banget"],
        "key": "eksternal",
        "next": "hasil",
    },
}

<<<<<<< Updated upstream
def parse_value(key, raw):
    """Convert user input to numeric values for inference"""
    if key == "kualitas_tidur":
        mapping = {"< 5 jam": 4.5, "5–7 jam": 6.0, "> 7 jam": 7.5}
        return mapping.get(raw, 6.0)
    if key == "tugas_tertunda":
        mapping = {"0": 0, "1–2": 1.5, "3–5": 4.0, "> 5": 6.5}
        return mapping.get(raw, 0)
    if key == "kondisi":
        mapping = {
            "😊 Baik, siap gas!": 2,
            "😐 Biasa aja": 4,
            "😮‍💨 Agak lelah": 7,
            "😭 Kewalahan banget": 10
        }
        return mapping.get(raw, 4)
    if key == "eksternal":
        mapping = {"Tidak ada": 0, "Sedikit": 1, "Banyak banget": 2}
        return mapping.get(raw, 0)
    try:
        return float(raw)
    except:
        return 0

def get_skor_psikologis(kondisi_val, tidur_val, eksternal_val):
    """
    Kalkulasi skor psikologis harian (Skala 0-12).
    Diselaraskan agar total skor maksimal 12 sesuai dokumen rancangan.
    """
    skor = 0
    
    # 1. Faktor Kondisi (Bobot proporsional max 8 poin)
    # Input asli dari fungsi parse_value: 2 (Baik), 4 (Biasa), 7 (Lelah), 10 (Kewalahan)
    if kondisi_val >= 10:
        skor += 8    # Kewalahan banget = 8 poin
    elif kondisi_val >= 7:
        skor += 5    # Agak lelah = 5 poin
    elif kondisi_val >= 4:
        skor += 2    # Biasa aja = 2 poin
    else:
        skor += 0    # Baik, siap gas = 0 poin
        
    # 2. Faktor Kualitas Tidur (Bobot max 2 poin)
    if tidur_val < 5.0:
        skor += 2    # Kurang dari 5 jam = 2 poin
    elif tidur_val < 7.0:
        skor += 1    # 5-7 jam = 1 poin
        
    # 3. Faktor Beban Eksternal (Bobot max 2 poin)
    # Input asli dari parse_value sudah berupa angka 0, 1, atau 2
    skor += eksternal_val
    
    # Pastikan output akhir maksimal 12 sesuai format kuesioner (6 item skor 0-2)
    return min(skor, 12)
=======
>>>>>>> Stashed changes

def run_inference(data):
    """
    Pipeline lengkap:
      1. Forward Chaining (cek aturan pasti)
      2. Decision Tree via model.pkl (jika tidak ada rule yang cocok)
    """
    ipk         = float(data.get("ipk", 3.0))
    sks_lulus   = float(data.get("sks_lulus", 0))
    sks_diambil = float(data.get("sks_diambil", 20))
    sisa_matkul = int(float(data.get("sisa_matkul", 0)))
    semester    = int(float(data.get("semester", 4)))

    tidur_jam   = parse_value("kualitas_tidur", data.get("kualitas_tidur", "5–7 jam"))
    tugas_num   = parse_value("tugas_tertunda",  data.get("tugas_tertunda", "0"))
    eksternal   = parse_value("eksternal",        data.get("eksternal", "Tidak ada"))
    motivasi_raw = data.get("motivasi", "Sedang")
    motivasi_num = parse_value("motivasi", motivasi_raw)

    kondisi_raw  = data.get("kondisi_emoji", "😐 Biasa aja")
    skor_psiko   = hitung_skor_psikologis(kondisi_raw, tidur_jam, eksternal, motivasi_raw)

    # ── Tahap 1: Forward Chaining ──────────────────────────────────────────
    fc_label, fc_reason = forward_chaining(ipk, sks_lulus, semester, sisa_matkul)
    if fc_label:
        faktor = [
            fc_reason,
            f"IPK: {ipk}",
            f"Semester ke-{semester}",
            f"SKS Lulus: {sks_lulus}",
        ]
        return generate_output(fc_label, 95, faktor)

    # ── Tahap 2: Decision Tree (model.pkl dari dataset.csv) ────────────────
    dt_label_raw, dt_prob, top_factors = run_decision_tree(
        semester, ipk, sks_lulus, sks_diambil,
        sisa_matkul, tugas_num, skor_psiko,
        tidur_jam, motivasi_num
    )
    dt_label = LABEL_MAP.get(dt_label_raw, "waspada")

    # Susun penjelasan faktor dominan berdasarkan feature_importances_ model
    faktor = []
    nama_fitur = {
        "IPK":                f"IPK {ipk} (fitur paling berpengaruh dalam model)",
        "Skor_Psikologis_Avg": f"Skor psikologis {skor_psiko}/12 — {'kritis' if skor_psiko >= 8 else 'waspada' if skor_psiko >= 5 else 'aman'}",
        "Tugas_Tertunda_Avg":  f"Tugas tertunda: {tugas_num} tugas",
        "Sisa_Matkul_Wajib":   f"{sisa_matkul} matkul wajib masih tersisa",
        "Semester":            f"Semester ke-{semester}",
        "SKS_Lulus":           f"SKS lulus: {sks_lulus}",
        "Tidur_Avg_Jam":       f"Rata-rata tidur {tidur_jam} jam/malam",
        "Motivasi":            f"Motivasi: {motivasi_raw}",
        "SKS_Diambil":         f"SKS diambil semester ini: {sks_diambil}",
    }
    for fname in top_factors:
        if top_factors[fname] > 0 and fname in nama_fitur:
            faktor.append(nama_fitur[fname])
    if not faktor:
        faktor.append("Semua indikator dalam batas normal berdasarkan pola data")

    return generate_output(dt_label, dt_prob, faktor, top_factors)


# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start():
    session.clear()
    session["state"] = "start"
    session["data"]  = {}
    step = FLOW["start"]
    return jsonify({
        "msg":        step["msg"],
        "input_type": step["input_type"],
        "placeholder": step.get("placeholder", ""),
        "choices":    step.get("choices", []),
        "state":      "start",
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    body        = request.get_json()
    user_input  = body.get("message", "").strip()
    curr_state  = session.get("state", "start")
    data        = session.get("data", {})

    if curr_state not in FLOW:
        return jsonify({"msg": "Sesi tidak valid. Silakan mulai ulang.", "input_type": "restart"})

    step = FLOW[curr_state]
    data[step["key"]] = user_input
    session["data"]   = data

    next_state = step["next"]

    if next_state == "hasil":
        result = run_inference(dict(data))
        session["state"] = "done"
        return jsonify({
            "msg":        "Analisis selesai!",
            "input_type": "result",
            "result":     result,
        })

    session["state"] = next_state
    next_step = FLOW[next_state]
    msg = next_step["msg"]
    for k, v in data.items():
        msg = msg.replace("{" + k + "}", str(v))

    return jsonify({
        "msg":         msg,
        "input_type":  next_step["input_type"],
        "placeholder": next_step.get("placeholder", ""),
        "choices":     next_step.get("choices", []),
        "state":       next_state,
    })


@app.route("/api/restart", methods=["POST"])
def restart():
    session.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
