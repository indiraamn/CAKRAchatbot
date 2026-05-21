from flask import Flask, render_template, request, jsonify, session
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = ""

# DECISION TREE (simulasi max_depth=4)
def decision_tree(ipk, sks_lulus, sks_diambil, sisa_matkul, semester,
                  tugas_tertunda, skor_psikologis, kualitas_tidur):
    # Level 1: skor psikologis
    if skor_psikologis <= 6.0:
        # Level 2 kiri: tugas tertunda
        if tugas_tertunda <= 2.0:
            # Level 3: IPK
            if ipk <= 3.0:
                return "waspada", 52
            else:
                return "tepat_waktu", 88
        else:
            # Level 3: semester
            if semester <= 8:
                return "waspada", 58
            else:
                return "terlambat_1sem", 67
    else:
        # Level 2 kanan: tugas tertunda
        if tugas_tertunda <= 4.0:
            # Level 3: IPK
            if ipk <= 3.0:
                return "terlambat_1sem", 71
            else:
                return "waspada", 60
        else:
            # Level 3: matkul wajib sisa
            if sisa_matkul <= 3:
                return "terlambat_1sem", 74
            else:
                return "terlambat_lebih", 82

# FORWARD CHAINING (rule-based) 
def forward_chaining(ipk, sks_lulus, semester, sisa_matkul):
    if ipk < 2.0:
        return "risiko_tinggi", "IPK di bawah 2.0 — mendekati batas Drop Out"
    if semester >= 14:
        return "risiko_tinggi", "Semester sudah mencapai batas maksimal studi"
    if sks_lulus >= 144 and sisa_matkul == 0:
        return "siap_lulus", "Semua syarat kelulusan terpenuhi"
    if sks_lulus >= 144 and sisa_matkul > 0:
        return "waspada", "SKS sudah cukup, tapi masih ada matkul wajib yang belum tuntas"
    if sks_lulus < 40 and semester >= 5:
        return "risiko_tinggi", "Progress SKS jauh di bawah ideal untuk semester ini"
    return None, None

# MEMBUAT OUTPUT 
def generate_output(label, prob, faktor_utama):
    templates = {
        "tepat_waktu": {
            "status": "Aman",
            "color": "green",
            "emoji": "✅",
            "judul": "Kondisimu terlihat baik!",
            "pesan": "Berdasarkan pola yang ada, kamu berada di jalur yang tepat. Pertahankan ritme belajar sekarang.",
            "rekomendasi": [
                "Pertahankan konsistensi mengerjakan tugas tepat waktu",
                "Jaga kualitas tidur minimal 7 jam per malam",
                "Lakukan review progress akademik tiap akhir semester"
            ]
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
                "Prioritaskan matkul wajib yang masih tersisa"
            ]
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
                "Jika rekomendasi diikuti, estimasi risiko bisa turun ke ~45%"
            ]
        },
        "terlambat_lebih": {
            "status": "Risiko Sangat Tinggi",
            "color": "red",
            "emoji": "🚨",
            "judul": "Risiko keterlambatan lebih dari 1 semester",
            "pesan": "Kombinasi beban psikologis tinggi, tugas yang menumpuk, dan banyaknya matkul wajib sisa adalah sinyal serius.",
            "rekomendasi": [
                "Segera konsultasi dengan dosen wali atau bagian akademik",
                "Pertimbangkan untuk mengambil cuti atau mengurangi beban SKS semester depan",
                "Prioritaskan kesehatan mental — hubungi konselor kampus jika perlu"
            ]
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
                "Jangan tunda — semakin cepat ditangani semakin baik"
            ]
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
                "Selesaikan tugas akhir / skripsi jika belum"
            ]
        }
    }
    result = templates.get(label, templates["waspada"])
    result["prob"] = prob
    result["faktor"] = faktor_utama
    return result

# MEMBUAT STATE MACHINE UNTUK CHATBOT
FLOW = {
    "start": {
        "msg": "Halo! Saya **CAKRA** — Chatbot Analisis Kondisi Risiko Akademik.\n\nSaya akan bantu kamu memantau risiko keterlambatan kelulusan. Kita mulai dengan beberapa pertanyaan singkat ya! 😊\n\nPertama, kamu sekarang semester berapa?",
        "input_type": "number",
        "placeholder": "Contoh: 7",
        "key": "semester",
        "next": "tanya_ipk"
    },
    "tanya_ipk": {
        "msg": "Oke, semester {semester}. Berapa IPK terakhirmu?",
        "input_type": "number",
        "placeholder": "Contoh: 3.25",
        "key": "ipk",
        "next": "tanya_sks_lulus"
    },
    "tanya_sks_lulus": {
        "msg": "Sudah berapa SKS yang berhasil kamu lulus?",
        "input_type": "number",
        "placeholder": "Contoh: 95",
        "key": "sks_lulus",
        "next": "tanya_sks_diambil"
    },
    "tanya_sks_diambil": {
        "msg": "Semester ini kamu mengambil berapa SKS?",
        "input_type": "number",
        "placeholder": "Contoh: 20",
        "key": "sks_diambil",
        "next": "tanya_sisa_matkul"
    },
    "tanya_sisa_matkul": {
        "msg": "Ada berapa matkul wajib yang belum kamu selesaikan?",
        "input_type": "number",
        "placeholder": "Contoh: 5",
        "key": "sisa_matkul",
        "next": "tanya_harian_intro"
    },
    "tanya_harian_intro": {
        "msg": "Mantap! Data akademikmu sudah tercatat. 📚\n\nSekarang kita cek kondisi harianmu. Pertanyaannya singkat, 3–4 pertanyaan saja.\n\nSemalam tidur berapa jam?",
        "input_type": "choices",
        "choices": ["< 5 jam", "5–7 jam", "> 7 jam"],
        "key": "kualitas_tidur",
        "next": "tanya_tugas"
    },
    "tanya_tugas": {
        "msg": "Hari ini ada berapa tugas yang belum selesai?",
        "input_type": "choices",
        "choices": ["0", "1–2", "3–5", "> 5"],
        "key": "tugas_tertunda",
        "next": "tanya_kondisi"
    },
    "tanya_kondisi": {
        "msg": "Kondisi kamu hari ini gimana?",
        "input_type": "choices",
        "choices": ["😊 Baik, siap gas!", "😐 Biasa aja", "😮‍💨 Agak lelah", "😭 Kewalahan banget"],
        "key": "kondisi",
        "next": "tanya_eksternal"
    },
    "tanya_eksternal": {
        "msg": "Ada tanggung jawab luar kuliah yang menyita waktu minggu ini? (kerja, organisasi, dll)",
        "input_type": "choices",
        "choices": ["Tidak ada", "Sedikit", "Banyak banget"],
        "key": "eksternal",
        "next": "hasil"
    }
}

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
    """Combine daily inputs into psychological score 0-12"""
    skor = kondisi_val
    if tidur_val < 5:
        skor += 2
    elif tidur_val < 6:
        skor += 1
    skor += eksternal_val
    return min(skor, 12)

def run_inference(data):
    """Full inference pipeline: forward chaining → decision tree"""
    ipk          = float(data.get("ipk", 3.0))
    sks_lulus    = float(data.get("sks_lulus", 0))
    sks_diambil  = float(data.get("sks_diambil", 20))
    sisa_matkul  = int(data.get("sisa_matkul", 0))
    semester     = int(data.get("semester", 4))
    tugas_raw    = parse_value("tugas_tertunda", data.get("tugas_tertunda", "0"))
    kondisi_raw  = parse_value("kondisi", data.get("kondisi", "😐 Biasa aja"))
    tidur_raw    = parse_value("kualitas_tidur", data.get("kualitas_tidur", "5–7 jam"))
    eksternal    = parse_value("eksternal", data.get("eksternal", "Tidak ada"))

    skor_psiko   = get_skor_psikologis(kondisi_raw, tidur_raw, eksternal)

    # Tahap 1: Forward Chaining
    fc_label, fc_reason = forward_chaining(ipk, sks_lulus, semester, sisa_matkul)
    if fc_label:
        faktor = [fc_reason, f"IPK: {ipk}", f"Semester: {semester}", f"SKS Lulus: {sks_lulus}"]
        return generate_output(fc_label, 95, faktor)

    # Tahap 2: Decision Tree
    dt_label, dt_prob = decision_tree(
        ipk, sks_lulus, sks_diambil, sisa_matkul,
        semester, tugas_raw, skor_psiko, tidur_raw
    )

    faktor = []
    if skor_psiko >= 8:
        faktor.append(f"Skor psikologis tinggi ({skor_psiko}/12 — kategori kritis)")
    elif skor_psiko >= 5:
        faktor.append(f"Skor psikologis sedang ({skor_psiko}/12 — perlu dipantau)")
    if tugas_raw >= 4:
        faktor.append(f"Tugas tertunda cukup banyak")
    if ipk <= 3.0:
        faktor.append(f"IPK {ipk} perlu ditingkatkan")
    if sisa_matkul > 3:
        faktor.append(f"{sisa_matkul} matkul wajib masih harus diselesaikan")
    if tidur_raw < 5:
        faktor.append("Kualitas tidur kurang dari 5 jam")
    if not faktor:
        faktor.append("Semua indikator dalam batas normal")

    return generate_output(dt_label, dt_prob, faktor)

# app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start():
    session.clear()
    session["state"] = "start"
    session["data"] = {}
    session["history"] = []
    step = FLOW["start"]
    return jsonify({
        "msg": step["msg"],
        "input_type": step["input_type"],
        "placeholder": step.get("placeholder", ""),
        "choices": step.get("choices", []),
        "state": "start"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json()
    user_input = body.get("message", "").strip()
    current_state = session.get("state", "start")
    data = session.get("data", {})

    if current_state not in FLOW:
        return jsonify({"msg": "Sesi tidak valid. Silakan mulai ulang.", "input_type": "restart"})

    step = FLOW[current_state]
    key  = step["key"]
    data[key] = user_input
    session["data"] = data

    next_state = step["next"]

    # Sudah waktunya hitung hasil
    if next_state == "hasil":
        # Simpan jawaban terakhir
        final_data = dict(data)
        result = run_inference(final_data)
        session["state"] = "done"
        return jsonify({
            "msg": "Analisis selesai!",
            "input_type": "result",
            "result": result
        })

    next_step = FLOW[next_state]
    session["state"] = next_state

    # Format msg dengan data yang sudah ada
    msg = next_step["msg"]
    for k, v in data.items():
        msg = msg.replace("{" + k + "}", str(v))

    return jsonify({
        "msg": msg,
        "input_type": next_step["input_type"],
        "placeholder": next_step.get("placeholder", ""),
        "choices": next_step.get("choices", []),
        "state": next_state
    })

@app.route("/api/restart", methods=["POST"])
def restart():
    session.clear()
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
