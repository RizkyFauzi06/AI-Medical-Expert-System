import json
import re
import math
import difflib
from pathlib import Path
from collections import defaultdict

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = Path(__file__).parent / "Penyakit.json" 

NEGATION_WORDS = ["tidak", "tdk", "nggak", "enggak", "engga", "gak", "ga", "bukan", "tanpa", "belum"]
POSITIVE_SHORT_REPLIES = {"ya", "iya", "iyah", "iy", "yes", "y", "betul", "benar", "ada", "yoi", "yap"}
NEGATIVE_SHORT_REPLIES = {"tidak", "tdk", "nggak", "enggak", "engga", "gak", "ga", "no", "bukan", "belum"}

ML_CONFIDENCE_THRESHOLD = 0.55  
DEFAULT_LIKELIHOOD_IF_UNLISTED = 0.15  


SYNONYMS = {
    "demam_tinggi": ["demam", "panas", "panas tinggi", "meriang", "badan panas", "badan panas banget"],
    "demam_ringan": ["demam", "meriang", "badan hangat", "sumeng"],
    "demam_menggigil": ["demam", "menggigil", "meriang", "dingin panas"],
    "bintik_merah_kulit": ["bintik merah", "bercak merah", "bintik di kulit"],
    "nyeri_sendi": ["nyeri sendi", "pegal sendi", "linu", "encok"],
    "nyeri_sendi_hebat": ["sendi sakit banget", "linu parah", "nyeri sendi hebat"],
    "mual_muntah": ["mual", "muntah", "eneg", "pengen muntah"],
    "sakit_perut": ["sakit perut", "perut sakit", "mules", "nyeri perut"],
    "lidah_putih": ["lidah putih", "lidah berselaput"],
    "diare": ["diare", "mencret"],
    "pilek_hidung_tersumbat": ["pilek", "hidung tersumbat", "hidung mampet", "meler"],
    "sakit_tenggorokan": ["sakit tenggorokan", "tenggorokan sakit"],
    "bersin_bersin": ["bersin", "sering bersin"],
    "batuk_kering": ["batuk kering", "batuk tanpa dahak"],
    "hilang_penciuman_rasa": ["hilang penciuman", "gak bisa cium bau", "gak bisa ngerasa rasa"],
    "sesak_napas": ["sesak napas", "susah bernapas", "napas berat", "sesak nafas"],
    "keringat_berlebih": ["keringat berlebih", "banyak keringat", "keringatan"],
    "sakit_kepala": ["sakit kepala", "pusing", "kepala pusing", "kepala sakit"],
    "nyeri_otot": ["nyeri otot", "pegal", "pegal-pegal", "pegal pegal", "otot sakit", "badan pegal"],
    "nyeri_ulu_hati": ["nyeri ulu hati", "perih lambung", "maag"],
    "perut_kembung": ["perut kembung", "kembung", "begah"],
    "cepat_kenyang": ["cepat kenyang", "gampang kenyang"],
    "napas_berbunyi_mengi": ["napas berbunyi", "mengi", "napas ngik-ngik"],
    "batuk_malam_hari": ["batuk malam", "batuk saat malam"],
    "dada_terasa_berat": ["dada berat", "dada sesak", "dada terasa berat"],
    "batuk_lebih_3_minggu": ["batuk lama", "batuk berkepanjangan", "batuk kronis"],
    "keringat_malam": ["keringat malam", "keringat dingin malam"],
    "berat_badan_turun": ["berat badan turun", "badan kurus", "berat badan menurun"],
    "batuk_berdarah": ["batuk darah", "batuk berdarah"],
    "BAB_cair_berulang": ["BAB cair", "diare terus", "mencret berulang"],
    "kram_perut": ["kram perut", "perut kram", "perut melilit"],
    "tanda_dehidrasi": ["dehidrasi", "bibir kering", "mulut kering"],
    "tengkuk_terasa_berat": ["tengkuk berat", "leher belakang berat", "tengkuk pegal"],
    "penglihatan_kabur": ["penglihatan kabur", "pandangan kabur", "mata buram"],
    "jantung_berdebar": ["jantung berdebar", "deg-degan", "jantung deg degan"],
    "sering_haus": ["sering haus", "gampang haus", "haus terus"],
    "sering_buang_air_kecil": ["sering pipis", "sering kencing"],
    "luka_sulit_sembuh": ["luka susah sembuh", "luka lama sembuh"],
    "cepat_lelah": ["cepat lelah", "gampang capek", "mudah lelah", "lemas"],
    "sakit_kepala_sebelah": ["sakit kepala sebelah", "migrain", "kepala sebelah nyut-nyutan"],
    "sensitif_cahaya_suara": ["silau", "sensitif cahaya", "sensitif suara"],
    "pandangan_berkunang": ["pandangan berkunang", "mata berkunang-kunang"],
    "kulit_pucat": ["kulit pucat", "wajah pucat"],
    "sering_pusing": ["sering pusing", "gampang pusing"],
    "mata_berkunang": ["mata berkunang", "pandangan gelap sesaat"],
    "sakit_menelan": ["sakit menelan", "susah nelan", "nyeri saat menelan"],
    "tenggorokan_kering": ["tenggorokan kering", "leher kering"],
    "suara_serak": ["suara serak", "suara hilang"],
    "ruam_merah_berair": ["ruam berair", "bintil berair", "cacar air"],
    "gatal_seluruh_tubuh": ["gatal seluruh badan", "badan gatal semua"],
    "kelelahan": ["lelah", "capek"],
    "ruam_merah_menyebar": ["ruam menyebar", "bercak merah menyebar"],
    "mata_merah_berair": ["mata merah", "mata berair"],
    "bercak_putih_mulut": ["bercak putih di mulut", "sariawan putih"],
    "batuk_berdahak_kental": ["batuk berdahak", "dahak kental"],
    "nyeri_dada_saat_napas": ["nyeri dada saat napas", "dada sakit kalau napas"],
    "sendi_bengkak_kemerahan": ["sendi bengkak", "sendi merah bengkak"],
    "terasa_panas_di_sendi": ["sendi panas", "sendi terasa panas"],
    "nyeri_saat_bergerak": ["sakit saat gerak", "nyeri saat bergerak"],
    "sering_kesemutan": ["kesemutan", "sering kesemutan"],
    "nyeri_dada_kiri": ["nyeri dada kiri", "dada kiri sakit"],
    "nyeri_perut_kanan_bawah": ["nyeri perut kanan bawah", "perut kanan bawah sakit"],
    "hilang_nafsu_makan": ["gak nafsu makan", "hilang nafsu makan", "males makan"],
}


@st.cache_data 
def load_penyakit(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["penyakit"]


def build_symptom_index(penyakit_list):
    index = {}
    for penyakit in penyakit_list:
        for key, info in penyakit["gejala"].items():
            label = key.replace("_", " ")
            if key not in index:
                keywords = {label.lower()}
                keywords.update(k.lower() for k in SYNONYMS.get(key, []))
                index[key] = {"label": label, "keywords": keywords, "kritis": False}
            if info.get("kritis"):
                index[key]["kritis"] = True
    return index


def _has_negation_before(text_low: str, match_pos: int, window: int = 18) -> bool:
    start = max(0, match_pos - window)
    snippet = text_low[start:match_pos]
    tokens = re.findall(r"\w+", snippet)
    return any(tok in NEGATION_WORDS for tok in tokens[-3:])


def extract_gejala_rule_based(text: str, symptom_index: dict):
    text_low = text.lower()
    words = re.findall(r"[a-z]+", text_low)
    confirmed, rejected = set(), set()

    for key, info in symptom_index.items():
        found_pos = None
        for kw in info["keywords"]:
            m = re.search(r"\b" + re.escape(kw) + r"\b", text_low)
            if m:
                found_pos = m.start()
                break

        if found_pos is None:
            for kw in info["keywords"]:
                if " " in kw or len(kw) < 6:
                    continue
                close = difflib.get_close_matches(kw, words, n=1, cutoff=0.93)
                if close:
                    m = re.search(r"\b" + re.escape(close[0]) + r"\b", text_low)
                    if m:
                        found_pos = m.start()
                    break

        if found_pos is not None:
            if _has_negation_before(text_low, found_pos):
                rejected.add(key)
            else:
                confirmed.add(key)

    return confirmed, rejected


def interpret_short_answer(text: str):
    t = text.lower().strip().rstrip("!.,")
    if t in POSITIVE_SHORT_REPLIES:
        return True
    if t in NEGATIVE_SHORT_REPLIES:
        return False
    if any(t.startswith(w) for w in NEGATIVE_SHORT_REPLIES):
        return False
    if any(t.startswith(w) for w in POSITIVE_SHORT_REPLIES):
        return True
    return None


def build_ml_training_data(symptom_index: dict):
    phrase_to_keys = defaultdict(set)
    for key, info in symptom_index.items():
        for kw in info["keywords"]:
            phrase_to_keys[kw].add(key)
    texts = list(phrase_to_keys.keys())
    labels = [sorted(phrase_to_keys[t]) for t in texts]
    return texts, labels


@st.cache_resource
def train_ml_classifier(_symptom_index: dict):
    texts, labels = build_ml_training_data(_symptom_index)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)

    clf = OneVsRestClassifier(MultinomialNB())
    clf.fit(X, Y)

    return vectorizer, mlb, clf


def ml_predict_gejala(text: str, ml_bundle, symptom_index: dict, log: list):
    vectorizer, mlb, clf = ml_bundle
    text_low = text.lower()
    X_test = vectorizer.transform([text_low])
    probs = clf.predict_proba(X_test)[0]

    confirmed, rejected = set(), set()
    for label, prob in zip(mlb.classes_, probs):
        if prob < ML_CONFIDENCE_THRESHOLD:
            continue
        info = symptom_index[label]
        pos_found = None
        for kw in info["keywords"]:
            m = re.search(r"\b" + re.escape(kw) + r"\b", text_low)
            if m:
                pos_found = m.start()
                break

        if pos_found is not None and _has_negation_before(text_low, pos_found):
            rejected.add(label)
            status = "rejected (negasi)"
        else:
            confirmed.add(label)
            status = "confirmed"

        log.append({
            "source": "ML (Naive Bayes)",
            "gejala": label,
            "confidence": round(float(prob), 3),
            "status": status,
        })

    return confirmed, rejected


def extract_gejala_combined(text: str, symptom_index: dict, ml_bundle, log: list):
    rule_conf, rule_rej = extract_gejala_rule_based(text, symptom_index)
    for g in rule_conf:
        log.append({"source": "RULE (keyword/fuzzy)", "gejala": g, "confidence": 1.0, "status": "confirmed"})
    for g in rule_rej:
        log.append({"source": "RULE (keyword/fuzzy)", "gejala": g, "confidence": 1.0, "status": "rejected (negasi)"})

    ml_conf, ml_rej = ml_predict_gejala(text, ml_bundle, symptom_index, log)

    already_handled = rule_conf | rule_rej
    confirmed = rule_conf | (ml_conf - already_handled)
    rejected = rule_rej | (ml_rej - already_handled)
    return confirmed, rejected



def calculate_probabilities(penyakit_list, confirmed: set, rejected: set, log: list):
    raw = {}
    breakdown = {}

    for penyakit in penyakit_list:
        prob = penyakit["prior_prob"]
        steps = [f"Prior={penyakit['prior_prob']:.3f}"]
        gejala_def = penyakit["gejala"]

        for g in confirmed:
            if g in gejala_def:
                lk = gejala_def[g]["likelihood"]
                prob *= lk
                steps.append(f"× P({g}|D)={lk:.2f}")
            else:
                prob *= DEFAULT_LIKELIHOOD_IF_UNLISTED
                steps.append(f"× penalti({DEFAULT_LIKELIHOOD_IF_UNLISTED}) [{g} tak terdaftar]")

        for g in rejected:
            if g in gejala_def:
                lk = gejala_def[g]["likelihood"]
                prob *= (1 - lk)
                steps.append(f"× (1-P({g}|D))={1 - lk:.2f}")

        raw[penyakit["nama"]] = prob
        breakdown[penyakit["nama"]] = " ".join(steps)

    total = sum(raw.values())
    normalized = {nama: (val / total if total > 0 else 0.0) for nama, val in raw.items()}
    normalized = dict(sorted(normalized.items(), key=lambda x: -x[1]))

    for nama, val in list(normalized.items())[:3]:
        log.append({
            "phase": "PROBABILITY",
            "nama": nama,
            "formula": breakdown[nama],
            "posterior": round(val, 4),
        })

    return normalized


def entropy(prob_dist: dict) -> float:
    h = 0.0
    for p in prob_dist.values():
        if p > 0:
            h -= p * math.log2(p)
    return h



def evaluate_rules(penyakit_list, symptom_index: dict, confirmed: set, probabilities: dict, log: list):
    darurat_g = None
    for g in confirmed:
        if symptom_index.get(g, {}).get("kritis"):
            darurat_g = g
            break

    log.append({
        "phase": "KRR",
        "rule": "Rule 1 (Darurat)",
        "kondisi": "Ada gejala confirmed dengan kritis=true?",
        "hasil": f"YA -> gejala '{darurat_g}'" if darurat_g else "TIDAK",
    })
    if darurat_g:
        return ("darurat", darurat_g)

    top_nama = top_prob = threshold = None
    lolos = False
    if probabilities:
        top_nama = next(iter(probabilities))
        top_prob = probabilities[top_nama]
        top_penyakit = next(p for p in penyakit_list if p["nama"] == top_nama)
        threshold = top_penyakit.get("threshold", 0.8)
        lolos = top_prob >= threshold

    log.append({
        "phase": "KRR",
        "rule": "Rule 2 (Vonis)",
        "kondisi": f"P({top_nama}) = {top_prob:.3f} >= threshold ({threshold})" if top_nama else "-",
        "hasil": "YA -> Diagnosis" if lolos else "TIDAK",
    })
    if lolos:
        return ("diagnosis", top_nama, top_prob, top_penyakit.get("tindakan", "-"))

    log.append({
        "phase": "KRR",
        "rule": "Rule 3 (Inkuiri Lanjut)",
        "kondisi": "Rule 1 & Rule 2 sama-sama gagal",
        "hasil": "Lanjut ke Fase 4 (Search)",
    })
    return ("inkuiri", None)


def _normalize(d: dict) -> dict:
    total = sum(d.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in d.items()}


def find_next_question_entropy(penyakit_list, current_probabilities: dict,
                                confirmed: set, rejected: set,
                                symptom_index: dict, log: list):
    asked = confirmed | rejected
    current_H = entropy(current_probabilities)

    candidates = []
    for g, info in symptom_index.items():
        if g in asked:
            continue

        post_ya, post_tidak = {}, {}
        p_ya = 0.0
        for p in penyakit_list:
            prior = current_probabilities.get(p["nama"], 0.0)
            lk = p["gejala"].get(g, {}).get("likelihood", DEFAULT_LIKELIHOOD_IF_UNLISTED)
            post_ya[p["nama"]] = prior * lk
            post_tidak[p["nama"]] = prior * (1 - lk)
            p_ya += prior * lk
        p_tidak = 1 - p_ya

        H_ya = entropy(_normalize(post_ya)) if p_ya > 0 else 0.0
        H_tidak = entropy(_normalize(post_tidak)) if p_tidak > 0 else 0.0
        expected_H = p_ya * H_ya + p_tidak * H_tidak
        info_gain = current_H - expected_H

        candidates.append((g, info_gain))

    candidates.sort(key=lambda x: -x[1])

    log.append({
        "phase": "SEARCH",
        "algoritma": "Greedy Best-First Search (heuristik = Expected Information Gain)",
        "current_entropy": round(current_H, 4),
        "top_candidates": [(g, round(ig, 4)) for g, ig in candidates[:5]],
        "terpilih": candidates[0][0] if candidates else None,
    })

    return candidates[0][0] if candidates else None


def run_pipeline(user_text: str):
    ss = st.session_state
    symptom_index = ss.symptom_index
    penyakit_list = ss.penyakit_list
    ml_bundle = ss.ml_bundle
    log = []
    new_symptom_found = False

    if ss.pending_gejala is not None:
        short_ans = interpret_short_answer(user_text)
        if short_ans is True:
            ss.gejala_terkonfirmasi.add(ss.pending_gejala)
            log.append({"source": "INPUT (jawaban langsung)", "gejala": ss.pending_gejala,
                        "confidence": 1.0, "status": "confirmed"})
            new_symptom_found = True
        elif short_ans is False:
            ss.gejala_ditolak.add(ss.pending_gejala)
            log.append({"source": "INPUT (jawaban langsung)", "gejala": ss.pending_gejala,
                        "confidence": 1.0, "status": "rejected"})
            new_symptom_found = True
        else:
            conf, rej = extract_gejala_combined(user_text, symptom_index, ml_bundle, log)
            ss.gejala_terkonfirmasi |= conf
            ss.gejala_ditolak |= rej
            if ss.pending_gejala not in ss.gejala_terkonfirmasi and ss.pending_gejala not in ss.gejala_ditolak:
                ss.gejala_ditolak.add(ss.pending_gejala)
                log.append({"source": "DEFAULT", "gejala": ss.pending_gejala,
                            "confidence": None, "status": "rejected (tidak disebut ulang)"})
            new_symptom_found = True
        ss.pending_gejala = None
    else:
        conf, rej = extract_gejala_combined(user_text, symptom_index, ml_bundle, log)
        if conf or rej:
            ss.gejala_terkonfirmasi |= conf
            ss.gejala_ditolak |= rej
            new_symptom_found = True

    if not new_symptom_found:
        if not ss.gejala_terkonfirmasi and not ss.gejala_ditolak:
            bot_reply = (
                "Maaf, aku belum menangkap gejala spesifik dari ceritamu 🙏 "
                "Coba sebutkan dengan istilah yang lebih umum, misalnya "
                "*'demam'*, *'batuk'*, *'pilek'*, *'pusing'*, atau *'mual'*."
            )
        else:
            bot_reply = (
                "Hmm, aku tidak menangkap gejala baru dari pesan itu. "
                "Bisa dijelaskan dengan istilah lain, atau lanjutkan menjawab "
                "pertanyaan sebelumnya?"
            )
        ss.chat_history.append(("bot", bot_reply))
        ss.reasoning_log = log
        return
    ss.current_probabilities = calculate_probabilities(
        penyakit_list, ss.gejala_terkonfirmasi, ss.gejala_ditolak, log
    )

    decision = evaluate_rules(penyakit_list, symptom_index, ss.gejala_terkonfirmasi, ss.current_probabilities, log)

    if decision[0] == "darurat":
        gejala_key = decision[1]
        label = symptom_index[gejala_key]["label"]
        bot_reply = (
            f"🚨 **PERINGATAN DARURAT** 🚨\n\n"
            f"Gejala **{label}** yang kamu sebutkan tergolong tanda bahaya. "
            f"Segera pergi ke **UGD / IGD terdekat** untuk penanganan medis langsung. "
            f"Jangan ditunda.\n\n"
            f"**Rekomendasi tindakan:** Cari fasilitas kesehatan terdekat SEKARANG, "
            f"jangan menunggu gejala memburuk."
        )
        ss.status_diagnosis = True
        ss.pending_gejala = None

    elif decision[0] == "diagnosis":
        _, nama, prob, tindakan = decision
        bot_reply = (
            f"📋 **Hasil Diagnosa Awal**\n\n"
            f"Kemungkinan terbesar: **{nama}** (keyakinan sistem: {prob * 100:.1f}%)\n\n"
            f"**✅ Rekomendasi Tindakan:**\n{tindakan}\n\n"
            f"_Catatan: ini bukan pengganti diagnosa dokter. Silakan konsultasi lebih lanjut "
            f"ke tenaga medis profesional._"
        )
        ss.status_diagnosis = True
        ss.pending_gejala = None

    else:
        next_key = find_next_question_entropy(
            penyakit_list, ss.current_probabilities, ss.gejala_terkonfirmasi,
            ss.gejala_ditolak, symptom_index, log
        )

        if next_key is None:
            top_nama = next(iter(ss.current_probabilities))
            top_prob = ss.current_probabilities[top_nama]
            top_penyakit = next(p for p in penyakit_list if p["nama"] == top_nama)
            bot_reply = (
                f"Saya sudah menanyakan semua gejala relevan yang saya tahu. "
                f"Kemungkinan terbesar saat ini: **{top_nama}** ({top_prob * 100:.1f}%).\n\n"
                f"**✅ Rekomendasi Tindakan:**\n{top_penyakit.get('tindakan', '-')}\n\n"
                f"_Ini bukan pengganti diagnosa dokter._"
            )
            ss.status_diagnosis = True
        else:
            ss.pending_gejala = next_key
            label = symptom_index[next_key]["label"]
            bot_reply = f"Apakah kamu juga mengalami **{label}**?"

    ss.chat_history.append(("bot", bot_reply))
    ss.reasoning_log = log


st.set_page_config(page_title="AI Medical Expert System", page_icon="🩺", layout="wide")


def init_state():
    ss = st.session_state
    if "initialized" in ss:
        return
    ss.initialized = True
    ss.penyakit_list = load_penyakit(DATA_PATH)
    ss.symptom_index = build_symptom_index(ss.penyakit_list)
    ss.ml_bundle = train_ml_classifier(ss.symptom_index)
    ss.chat_history = [
        ("bot", "Halo 👋 Saya asisten diagnosa awal. Ceritakan keluhan kamu, misalnya "
                "*'saya demam tinggi dari kemarin dan sakit kepala'*.")
    ]
    ss.gejala_terkonfirmasi = set()
    ss.gejala_ditolak = set()
    ss.current_probabilities = {p["nama"]: p["prior_prob"] for p in ss.penyakit_list}
    ss.status_diagnosis = False
    ss.pending_gejala = None
    ss.reasoning_log = []


init_state()
ss = st.session_state

# ---- Sidebar: Debug Panel ----
with st.sidebar:
    st.markdown("### 🔍 Debug Panel (Dosen)")
    st.caption("Status internal sistem — real-time")

    if ss.current_probabilities:
        chart_data = dict(list(ss.current_probabilities.items())[:10])
        st.bar_chart(chart_data)
        st.metric("Entropy saat ini (bit)", f"{entropy(ss.current_probabilities):.3f}",
                   help="0 = sistem sangat yakin, semakin tinggi = semakin tidak pasti")

    st.markdown("**✅ Gejala Terkonfirmasi**")
    st.write(sorted(ss.gejala_terkonfirmasi) or "-")
    st.markdown("**❌ Gejala Ditolak**")
    st.write(sorted(ss.gejala_ditolak) or "-")
    st.markdown("**📊 Status Diagnosis**")
    st.write(ss.status_diagnosis)

    st.divider()
    st.markdown("### 🧠 Reasoning Trace (giliran terakhir)")

    log = ss.reasoning_log
    if log:
        input_entries = [e for e in log if "source" in e]
        prob_entries = [e for e in log if e.get("phase") == "PROBABILITY"]
        krr_entries = [e for e in log if e.get("phase") == "KRR"]
        search_entries = [e for e in log if e.get("phase") == "SEARCH"]

        if input_entries:
            with st.expander(f"🔎 Fase 1: Ekstraksi Gejala ({len(input_entries)} entri)", expanded=True):
                for e in input_entries:
                    conf_str = f"  _(confidence={e['confidence']})_" if e.get("confidence") is not None else ""
                    st.write(f"- **[{e['source']}]** `{e['gejala']}` → {e['status']}{conf_str}")

        if prob_entries:
            with st.expander("📐 Fase 2: Bayes Update (top 3 penyakit)"):
                for e in prob_entries:
                    st.write(f"**{e['nama']}** = {e['posterior']}")
                    st.caption(e["formula"])

        if krr_entries:
            with st.expander("⚖️ Fase 3: Rule Engine (Forward Chaining)", expanded=True):
                for e in krr_entries:
                    st.write(f"**{e['rule']}**")
                    st.caption(f"Kondisi: {e['kondisi']} → **{e['hasil']}**")

        if search_entries:
            with st.expander("🔍 Fase 4: Greedy Best-First Search", expanded=True):
                for e in search_entries:
                    st.caption(e["algoritma"])
                    st.write(f"Entropy saat ini: **{e['current_entropy']} bit**")
                    for g, ig in e["top_candidates"]:
                        marker = "👉 " if g == e["terpilih"] else "　"
                        st.write(f"{marker}`{g}` — info gain = {ig}")
    else:
        st.caption("Belum ada aktivitas.")

    if st.button("🔄 Reset Percakapan"):
        for key in ["initialized"]:
            ss.pop(key, None)
        st.rerun()

# ---- Main chat area ----
st.title("🩺 AI Medical Expert System")
st.caption("Search (Greedy+Entropy) · KRR (Forward Chaining) · ML (Naive Bayes) · Probability (Bayes Theorem)")

for role, msg in ss.chat_history:
    avatar = "🧑" if role == "user" else "🩺"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg)

if ss.pending_gejala is not None and not ss.status_diagnosis:
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ Ya", use_container_width=True):
            ss.chat_history.append(("user", "Ya"))
            run_pipeline("ya")
            st.rerun()
    with col2:
        if st.button("❌ Tidak", use_container_width=True):
            ss.chat_history.append(("user", "Tidak"))
            run_pipeline("tidak")
            st.rerun()

if not ss.status_diagnosis:
    user_input = st.chat_input("Ketik keluhan atau jawaban di sini...")
    if user_input:
        ss.chat_history.append(("user", user_input))
        run_pipeline(user_input)
        st.rerun()
else:
    st.info("Sesi diagnosa selesai. Klik **Reset Percakapan** di sidebar untuk mulai lagi.")