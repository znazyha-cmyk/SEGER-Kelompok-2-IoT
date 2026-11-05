"""
SEGER - Sistem Evaluasi Gas dan Environmental Room
Edge 2 (HP Android via Termux) - VERSI AI

Alur:
  ESP32/Wokwi --(MQTT: seger/<ID>/data)--> script ini
  script ini --(MQTT: seger/<ID>/status)--> ESP32 + aplikasi MQTT di HP

Data processing 2 (dua lapis AI):
  1. KLASIFIKASI  : Decision Tree (hasil training, file seger_model.py)
                    -> NYAMAN / MULAI PENGAP / PENGAP
                    Rule-based tetap dijalankan sebagai pembanding & cadangan.
  2. PREDIKSI TREN: Regresi linier pada riwayat CO2 terakhir
                    -> memperkirakan CO2 beberapa detik ke depan dan memberi
                       PERINGATAN DINI sebelum ruangan benar-benar pengap.

Install di Termux:
  pkg update && pkg install python
  pip install paho-mqtt
Salin ke folder yang sama: seger_edge2.py + seger_model.py
Jalankan:
  python seger_edge2.py
"""

import csv
import json
import os
import time
from collections import deque
from datetime import datetime

import paho.mqtt.client as mqtt

try:
    from seger_model import predict_ai  # hasil training ai/train_model.py
    MODEL_OK = True
except ImportError:
    MODEL_OK = False

# ================== KONFIGURASI ==================
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_ID = "kelompok2"  # HARUS sama dengan TOPIC_ID di sketch.ino

TOPIC_DATA = f"seger/{TOPIC_ID}/data"
TOPIC_STATUS = f"seger/{TOPIC_ID}/status"

LOG_FILE = "seger_log.csv"

# Prediksi tren
TREND_WINDOW = 15      # jumlah data terakhir yang dipakai (15 x 2 dtk = 30 dtk)
TREND_MIN_POINTS = 5   # minimal data sebelum prediksi dipercaya
HORIZON_S = 60         # prediksi CO2 berapa detik ke depan
CO2_MULAI = 800        # batas "mulai pengap" (ppm)
CO2_PENGAP = 1200      # batas "pengap" (ppm)
MIN_SLOPE = 0.5        # ppm/detik; di bawah ini dianggap tidak naik
# =================================================

LEVEL = {"NYAMAN": 0, "MULAI PENGAP": 1, "PENGAP": 2}
SARAN = {
    "NYAMAN": "Udara baik.",
    "MULAI PENGAP": "Pertimbangkan membuka ventilasi.",
    "PENGAP": "Buka jendela/pintu atau nyalakan ventilasi sekarang!",
}


# ---------- Algoritma 1a: rule-based (pembanding/cadangan) ----------
def classify_rule(temp, hum, co2):
    """Skor: CO2 800-1199 (+1), CO2>=1200 (+2), suhu>30 (+1), lembap>70 (+1)."""
    score = 0
    if co2 >= CO2_PENGAP:
        score += 2
    elif co2 >= CO2_MULAI:
        score += 1
    if temp > 30:
        score += 1
    if hum > 70:
        score += 1
    return ["NYAMAN", "MULAI PENGAP", "PENGAP"][min(score, 2)]


# ---------- Algoritma 1b: Decision Tree ----------
def classify_ai(temp, hum, co2):
    """Pakai Decision Tree; kalau model tidak ada, jatuh ke rule-based."""
    if MODEL_OK:
        return predict_ai(temp, hum, co2)
    return classify_rule(temp, hum, co2)


# ---------- Algoritma 2: prediksi tren CO2 (regresi linier) ----------
class TrendPredictor:
    """Regresi linier (kuadrat terkecil) pada pasangan (waktu, CO2)."""

    def __init__(self, window=TREND_WINDOW):
        self.points = deque(maxlen=window)

    def add(self, t, co2):
        self.points.append((t, co2))

    def fit(self):
        """Return (slope ppm/detik, intercept) atau None kalau data kurang."""
        n = len(self.points)
        if n < TREND_MIN_POINTS:
            return None
        mean_t = sum(p[0] for p in self.points) / n
        mean_c = sum(p[1] for p in self.points) / n
        var = sum((p[0] - mean_t) ** 2 for p in self.points)
        if var == 0:
            return None
        cov = sum((p[0] - mean_t) * (p[1] - mean_c) for p in self.points)
        slope = cov / var
        return slope, mean_c - slope * mean_t

    def predict(self, t_now, co2_now, horizon=HORIZON_S):
        """Return (co2_prediksi, slope, eta_detik_ke_batas_berikutnya)."""
        model = self.fit()
        if model is None:
            return co2_now, 0.0, None
        slope, intercept = model
        co2_pred = max(0.0, intercept + slope * (t_now + horizon))

        eta = None
        if slope >= MIN_SLOPE:
            if co2_now < CO2_MULAI:
                target = CO2_MULAI
            elif co2_now < CO2_PENGAP:
                target = CO2_PENGAP
            else:
                target = None
            if target is not None:
                eta = max(0.0, (target - co2_now) / slope)
        return co2_pred, slope, eta


def process(temp, hum, co2, trend, t_now):
    """Seluruh data processing 2. Return dict hasil (siap di-publish)."""
    status_ai = classify_ai(temp, hum, co2)
    status_rule = classify_rule(temp, hum, co2)

    trend.add(t_now, co2)
    co2_pred, slope, eta = trend.predict(t_now, co2)
    status_pred = classify_ai(temp, hum, co2_pred)

    # Peringatan dini: kondisi sekarang masih lebih baik daripada prediksi
    early = LEVEL[status_pred] > LEVEL[status_ai] and slope >= MIN_SLOPE
    if early:
        saran = (f"PERINGATAN DINI: diprediksi {status_pred}"
                 + (f" dalam ~{int(eta)} dtk" if eta is not None else "")
                 + ". Buka ventilasi sekarang.")
    else:
        saran = SARAN[status_ai]

    return {
        "status": status_ai,
        "status_rule": status_rule,
        "peringatan_dini": early,
        "co2_prediksi": round(co2_pred),
        "tren_ppm_per_dtk": round(slope, 2),
        "eta_dtk": None if eta is None else int(eta),
        "saran": saran,
        "temp": temp,
        "hum": hum,
        "co2": co2,
    }


def log_to_csv(row):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["waktu", "temp", "hum", "co2", "status_ai",
                             "status_rule", "co2_prediksi", "peringatan_dini"])
        writer.writerow(row)


trend_model = TrendPredictor()


def on_connect(client, userdata, *args):
    print(f"Terhubung ke {BROKER}")
    client.subscribe(TOPIC_DATA)
    print(f"Subscribe: {TOPIC_DATA}")
    print(f"Publish  : {TOPIC_STATUS}")
    print("Model     :", "Decision Tree (seger_model.py)" if MODEL_OK
          else "seger_model.py TIDAK ditemukan -> pakai rule-based")
    print("Menunggu data dari ESP32...\n")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data["temp"])
        hum = float(data["hum"])
        co2 = float(data["co2"])
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"Data tidak valid, dilewati: {e}")
        return

    result = process(temp, hum, co2, trend_model, time.time())
    client.publish(TOPIC_STATUS, json.dumps(result))

    waktu = datetime.now().strftime("%H:%M:%S")
    tanda = "  !! PERINGATAN DINI" if result["peringatan_dini"] else ""
    print(f"[{waktu}] T={temp}C RH={hum}% CO2={co2:.0f}ppm | "
          f"AI={result['status']} (rule={result['status_rule']}) | "
          f"prediksi {HORIZON_S}dtk: {result['co2_prediksi']}ppm{tanda}")
    log_to_csv([datetime.now().isoformat(), temp, hum, co2, result["status"],
                result["status_rule"], result["co2_prediksi"],
                result["peringatan_dini"]])


def make_client():
    # Kompatibel dengan paho-mqtt v1 dan v2
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def main():
    client = make_client()
    client.on_connect = on_connect
    client.on_message = on_message

    while True:
        try:
            client.connect(BROKER, PORT, keepalive=60)
            client.loop_forever()
        except KeyboardInterrupt:
            print("\nBerhenti.")
            break
        except Exception as e:
            print(f"Koneksi bermasalah ({e}), coba lagi 5 detik...")
            time.sleep(5)


if __name__ == "__main__":
    main()
