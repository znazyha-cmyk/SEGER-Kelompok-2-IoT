# SEGER - Sistem Evaluasi Gas dan Environmental Room

Sistem evaluasi kualitas udara dan kondisi lingkungan ruangan berbasis **multi-edge computing** untuk **deteksi dini kondisi pengap**. Tugas IoT - Kelompok 2.

**Anggota:** [Nazyha Zahara Al-Mumtaza] ([24060125130054]), [Nadia Ayu Pratiwi] ([24060125120004]), [Amara Revi Khayla Karunia Mulya] ([24060125130059])
**Simulasi Wokwi:** [(https://wokwi.com/projects/479852805800868865)]
**Video demo:** [tempel link video di sini]

## Ide singkat

Ruangan tertutup sering pengap karena CO2 menumpuk, suhu naik, dan kelembapan tinggi, tetapi orang baru sadar setelah pusing atau mengantuk. SEGER membaca kondisi ruangan, mengklasifikasikannya dengan AI, dan **memberi peringatan dini** sebelum ruangan benar-benar pengap.

## Arsitektur

```
Sensor (DHT22 + CO2*)
        |
   ESP32 (Edge 1)  -- moving average (data processing 1)
        |
   MQTT: seger/kelompok2/data   (broker.hivemq.com)
        |
   HP Android (Edge 2, Termux)
        - Decision Tree: NYAMAN / MULAI PENGAP / PENGAP
        - Regresi linier: prediksi CO2 & peringatan dini   (data processing 2)
        |
   MQTT: seger/kelompok2/status
        |
   ESP32 (Serial Monitor) + aplikasi IoT MQTT Panel di HP (dashboard)
```
\* Di Wokwi, sensor CO2 disimulasikan dengan potensiometer. Pada implementasi nyata diganti MQ-135 atau SCD40.

## Pemenuhan ketentuan tugas

| Ketentuan | Pemenuhan |
|---|---|
| Berkelompok 3 orang | Ya |
| Min. 2 edge device | ESP32 + HP Android |
| Min. 2 sensor | DHT22 (suhu, kelembapan) + sensor CO2 (simulasi) |
| Min. 2 jenis data processing | (1) Moving average di ESP32, (2) klasifikasi + prediksi tren di HP |
| Algoritma AI/lain | Decision Tree + regresi linier |
| Min. 1 protokol ke cloud | MQTT (HiveMQ) |
| Dashboard | Opsional: IoT MQTT Panel di HP |

## Struktur folder

```
edge1/   sketch.ino, diagram.json, libraries.txt   -> project Wokwi (ESP32)
edge2/   seger_edge2.py, seger_model.py            -> script HP (Termux)
ai/      generate_dataset.py, train_model.py, dataset_seger.csv
docs/    confusion_matrix.png, decision_tree.png
```

## Cara menjalankan

### 1. Edge 1 (Wokwi)
1. Buat project ESP32 baru di wokwi.com.
2. Salin isi `edge1/sketch.ino`, `edge1/diagram.json`, dan `edge1/libraries.txt` ke tab yang sesuai.
3. Pastikan `TOPIC_ID` = `kelompok2`, lalu klik **Play**.

### 2. Edge 2 (HP / Termux)
```bash
pkg update && pkg install python
pip install paho-mqtt
# salin seger_edge2.py DAN seger_model.py ke folder yang sama
python seger_edge2.py
```

### 3. Tampilan di HP
Aplikasi **IoT MQTT Panel** -> broker `broker.hivemq.com` port `1883` -> tambahkan panel untuk topic `seger/kelompok2/status` (dan `seger/kelompok2/data`).

### 4. Demo
Saat simulasi berjalan, klik potensiometer di Wokwi lalu geser slider untuk menaikkan CO2. Status berubah bertahap karena filtering; peringatan dini muncul saat tren naik. DHT22 juga bisa diklik untuk mengubah suhu dan kelembapan.

## Melatih ulang model AI (opsional)

```bash
cd ai
pip install scikit-learn pandas matplotlib
python generate_dataset.py     # atau ganti dataset_seger.csv dengan data nyata
python train_model.py          # menulis ulang edge2/seger_model.py dan grafik di docs/
```

## Hasil AI

| Metode | Akurasi (data uji) |
|---|---|
| Decision Tree | 85,2% |
| Rule-based (pembanding) | 78,3% |

