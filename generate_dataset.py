"""
SEGER - Pembuat dataset (sintetis) untuk melatih model AI.

Kenapa sintetis?
  Prototype berjalan di simulator Wokwi, jadi belum ada data ruangan sungguhan
  yang berlabel. Dataset dibuat dari "indeks ketidaknyamanan" berikut, ditambah
  noise supaya batas antar kelas tidak kaku (meniru persepsi orang yang berbeda-beda):

    skor = (CO2 - 400) * 0.05          -> kontribusi CO2 (0 .. 80)
         + max(0, suhu - 28) * 3        -> kontribusi suhu
         + max(0, lembap - 65) * 0.6    -> kontribusi kelembapan
         + noise Normal(0, 4)

    skor < 20        -> NYAMAN
    20 <= skor < 40  -> MULAI PENGAP
    skor >= 40       -> PENGAP

  Dengan bobot ini, CO2 saja (suhu & lembap normal) menghasilkan batas
  sekitar 800 ppm (mulai pengap) dan 1200 ppm (pengap), sesuai pedoman umum
  kualitas udara dalam ruangan.

PENTING: kalau nanti ada data nyata yang diberi label manual (misal dari
survei "apakah ruangan ini terasa pengap?"), ganti file hasil ini dengan data
tersebut lalu jalankan ulang train_model.py.

Jalankan:  python generate_dataset.py
Output  :  dataset_seger.csv
"""

import csv
import numpy as np

N = 3000
SEED = 42
OUT = "dataset_seger.csv"

LABELS = ["NYAMAN", "MULAI PENGAP", "PENGAP"]


def label_from_score(score):
    if score < 20:
        return "NYAMAN"
    if score < 40:
        return "MULAI PENGAP"
    return "PENGAP"


def main():
    rng = np.random.default_rng(SEED)

    co2 = rng.uniform(400, 1600, N)
    temp = rng.uniform(22, 33, N)
    hum = rng.uniform(35, 85, N)

    score = (
        (co2 - 400) * 0.05
        + np.maximum(0, temp - 28) * 3
        + np.maximum(0, hum - 65) * 0.6
        + rng.normal(0, 4, N)
    )

    rows = []
    counts = {k: 0 for k in LABELS}
    for t, h, c, s in zip(temp, hum, co2, score):
        lab = label_from_score(s)
        counts[lab] += 1
        rows.append([round(t, 1), round(h, 1), round(c), lab])

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temp", "hum", "co2", "label"])
        w.writerows(rows)

    print(f"Dataset disimpan: {OUT} ({N} baris)")
    for k, v in counts.items():
        print(f"  {k:13s}: {v}")


if __name__ == "__main__":
    main()
