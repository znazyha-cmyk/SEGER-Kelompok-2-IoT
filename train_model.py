"""
SEGER - Training model AI (Decision Tree)

Dijalankan di LAPTOP (butuh scikit-learn), bukan di HP.
Hasilnya diekspor jadi file Python murni (seger_model.py) tanpa library
tambahan, sehingga ringan dan langsung jalan di Termux (edge 2).

Install:   pip install scikit-learn pandas matplotlib
Jalankan:  python generate_dataset.py
           python train_model.py
Output  :  ../edge2/seger_model.py       (model untuk HP)
           ../docs/confusion_matrix.png  (untuk PPT)
           ../docs/decision_tree.png     (untuk PPT)
           ../docs/hasil_training.txt    (angka akurasi untuk PPT)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA = "dataset_seger.csv"
MODEL_OUT = os.path.join("..", "edge2", "seger_model.py")
DOCS = os.path.join("..", "docs")
FEATURES = ["temp", "hum", "co2"]
CLASS_ORDER = ["NYAMAN", "MULAI PENGAP", "PENGAP"]
MAX_DEPTH = 4  # dibatasi supaya model kecil, mudah dijelaskan, tidak overfit
MIN_LEAF = 25  # minimal sampel per daun, supaya tidak menghafal noise


def rule_based(temp, hum, co2):
    """Baseline: algoritma skor sederhana (sama dengan seger_edge2.py)."""
    score = 0
    if co2 >= 1200:
        score += 2
    elif co2 >= 800:
        score += 1
    if temp > 30:
        score += 1
    if hum > 70:
        score += 1
    return CLASS_ORDER[min(score, 2)]


def export_tree_to_python(clf, class_names):
    """Ubah decision tree menjadi fungsi if/else Python murni (cabang kembar digabung)."""
    t = clf.tree_

    def build(node):
        """Return ('leaf', kelas) atau ('node', fitur, batas, kiri, kanan)."""
        if t.feature[node] == -2:  # daun
            return ("leaf", class_names[int(np.argmax(t.value[node][0]))])
        left = build(t.children_left[node])
        right = build(t.children_right[node])
        if left[0] == "leaf" and right[0] == "leaf" and left[1] == right[1]:
            return left  # kedua cabang hasilnya sama -> tidak perlu percabangan
        return ("node", FEATURES[t.feature[node]],
                round(float(t.threshold[node]), 2), left, right)

    lines = []

    def emit(n, indent):
        pad = "    " * indent
        if n[0] == "leaf":
            lines.append(f"{pad}return {n[1]!r}")
            return
        _, name, thr, left, right = n
        lines.append(f"{pad}if {name} <= {thr}:")
        emit(left, indent + 1)
        lines.append(f"{pad}else:")
        emit(right, indent + 1)

    emit(build(0), 1)
    return "\n".join(lines)


def main():
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=42
    )
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)

    base_pred = [rule_based(r.temp, r.hum, r.co2) for r in X_te.itertuples()]
    base_acc = accuracy_score(y_te, base_pred)

    report = classification_report(y_te, pred, labels=CLASS_ORDER, digits=3)
    importance = {k: float(round(v, 3)) for k, v in zip(FEATURES, clf.feature_importances_)}

    summary = (
        "HASIL TRAINING SEGER\n"
        "====================\n"
        f"Data latih / uji      : {len(X_tr)} / {len(X_te)}\n"
        f"Algoritma             : Decision Tree (max_depth={MAX_DEPTH}, min_samples_leaf={MIN_LEAF})\n"
        f"Akurasi Decision Tree : {acc * 100:.1f}%\n"
        f"Akurasi rule-based    : {base_acc * 100:.1f}%  (pembanding)\n"
        f"Pentingnya fitur      : {importance}\n\n"
        f"{report}\n"
    )
    print(summary)

    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "hasil_training.txt"), "w") as f:
        f.write(summary)

    # Confusion matrix
    cm = confusion_matrix(y_te, pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix - Decision Tree SEGER")
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Gambar pohon (dibatasi 3 level supaya terbaca di slide)
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(clf, feature_names=FEATURES, class_names=list(clf.classes_),
              filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title("Decision Tree SEGER (3 level teratas)")
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS, "decision_tree.png"), dpi=150)
    plt.close()

    # Ekspor model ke Python murni
    body = export_tree_to_python(clf, list(clf.classes_))
    code = f'''"""
SEGER - Model Decision Tree (dibuat otomatis oleh ai/train_model.py)
JANGAN diedit manual. Latih ulang lewat train_model.py kalau data berubah.

Akurasi uji: {acc * 100:.1f}%  |  Data latih: {len(X_tr)}  |  max_depth: {MAX_DEPTH}
Fitur: temp (C), hum (%), co2 (ppm)
Kelas: {", ".join(clf.classes_)}
"""


def predict_ai(temp, hum, co2):
    """Return label kondisi ruangan hasil prediksi Decision Tree."""
{body}
'''
    with open(MODEL_OUT, "w") as f:
        f.write(code)
    print(f"Model diekspor ke: {MODEL_OUT}")


if __name__ == "__main__":
    main()
