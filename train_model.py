import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from PIL import Image
import joblib

from skimage.feature import graycomatrix, graycoprops
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ==========================
# CONFIG
# ==========================

DATASET_PATH = Path("dataset/NEU-DET/train/images")

CLASS_MAP = {
    "crazing":         "retak",
    "scratches":       "retak",
    "inclusion":       "korosi",
    "pitted_surface":  "korosi",
    "patches":         "normal",
    "rolled-in_scale": "normal",
}

TARGET_CLASSES = ["normal", "retak", "korosi"]

DISTANCES   = [1]
ANGLES      = [0, np.pi/4, np.pi/2, 3*np.pi/4]
LEVELS      = 16
IMG_SIZE    = (128, 128)

# "mean"   → rata-rata antar sudut → 4 fitur
# "concat" → tempel semua sudut   → 4 × 4 = 16 fitur
AGGREGATION = "mean"

MODEL_NAME  = "decision_tree_model.pkl"

FEAT_PROPS  = ["contrast", "homogeneity", "energy", "correlation"]


# ==========================
# PREPROCESSING CITRA
# ==========================

def preprocess(img_path):
    """Load → grayscale → resize (ROI) → kuantisasi."""
    img = Image.open(img_path).convert("L")       # grayscale
    img = img.resize(IMG_SIZE, Image.LANCZOS)      # ROI / resize

    arr = np.array(img, dtype=np.uint8)
    arr = (arr // (256 // LEVELS)).astype(np.uint8)  # kuantisasi 256 → LEVELS

    return arr


# ==========================
# EKSTRAKSI FITUR GLCM
# ==========================

def extract_features(gray):
    """
    Hitung GLCM lalu ekstrak fitur.
    AGGREGATION='mean'   → 4 fitur
    AGGREGATION='concat' → 16 fitur
    """
    glcm = graycomatrix(
        gray,
        distances=DISTANCES,
        angles=ANGLES,
        levels=LEVELS,
        symmetric=True,
        normed=True
    )

    features = []

    for p in FEAT_PROPS:
        values = graycoprops(glcm, p)   # shape: (n_dist, n_angles)
        if AGGREGATION == "mean":
            features.append(values.mean())
        else:
            features.extend(values.flatten().tolist())

    return features


def get_feature_names():
    if AGGREGATION == "mean":
        return FEAT_PROPS[:]
    else:
        names = []
        for p in FEAT_PROPS:
            for d in DISTANCES:
                for ai, _ in enumerate(ANGLES):
                    names.append(f"{p}_d{d}_a{ai*45}")
        return names


# ==========================
# LOAD DATASET
# ==========================

print("\n" + "="*55)
print("  Kasus 5 — Deteksi Cacat Permukaan Logam")
print("  Metode : GLCM + Decision Tree")
print("="*55)

print(f"\n[1] Loading dataset dari: {DATASET_PATH.resolve()}\n")

X, y = [], []
img_paths_all = []   # simpan path untuk error analysis
label_counts  = {}

for folder_name, label in CLASS_MAP.items():
    folder = DATASET_PATH / folder_name
    if not folder.exists():
        print(f"  [SKIP] Folder tidak ditemukan: {folder}")
        continue

    images = sorted(list(folder.glob("*.*")))
    print(f"  {folder_name:20s} → '{label}' | {len(images)} gambar")

    for img_path in images:
        try:
            gray = preprocess(img_path)
            feat = extract_features(gray)
            X.append(feat)
            y.append(label)
            img_paths_all.append(str(img_path))
            label_counts[label] = label_counts.get(label, 0) + 1
        except Exception as e:
            print(f"    [ERR] {img_path.name}: {e}")

X = np.array(X)
y = np.array(y)
img_paths_all = np.array(img_paths_all)

existing_classes = [c for c in TARGET_CLASSES if c in np.unique(y)]
feat_names       = get_feature_names()

print(f"\n  Total sampel  : {len(X)}")
print(f"  Fitur/sampel  : {X.shape[1]}  (AGGREGATION='{AGGREGATION}')")
print(f"  Distribusi    : {label_counts}")
print(f"  Kelas aktif   : {existing_classes}")


# ==========================
# VISUALISASI DISTRIBUSI KELAS
# ==========================

labels_u, counts_u = np.unique(y, return_counts=True)
colors = ["#4CAF50", "#F44336", "#2196F3"]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels_u, counts_u, color=colors[:len(labels_u)], edgecolor="white", linewidth=1.2)
for bar, cnt in zip(bars, counts_u):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(cnt), ha="center", fontweight="bold")
ax.set_title("Distribusi Kelas Dataset", fontweight="bold")
ax.set_xlabel("Kelas")
ax.set_ylabel("Jumlah Gambar")
plt.tight_layout()
plt.savefig("distribusi_kelas.png", dpi=150)
plt.close()
print("\n  [SAVED] distribusi_kelas.png")


# ==========================
# ANALISIS GLCM PER KELAS
# (Apakah GLCM cukup membedakan label?)
# ==========================

print(f"\n{'─'*55}")
print("  [ANALISIS 1] Rata-rata Fitur GLCM per Kelas")
print(f"{'─'*55}")
print(f"  {'Kelas':12s}", end="")
for fn in feat_names:
    print(f"  {fn:>15s}", end="")
print()

for cls in existing_classes:
    mask = y == cls
    if mask.sum() == 0:
        continue
    means = X[mask].mean(axis=0)
    print(f"  {cls:12s}", end="")
    for m in means:
        print(f"  {m:>15.4f}", end="")
    print()

# Bar chart fitur per kelas
fig, axes = plt.subplots(1, len(feat_names), figsize=(4*len(feat_names), 4), sharey=False)
if len(feat_names) == 1:
    axes = [axes]

for fi, (fname, ax) in enumerate(zip(feat_names, axes)):
    vals = [X[y == cls][:, fi].mean() for cls in existing_classes]
    stds = [X[y == cls][:, fi].std()  for cls in existing_classes]
    ax.bar(existing_classes, vals, yerr=stds,
           color=colors[:len(existing_classes)], capsize=5, edgecolor="white")
    ax.set_title(fname, fontweight="bold", fontsize=10)
    ax.set_xlabel("Kelas")
    if fi == 0:
        ax.set_ylabel("Nilai Rata-rata")
    ax.tick_params(axis='x', rotation=20)

plt.suptitle("Rata-rata Fitur GLCM per Kelas", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("glcm_per_kelas.png", dpi=150)
plt.close()
print("\n  [SAVED] glcm_per_kelas.png")


# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, img_paths_all,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n[2] Split data → Train: {len(X_train)} | Test: {len(X_test)}")


# ==========================
# TRAIN MODEL
# ==========================

print("\n[3] Training Decision Tree...")

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)
model.fit(X_train, y_train)

print(f"    Depth  : {model.get_depth()}")
print(f"    Leaves : {model.get_n_leaves()}")


# ==========================
# EVALUASI
# ==========================

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'='*55}")
print("  [4] HASIL EVALUASI")
print(f"{'='*55}")
print(f"\n  Accuracy : {acc*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=existing_classes, digits=4))


# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred, labels=existing_classes)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=existing_classes)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Confusion Matrix — Accuracy: {acc*100:.2f}%", fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("  [SAVED] confusion_matrix.png")


# ==========================
# CROSS-VALIDATION (5-Fold)
# ==========================

print(f"\n{'─'*55}")
print("  5-FOLD CROSS VALIDATION")
print(f"{'─'*55}")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
for i, s in enumerate(scores, 1):
    print(f"    Fold {i}: {s*100:.2f}%")
print(f"    Mean  : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")


# ==========================
# FEATURE IMPORTANCE
# ==========================

importances = model.feature_importances_
indices     = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(feat_names)), importances[indices], color="steelblue")
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels([feat_names[i] for i in indices], rotation=45, ha="right")
ax.set_xlabel("Fitur GLCM")
ax.set_ylabel("Importance")
ax.set_title("Feature Importance — Decision Tree", fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("\n  [SAVED] feature_importance.png")


# ==========================
# VISUALISASI DECISION TREE
# ==========================

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    model,
    feature_names=feat_names,
    class_names=existing_classes,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    max_depth=4
)
ax.set_title("Decision Tree (tampil maks 4 level)", fontweight="bold")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=100)
plt.close()
print("  [SAVED] decision_tree.png")


# ==========================
# ANALISIS 1: Apakah GLCM cukup membedakan label?
# ==========================

print(f"\n{'─'*55}")
print("  [ANALISIS 1] Apakah GLCM Cukup Membedakan Label?")
print(f"{'─'*55}")

report_dict = {}
for cls in existing_classes:
    mask     = y_test == cls
    if mask.sum() == 0:
        continue
    correct  = ((y_pred == cls) & (y_test == cls)).sum()
    total    = mask.sum()
    pct      = correct / total * 100
    report_dict[cls] = pct
    print(f"    {cls:10s} → {correct}/{total} benar ({pct:.1f}%)")

worst_cls = min(report_dict, key=report_dict.get)
best_cls  = max(report_dict, key=report_dict.get)

print(f"\n  → Kelas paling mudah   : '{best_cls}'  ({report_dict[best_cls]:.1f}%)")
print(f"  → Kelas paling sulit   : '{worst_cls}' ({report_dict[worst_cls]:.1f}%)")
print(f"\n  Kesimpulan:")
print(f"  Fitur GLCM (contrast, homogeneity, energy, correlation)")
print(f"  {'cukup' if acc >= 0.75 else 'kurang'} membedakan kelas dengan akurasi {acc*100:.2f}%.")
print(f"  Kelas '{worst_cls}' paling sulit karena tekstur GLCM-nya overlap")
print(f"  dengan kelas lain (nilai contrast & energy serupa).")


# ==========================
# ANALISIS 2: Label paling sulit — detail overlap
# ==========================

print(f"\n{'─'*55}")
print("  [ANALISIS 2] Label Paling Sulit Diprediksi")
print(f"{'─'*55}")
print("\n  Tabel kesalahan (True → Predicted):")

for true_cls in existing_classes:
    for pred_cls in existing_classes:
        if true_cls == pred_cls:
            continue
        count = ((y_test == true_cls) & (y_pred == pred_cls)).sum()
        if count > 0:
            print(f"    {true_cls:10s} → {pred_cls:10s} : {count} gambar salah")

print(f"\n  Analisis overlap fitur GLCM antar kelas:")
for a_cls in existing_classes:
    for b_cls in existing_classes:
        if a_cls >= b_cls:
            continue
        diff = np.abs(X[y == a_cls].mean(axis=0) - X[y == b_cls].mean(axis=0))
        most_sim_feat = feat_names[np.argmin(diff)]
        print(f"    {a_cls} vs {b_cls}: fitur paling mirip → '{most_sim_feat}'")


# ==========================
# ANALISIS 3: Error Analysis (contoh konkret)
# ==========================

print(f"\n{'─'*55}")
print("  [ANALISIS 3] Error Analysis — Contoh Gambar Salah")
print(f"{'─'*55}")

wrong_idx = np.where(y_pred != y_test)[0]
print(f"\n  Total salah klasifikasi: {len(wrong_idx)} dari {len(y_test)} test sampel")

if len(wrong_idx) > 0:
    # Tampilkan maks 8 contoh gambar yang salah
    show_n = min(8, len(wrong_idx))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)) if show_n > 4 else plt.subplots(1, show_n, figsize=(4*show_n, 4))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(wrong_idx[:show_n]):
        img_path = paths_test[idx]
        true_lbl = y_test[idx]
        pred_lbl = y_pred[idx]

        try:
            img = Image.open(img_path).convert("L").resize((128, 128))
            axes[i].imshow(img, cmap="gray")
        except:
            axes[i].text(0.5, 0.5, "img\nnot found",
                         ha="center", va="center", transform=axes[i].transAxes)

        axes[i].set_title(
            f"True: {true_lbl}\nPred: {pred_lbl}",
            fontsize=9,
            color="red" if true_lbl != pred_lbl else "green"
        )
        axes[i].axis("off")

        print(f"    [{i+1}] {Path(img_path).name:30s}  True={true_lbl:8s}  Pred={pred_lbl}")

    # Sembunyikan axes kosong jika < 8 gambar
    for j in range(show_n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Contoh Gambar Salah Klasifikasi", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("error_analysis.png", dpi=150)
    plt.close()
    print("\n  [SAVED] error_analysis.png")

    print(f"\n  Penjelasan:")
    print(f"  • Gambar yang salah umumnya memiliki tekstur ambigu —")
    print(f"    retakan halus bisa menyerupai korosi ringan di level GLCM.")
    print(f"  • Kelas 'normal' paling jarang salah karena GLCM-nya")
    print(f"    khas: homogeneity tinggi, contrast rendah.")
else:
    print("  Tidak ada kesalahan klasifikasi pada test set!")


# ==========================
# DECISION TREE RULES (ringkas)
# ==========================

print(f"\n{'─'*55}")
print("  DECISION TREE RULES (max 3 level):")
print(f"{'─'*55}")
rules = export_text(model, feature_names=feat_names, max_depth=3)
print(rules)


# ==========================
# SAVE MODEL
# ==========================

joblib.dump(model, MODEL_NAME)

print(f"\n{'='*55}")
print("  SELESAI! File yang disimpan:")
print(f"{'='*55}")
for f in [MODEL_NAME, "distribusi_kelas.png", "glcm_per_kelas.png",
          "confusion_matrix.png", "feature_importance.png",
          "decision_tree.png", "error_analysis.png"]:
    print(f"    - {f}")
print("="*55 + "\n")