"""
=============================================================
  REAL-TIME CREDIT CARD FRAUD DETECTION
  Using Kaggle Credit Card Fraud Dataset
  
  Dataset : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  File    : creditcard.csv (place in same folder as this script)

  Run     : python fraud_detection.py
  Install : pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
=============================================================
"""

# ─── IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# Colour palette used throughout plots
C_FRAUD  = "#C94A2A"   # red-orange  → fraud
C_LEGIT  = "#2A6B4A"   # forest green → legitimate
C_BLUE   = "#1A5E8A"   # deep blue    → model 2


# ══════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 — Loading dataset")
print("═"*60)

CSV_PATH = "creditcard.csv"

if not os.path.exists(CSV_PATH):
    print(f"\n  ERROR: '{CSV_PATH}' not found in current directory.")
    print("  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  Then place creditcard.csv in the same folder as this script.\n")
    exit(1)

df = pd.read_csv(CSV_PATH)

print(f"\n  Rows      : {len(df):,}")
print(f"  Columns   : {df.shape[1]}")
print(f"  Fraud     : {df['Class'].sum():,}  ({df['Class'].mean()*100:.3f}%)")
print(f"  Legitimate: {(df['Class']==0).sum():,}  ({(df['Class']==0).mean()*100:.3f}%)")
print(f"\n  Columns   : {list(df.columns)}")
print(f"\n  First 3 rows:\n{df.head(3).to_string()}")


# ══════════════════════════════════════════════════════════
# STEP 2 — EXPLORE CLASS IMBALANCE
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 — Visualising class imbalance")
print("═"*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Step 2 — Class Imbalance & Key Feature Distributions", fontsize=13, fontweight="bold")

# 2a. Class balance bar
counts = df["Class"].value_counts()
axes[0].bar(["Legitimate", "Fraud"], counts.values, color=[C_LEGIT, C_FRAUD], width=0.5, edgecolor="white")
axes[0].set_title("Class Distribution", fontsize=11)
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

# 2b. Transaction amount distribution
axes[1].hist(df[df.Class==0]["Amount"], bins=60, alpha=0.7, color=C_LEGIT, label="Legitimate", density=True)
axes[1].hist(df[df.Class==1]["Amount"], bins=60, alpha=0.7, color=C_FRAUD, label="Fraud", density=True)
axes[1].set_title("Transaction Amount (₹)", fontsize=11)
axes[1].set_xlabel("Amount"); axes[1].set_ylabel("Density")
axes[1].legend(); axes[1].set_xlim(0, 2000)

# 2c. Time distribution
axes[2].hist(df[df.Class==0]["Time"]/3600, bins=48, alpha=0.7, color=C_LEGIT, label="Legitimate", density=True)
axes[2].hist(df[df.Class==1]["Time"]/3600, bins=48, alpha=0.7, color=C_FRAUD, label="Fraud", density=True)
axes[2].set_title("Transaction Time (hours)", fontsize=11)
axes[2].set_xlabel("Hours elapsed"); axes[2].set_ylabel("Density")
axes[2].legend()

plt.tight_layout()
plt.savefig("plot_step2_imbalance.png", dpi=130, bbox_inches="tight")
plt.show()
print("  Saved → plot_step2_imbalance.png")

print(f"""
  KEY INSIGHT:
  Fraud is only {df['Class'].mean()*100:.3f}% of transactions!
  If a model just predicts "Legitimate" for everything, it gets
  {(df['Class']==0).mean()*100:.2f}% accuracy — but catches ZERO fraud cases.
  This is why accuracy is a useless metric here.
  We need: Recall (did we catch the fraud?) and Precision (was our alert correct?)
""")


# ══════════════════════════════════════════════════════════
# STEP 3 — EXPLORE FEATURE SEPARABILITY (V-features)
# ══════════════════════════════════════════════════════════
print("═"*60)
print("  STEP 3 — Exploring V-feature distributions (PCA components)")
print("═"*60)

# Pick top 6 V-features most different between fraud/legit
v_cols = [f"V{i}" for i in range(1, 29)]
diffs = []
for col in v_cols:
    m_legit = df[df.Class==0][col].mean()
    m_fraud = df[df.Class==1][col].mean()
    diffs.append((col, abs(m_fraud - m_legit)))

top6 = sorted(diffs, key=lambda x: x[1], reverse=True)[:6]
top6_cols = [t[0] for t in top6]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("Step 3 — Top 6 PCA Features Separating Fraud vs Legitimate", fontsize=13, fontweight="bold")

for ax, col in zip(axes.flatten(), top6_cols):
    ax.hist(df[df.Class==0][col], bins=50, alpha=0.65, color=C_LEGIT, label="Legitimate", density=True)
    ax.hist(df[df.Class==1][col], bins=50, alpha=0.65, color=C_FRAUD, label="Fraud", density=True)
    ax.set_title(col, fontsize=11); ax.legend(fontsize=8); ax.set_ylabel("Density")

plt.tight_layout()
plt.savefig("plot_step3_features.png", dpi=130, bbox_inches="tight")
plt.show()
print("  Saved → plot_step3_features.png")
print(f"  Top separating features: {top6_cols}")


# ══════════════════════════════════════════════════════════
# STEP 4 — FEATURE SCALING
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 — Scaling Amount and Time features")
print("═"*60)

# V1-V28 are already PCA-scaled by Kaggle.
# Amount and Time are raw → we scale them to match the same range.
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
X = df[feature_cols].values
y = df["Class"].values

print(f"  Features used : {len(feature_cols)}")
print(f"  V1–V28        : PCA components (already scaled by Kaggle)")
print(f"  Amount_scaled : StandardScaler applied")
print(f"  Time_scaled   : StandardScaler applied")


# ══════════════════════════════════════════════════════════
# STEP 5 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 5 — Stratified train/test split (80/20)")
print("═"*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train set : {len(X_train):,} rows  |  Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
print(f"  Test set  : {len(X_test):,}  rows  |  Fraud in test : {y_test.sum():,}  ({y_test.mean()*100:.3f}%)")
print(f"\n  stratify=y ensures the same fraud ratio in both sets — critical for imbalanced data!")


# ══════════════════════════════════════════════════════════
# STEP 6 — SMOTE (Synthetic Minority Oversampling)
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 6 — Applying SMOTE to training set ONLY")
print("═"*60)

print(f"\n  Before SMOTE — Train fraud : {y_train.sum():,} / {len(y_train):,}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"  After  SMOTE — Train fraud : {y_train_sm.sum():,} / {len(y_train_sm):,}")
print(f"\n  SMOTE creates SYNTHETIC fraud examples by interpolating between")
print(f"  existing fraud cases — it only touches the TRAINING set.")
print(f"  The test set stays untouched (real-world distribution).")

# Visualise SMOTE effect
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Step 6 — Class Balance Before and After SMOTE", fontsize=13, fontweight="bold")
for ax, (title, labels, counts) in zip(axes, [
    ("Before SMOTE", ["Legit", "Fraud"], [np.sum(y_train==0), np.sum(y_train==1)]),
    ("After SMOTE",  ["Legit", "Fraud"], [np.sum(y_train_sm==0), np.sum(y_train_sm==1)])
]):
    bars = ax.bar(labels, counts, color=[C_LEGIT, C_FRAUD], width=0.45, edgecolor="white")
    ax.set_title(title, fontsize=11); ax.set_ylabel("Count")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, v+200, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_step6_smote.png", dpi=130, bbox_inches="tight")
plt.show()
print("  Saved → plot_step6_smote.png")


# ══════════════════════════════════════════════════════════
# STEP 7 — TRAIN MODELS
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 7 — Training models")
print("═"*60)

# Model A: Logistic Regression (simple, interpretable baseline)
print("\n  Training Logistic Regression …")
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train_sm, y_train_sm)
print("  ✓ Logistic Regression trained")

# Model B: Random Forest (stronger, handles non-linearity)
print("  Training Random Forest …")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=12,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)
print("  ✓ Random Forest trained")


# ══════════════════════════════════════════════════════════
# STEP 8 — EVALUATE: WHY ACCURACY IS MISLEADING
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 8 — Evaluation: accuracy vs precision/recall")
print("═"*60)

naive_acc = (y_test == 0).mean()
print(f"""
  A "predict everything as Legitimate" dummy model achieves:
  Accuracy = {naive_acc*100:.2f}% — sounds great, catches 0 fraud!

  This is why we use:
  • Recall    — of all real fraud cases, what % did we catch?
  • Precision — of all our fraud alerts, what % were actually fraud?
  • F1-Score  — harmonic mean of precision and recall
  • ROC-AUC   — overall discrimination ability (1.0 = perfect)
""")

for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
    y_pred = model.predict(X_test)
    print(f"\n  ── {name} ──")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"  ROC-AUC: {auc:.4f}")


# ══════════════════════════════════════════════════════════
# STEP 9 — CONFUSION MATRICES
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 9 — Confusion matrices")
print("═"*60)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Step 9 — Confusion Matrices", fontsize=13, fontweight="bold")

for ax, (name, model) in zip(axes, [("Logistic Regression", lr_model), ("Random Forest", rf_model)]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    tn, fp, fn, tp = cm.ravel()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Greens",
                xticklabels=["Legit (pred)", "Fraud (pred)"],
                yticklabels=["Legit (actual)", "Fraud (actual)"])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    ax.set_title(f"{name}  |  AUC={auc:.4f}", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    print(f"\n  {name}:")
    print(f"    True Positives  (fraud caught)  : {tp}")
    print(f"    False Negatives (fraud missed)  : {fn}  ← minimise this!")
    print(f"    False Positives (false alarms)  : {fp}")
    print(f"    True Negatives  (legit correct) : {tn}")

plt.tight_layout()
plt.savefig("plot_step9_confusion.png", dpi=130, bbox_inches="tight")
plt.show()
print("\n  Saved → plot_step9_confusion.png")


# ══════════════════════════════════════════════════════════
# STEP 10 — ROC CURVES
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 10 — ROC Curves (Receiver Operating Characteristic)")
print("═"*60)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.set_title("Step 10 — ROC Curves: Both Models vs Random", fontsize=12, fontweight="bold")

for name, model, color in [
    ("Logistic Regression", lr_model, C_FRAUD),
    ("Random Forest",       rf_model, C_BLUE),
]:
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.4f})", color=color, linewidth=2.5)
    print(f"  {name}: AUC = {auc:.4f}")

ax.plot([0, 1], [0, 1], "--", color="#aaa", linewidth=1.2, label="Random classifier (AUC = 0.50)")
ax.fill_between([0,1],[0,1],[0,1], alpha=0.04, color="#aaa")
ax.set_xlabel("False Positive Rate (false alarms)", fontsize=11)
ax.set_ylabel("True Positive Rate (fraud caught)", fontsize=11)
ax.legend(fontsize=10); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig("plot_step10_roc.png", dpi=130, bbox_inches="tight")
plt.show()
print("  Saved → plot_step10_roc.png")
print("""
  HOW TO READ THE ROC CURVE:
  • X-axis = False Positive Rate (how often do we wrongly flag legit txns?)
  • Y-axis = True Positive Rate  (how often do we correctly catch fraud?)
  • AUC > 0.95 = excellent for real-world fraud detection
  • The closer the curve hugs the top-left corner, the better
""")


# ══════════════════════════════════════════════════════════
# STEP 11 — PRECISION-RECALL CURVE
# ══════════════════════════════════════════════════════════
print("═"*60)
print("  STEP 11 — Precision-Recall Curve")
print("═"*60)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.set_title("Step 11 — Precision-Recall Curves", fontsize=12, fontweight="bold")

for name, model, color in [
    ("Logistic Regression", lr_model, C_FRAUD),
    ("Random Forest",       rf_model, C_BLUE),
]:
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax.plot(recall, precision, label=f"{name}  (AP = {ap:.4f})", color=color, linewidth=2.5)
    print(f"  {name}: Average Precision = {ap:.4f}")

baseline = y_test.mean()
ax.axhline(y=baseline, linestyle="--", color="#aaa", linewidth=1.2,
           label=f"Baseline (fraud rate = {baseline:.4f})")
ax.set_xlabel("Recall (fraction of actual fraud caught)", fontsize=11)
ax.set_ylabel("Precision (fraction of alerts that are real fraud)", fontsize=11)
ax.legend(fontsize=10); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig("plot_step11_pr_curve.png", dpi=130, bbox_inches="tight")
plt.show()
print("  Saved → plot_step11_pr_curve.png")
print("""
  WHY PRECISION-RECALL MATTERS FOR FRAUD:
  • With imbalanced data (0.17% fraud), ROC can look great even for bad models.
  • Precision-Recall directly shows the tradeoff:
    - High Recall  = catch more fraud, but get more false alarms
    - High Precision = fewer false alarms, but might miss some fraud
  • A bank typically prioritises HIGH RECALL (missing fraud is expensive!)
""")


# ══════════════════════════════════════════════════════════
# STEP 12 — FEATURE IMPORTANCE (Random Forest)
# ══════════════════════════════════════════════════════════
print("═"*60)
print("  STEP 12 — Feature Importance (Random Forest)")
print("═"*60)

importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = ["#C94A2A" if v > 0.05 else "#E8956D" if v > 0.02 else "#F2C4A8" for v in feat_df["Importance"]]
ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors_bar, edgecolor="white", height=0.65)
for i, (val, _) in enumerate(zip(feat_df["Importance"], feat_df["Feature"])):
    ax.text(val + 0.0005, i, f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title("Step 12 — Top 15 Feature Importances (Random Forest)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("plot_step12_feature_importance.png", dpi=130, bbox_inches="tight")
plt.show()
print("  Saved → plot_step12_feature_importance.png")

print("\n  Top 10 most important features:")
top10 = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:10]
for rank, (feat, imp) in enumerate(top10, 1):
    bar = "█" * int(imp * 400)
    print(f"  {rank:2}. {feat:<16} {bar} {imp:.4f}")


# ══════════════════════════════════════════════════════════
# STEP 13 — LIVE PREDICTION ON REAL TRANSACTIONS
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 13 — Live prediction on real transactions from the dataset")
print("═"*60)

# Pick one real fraud and one real legit row from the TEST set
fraud_indices  = np.where(y_test == 1)[0]
legit_indices  = np.where(y_test == 0)[0]

fraud_sample   = X_test[fraud_indices[0]]
legit_sample   = X_test[legit_indices[0]]

fraud_proba_lr = lr_model.predict_proba(fraud_sample.reshape(1, -1))[0][1]
fraud_proba_rf = rf_model.predict_proba(fraud_sample.reshape(1, -1))[0][1]
legit_proba_lr = lr_model.predict_proba(legit_sample.reshape(1, -1))[0][1]
legit_proba_rf = rf_model.predict_proba(legit_sample.reshape(1, -1))[0][1]

def risk_label(prob):
    if prob > 0.70: return "🚨 HIGH RISK — BLOCK"
    elif prob > 0.40: return "⚠️  MEDIUM RISK — REVIEW"
    else: return "✅ LOW RISK — CLEAR"

print(f"""
  ┌─────────────────────────────────────────────────┐
  │  TRANSACTION A  (actual label: FRAUD)           │
  ├─────────────────────────────────────────────────┤
  │  Logistic Regression fraud prob : {fraud_proba_lr:.4f}         │
  │  Random Forest       fraud prob : {fraud_proba_rf:.4f}         │
  │  Decision (RF)  : {risk_label(fraud_proba_rf):<30} │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │  TRANSACTION B  (actual label: LEGITIMATE)      │
  ├─────────────────────────────────────────────────┤
  │  Logistic Regression fraud prob : {legit_proba_lr:.4f}         │
  │  Random Forest       fraud prob : {legit_proba_rf:.4f}         │
  │  Decision (RF)  : {risk_label(legit_proba_rf):<30} │
  └─────────────────────────────────────────────────┘
""")


# ══════════════════════════════════════════════════════════
# STEP 14 — THRESHOLD TUNING
# ══════════════════════════════════════════════════════════
print("═"*60)
print("  STEP 14 — Threshold tuning (finding the best decision boundary)")
print("═"*60)

print("""
  By default, models predict fraud when probability > 0.50.
  But for fraud detection, we often LOWER the threshold to catch more fraud
  at the cost of more false alarms.

  Threshold | Recall (fraud caught) | Precision (accuracy of alerts)
  ──────────┼───────────────────────┼────────────────────────────────""")

rf_proba = rf_model.predict_proba(X_test)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

best_f1 = 0; best_thresh = 0.5
for thresh in thresholds:
    preds = (rf_proba >= thresh).astype(int)
    tp = np.sum((preds==1) & (y_test==1))
    fp = np.sum((preds==1) & (y_test==0))
    fn = np.sum((preds==0) & (y_test==1))
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    marker    = " ← best F1" if f1 > best_f1 else ""
    if f1 > best_f1: best_f1 = f1; best_thresh = thresh
    print(f"    {thresh:.1f}      |  {recall*100:6.2f}%               |  {precision*100:6.2f}%  {marker}")

print(f"\n  Best threshold = {best_thresh}  |  Best F1 = {best_f1:.4f}")


# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*58 + "╗")
print("║   FINAL SUMMARY                                          ║")
print("╠" + "═"*58 + "╣")

rf_auc   = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
lr_auc   = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1])
rf_preds = rf_model.predict(X_test)
tp       = np.sum((rf_preds==1) & (y_test==1))
fn       = np.sum((rf_preds==0) & (y_test==1))

print(f"║  Dataset          : {len(df):,} transactions                   ║")
print(f"║  Fraud rate       : {df['Class'].mean()*100:.3f}%                              ║")
print(f"║  SMOTE applied    : Training set balanced 50/50             ║")
print(f"║                                                            ║")
print(f"║  Logistic Regression AUC : {lr_auc:.4f}                       ║")
print(f"║  Random Forest AUC       : {rf_auc:.4f}  ← use this one       ║")
print(f"║                                                            ║")
print(f"║  Fraud cases in test : {y_test.sum()}                              ║")
print(f"║  Caught by RF        : {tp}  ({tp/y_test.sum()*100:.1f}% recall)              ║")
print(f"║  Missed by RF        : {fn}                                   ║")
print(f"║                                                            ║")
print(f"║  Output plots        : plot_step2 through plot_step12     ║")
print("╚" + "═"*58 + "╝")

print("""
  CONCEPTS YOU JUST APPLIED:
  ✓ Exploratory Data Analysis (EDA)
  ✓ Feature scaling (StandardScaler)
  ✓ Stratified train/test split
  ✓ SMOTE oversampling for class imbalance
  ✓ Logistic Regression (baseline model)
  ✓ Random Forest (ensemble model)
  ✓ Confusion matrix interpretation
  ✓ ROC-AUC curve
  ✓ Precision-Recall curve
  ✓ Feature importance
  ✓ Threshold tuning
  ✓ Live inference on real transactions

  NEXT STEPS TO GO FURTHER:
  → Try XGBoost/LightGBM (usually best on tabular fraud data)
  → Try ADASYN instead of SMOTE
  → Use cross-validation instead of single train/test split
  → Add SHAP values for explainability (pip install shap)
  → Deploy as a FastAPI endpoint
""")