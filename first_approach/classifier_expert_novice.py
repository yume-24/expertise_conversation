import pandas as pd
import pandas as py
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance




#python build_feature_dataframe.py
#python classifier_expert_novice.py
################################
#config stuff and loading data #
################################
DATA_PATH = Path("outputs/features_for_classification.csv")
RANDOM_STATE = 42
N_SPLITS = 5

df = pd.read_csv(DATA_PATH)

y = df['label']
X = df.drop(columns=["label", "Unnamed: 0"], errors="ignore")

print('feature matrix:', X.shape)


#feature groups
graph_features = [
    c for c in X.columns
    if c.startswith(("n_", "avg_", "density", "weak_", "strong_", "reciprocity", "assortativity", "diameter"))
]

embedding_features = [
    c for c in X.columns
    if c.startswith(("emb_", "pca_", "avg_pairwise"))
]

hybrid_features = [
    c for c in X.columns
    if c.startswith(("mean_edge_semantic", "std_edge_semantic", "long_edge", "semantic_betweenness"))
]

feature_sets = {
    "Graph-only": graph_features,
    "Embedding-only": embedding_features,
    "Hybrid-only": hybrid_features,
    "Combined": graph_features + embedding_features + hybrid_features
}

for name, feats in feature_sets.items():
    print(f"{name}: {len(feats)} features")





#####################
#MODEL STUFFFFFFFFF##
#####################

models ={
    'Logreg': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ]),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
}


#cross validation

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
results = []

for feat_name, feats in feature_sets.items():
    X_sub = X[feats].fillna(0)

    for model_name, model in models.items():
        auc_scores = cross_val_score(
            model, X_sub, y,
            cv=cv,
            scoring="roc_auc"
        )
        f1_scores = cross_val_score(
            model, X_sub, y,
            cv=cv,
            scoring="f1"
        )

        results.append({
            "Feature set": feat_name,
            "Model": model_name,
            "ROC-AUC mean": auc_scores.mean(),
            "ROC-AUC std": auc_scores.std(),
            "F1 mean": f1_scores.mean(),
            "F1 std": f1_scores.std()
        })

results_df = pd.DataFrame(results)
print("\n=== Cross-validated results ===")
print(results_df)


#performance comparision visualization
plt.figure()
for model_name in models.keys():
    subset = results_df[results_df["Model"] == model_name]
    plt.bar(
        subset["Feature set"] + " (" + model_name + ")",
        subset["ROC-AUC mean"]
    )

plt.xticks(rotation=45, ha="right")
plt.ylabel("ROC-AUC")
plt.title("Expert vs Novice Classification Performance")
plt.tight_layout()
plt.show()

# ------------------
# FIT FINAL MODEL (Combined + Logistic Regression)
# ------------------
best_feats = feature_sets["Combined"]
X_best = X[best_feats].fillna(0)

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

final_model.fit(X_best, y)


# ------------------
# VISUALIZATION 2: LOGISTIC REGRESSION COEFFICIENTS
# ------------------
coefs = final_model.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": best_feats,
    "coef": coefs
}).sort_values(by="coef", key=abs, ascending=False)

plt.figure(figsize=(6, 8))
plt.barh(coef_df["feature"][:20], coef_df["coef"][:20])
plt.gca().invert_yaxis()
plt.title("Top Logistic Regression Coefficients (Combined)")
plt.xlabel("Coefficient value")
plt.tight_layout()
plt.show()


# ------------------
# VISUALIZATION 3: PERMUTATION IMPORTANCE (MODEL-AGNOSTIC)
# ------------------
rf = models["RandomForest"]
rf.fit(X_best, y)

perm = permutation_importance(
    rf, X_best, y,
    n_repeats=20,
    random_state=RANDOM_STATE,
    scoring="roc_auc"
)

imp_df = pd.DataFrame({
    "feature": best_feats,
    "importance": perm.importances_mean
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(6, 8))
plt.barh(imp_df["feature"][:20], imp_df["importance"][:20])
plt.gca().invert_yaxis()
plt.title("Permutation Importance (Random Forest)")
plt.xlabel("Mean decrease in ROC-AUC")
plt.tight_layout()
plt.show()







