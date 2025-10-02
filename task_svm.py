# task7_svm.py
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# -----------------------------
# 1. Load dataset
# -----------------------------
data = load_breast_cancer()
X, y = data.data, data.target
print("Dataset shape:", X.shape)

# -----------------------------
# 2. Train/test split + scaling
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -----------------------------
# 3. Train Linear SVM
# -----------------------------
svc_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svc_lin.fit(X_train_s, y_train)
y_pred_lin = svc_lin.predict(X_test_s)
print("\n=== Linear SVM ===")
print(classification_report(y_test, y_pred_lin))

# -----------------------------
# 4. Train RBF SVM
# -----------------------------
svc_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svc_rbf.fit(X_train_s, y_train)
y_pred_rbf = svc_rbf.predict(X_test_s)
print("\n=== RBF SVM ===")
print(classification_report(y_test, y_pred_rbf))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(svc_rbf, X_test_s, y_test)
plt.title("Confusion Matrix (RBF SVM)")
plt.savefig("plots/conf_matrix_rbf.png", dpi=150)
plt.close()

# -----------------------------
# 5. Cross-validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "f1", "roc_auc"]

scores = cross_validate(
    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
    X_train_s,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
)

print("\n=== Cross-validation (RBF, C=1, gamma=scale) ===")
print("Mean accuracy:", scores["test_accuracy"].mean())
print("Mean F1:", scores["test_f1"].mean())
print("Mean ROC AUC:", scores["test_roc_auc"].mean())

# -----------------------------
# 6. Hyperparameter tuning
# -----------------------------
param_grid_rbf = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": [1e-3, 1e-2, 0.1, "scale", "auto"],
}
grid_rbf = GridSearchCV(
    SVC(kernel="rbf"), param_grid_rbf, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_rbf.fit(X_train_s, y_train)
print("\nBest RBF params:", grid_rbf.best_params_)

# Evaluate best model
best_rbf = grid_rbf.best_estimator_
y_pred_best = best_rbf.predict(X_test_s)
print("\n=== Tuned RBF SVM ===")
print(classification_report(y_test, y_pred_best))

# -----------------------------
# 7. Visualization - synthetic 2D dataset
# -----------------------------
X2, y2 = make_classification(
    n_samples=400,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=42,
)
sc2 = StandardScaler()
X2s = sc2.fit_transform(X2)

clf_lin2 = SVC(kernel="linear").fit(X2s, y2)
clf_rbf2 = SVC(kernel="rbf", gamma=1.0, C=1.0).fit(X2s, y2)

def plot_decision_boundary(clf, X, y, title, fname):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=30)
    plt.title(title)
    plt.savefig(fname, dpi=150)
    plt.close()

plot_decision_boundary(clf_lin2, X2s, y2, "Linear SVM (Synthetic)", "plots/linear_svm.png")
plot_decision_boundary(clf_rbf2, X2s, y2, "RBF SVM (Synthetic)", "plots/rbf_svm.png")

# -----------------------------
# 8. Visualization - PCA projection of real dataset
# -----------------------------
pca = PCA(n_components=2)
Xp = pca.fit_transform(X_train_s)
clf_pca = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xp, y_train)
plot_decision_boundary(clf_pca, Xp, y_train, "RBF SVM on PCA Projection", "plots/pca_rbf.png")

# -----------------------------
# 9. Save models
# -----------------------------
joblib.dump(svc_lin, "models/svc_linear.joblib")
joblib.dump(best_rbf, "models/svc_best_rbf.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\nAll tasks completed. Models & plots saved in folders.")
