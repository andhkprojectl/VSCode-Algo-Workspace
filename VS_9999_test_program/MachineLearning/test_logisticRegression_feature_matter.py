from sklearn.metrics import roc_auc_score
import numpy as np

# Fit model with and without x1
logit_all = LogisticRegression().fit(X_train, y_train)
auc_all = roc_auc_score(y_test, logit_all.predict_proba(X_test)[:, 1])

logit_only_x1 = LogisticRegression().fit(X_train[['x1']], y_train)
auc_x1 = roc_auc_score(y_test, logit_only_x1.predict_proba(X_test[['x1']])[:, 1])

print("AUC with all features:", auc_all)
print("AUC with x1 only:", auc_x1)


# If auc_x1 ≈ 0.5, then even if the coefficient is 0.3, it’s not predictively important — just statistically fitted.
# 📈 AUC value interpretation:
# AUC	Meaning	Performance
# 1.0	Perfect classifier	✅ Excellent
# 0.9–1.0	Great separation	💪 Strong
# 0.7–0.9	Good separation	👍 Acceptable
# 0.5–0.7	Weak separation	⚠️ Poor
# 0.5	Random guessing	❌ No predictive power
# < 0.5	Worse than random	🚫 Predicts oppositely