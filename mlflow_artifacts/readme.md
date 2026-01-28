## 📊 Model Evaluation & Results Interpretation

This section summarizes the performance of the Logistic Regression model used for credit default risk prediction. Given the highly imbalanced nature of credit datasets, evaluation focuses on **risk ranking behavior** rather than raw accuracy.

---

### 1️⃣ ROC Curve – Risk Ranking Capability

The ROC (Receiver Operating Characteristic) curve evaluates how well the model ranks applicants by default risk across all possible thresholds.

- The curve lies consistently above the random baseline, indicating the model captures **signal beyond chance**.
- The observed ROC-AUC is approximately **0.61**, which is modest but realistic for consumer credit default prediction.
- This level of AUC suggests the model can **order applicants by relative risk**, even though absolute default prediction remains challenging.

**Interpretation:**  
The model is suitable for **probability-based ranking** (e.g., percentiles or score bands) rather than strict approve/reject decisions using a single cutoff.

---

### 2️⃣ Confusion Matrix – Effect of Class Imbalance

At a default probability threshold of 0.5, the confusion matrix shows:

- **True Negatives:** 56,327  
- **False Positives:** 1  
- **False Negatives:** 5,015  
- **True Positives:** 0  

This reveals that:
- The model strongly favors predicting the majority class (non-default).
- Overall accuracy is high, but **recall for defaulters is effectively zero** at this threshold.
- This behavior is expected when modeling rare events such as loan defaults.

**Interpretation:**  
A fixed probability threshold is not appropriate for credit risk decisions. High accuracy here is misleading and does not reflect the model’s usefulness in identifying risky applicants.

---

### 🔑 Key Insight

- **Accuracy is not a meaningful success metric** for this problem.
- The model’s value lies in its **relative risk scores**, not binary predictions.
- By converting predicted probabilities into **percentile-based risk buckets**, the system enables:
  - Auto-approval for low-risk applicants
  - Manual review for medium-risk cases
  - Rejection or collateral checks for high-risk applicants

---

### ✅ Conclusion

This evaluation demonstrates why **simple, interpretable models combined with business-aligned decision logic** are effective in real-world credit systems. Rather than optimizing for misleading metrics, the project prioritizes **risk ordering, transparency, and operational usability**.

Evaluation artifacts (ROC curve and confusion matrix) were generated and tracked using **MLflow** and are included in this repository for reproducibility.

