# Model Comparison — Loan Approval Prediction

## Overview

Seven machine learning classifiers were trained and optimized using **5-fold Stratified Cross-Validation**, with the **F1-score** selected as the primary optimization metric to account for the class imbalance in the dataset. The final models were evaluated on a held-out **20% test set**, stratified on `loan_status` to preserve the original class distribution of **62.2% Approved** and **37.8% Rejected**.

---

## Results Table

| Model | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **Decision Tree** | **0.9998** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest | 0.9988 | 0.9988 | 0.9981 | 1.0000 | 0.9991 | 1.0000 |
| XGBoost | 0.9981 | 0.9965 | 0.9944 | 1.0000 | 0.9972 | 1.0000 |
| SVM | 0.9621 | 0.9614 | 0.9583 | 0.9838 | 0.9709 | 0.9952 |
| Naive Bayes | 0.9435 | 0.9403 | 0.9615 | 0.9474 | 0.9542 | 0.9780 |
| KNN | 0.9327 | 0.9192 | 0.9432 | 0.9427 | 0.9430 | 0.9824 |
| Logistic Regression | 0.9296 | 0.9145 | 0.9315 | 0.9394 | 0.9354 | 0.9737 |

> *Performance metrics are rounded from the notebook output. The complete evaluation results and classification reports are available in `notebooks/04_model_training_evaluation.ipynb`.*

---

## Best Hyperparameters

| Model | Best Parameters |
|---|---|
| Logistic Regression | `C=0.01`, `penalty='l2'` |
| Decision Tree | `max_depth=4`, `min_samples_split=10` |
| Random Forest | `n_estimators=100`, `max_depth=6`, `min_samples_split=5` |
| KNN | `n_neighbors=9`, `weights='distance'` |
| Naive Bayes | `var_smoothing=1e-09` |
| SVM | `C=10`, `kernel='rbf'` |
| XGBoost | `n_estimators=100`, `max_depth=5`, `learning_rate=0.05` |

---

## Final Model Selection

The **Decision Tree** model was selected as the final model because it achieved the highest cross-validated F1-score (**0.9998**) and perfect performance on the test dataset (**F1 = 1.0000**, **ROC-AUC = 1.0000**).

---

## Analysis of the High Model Performance

The perfect test performance was further investigated to verify that it resulted from meaningful feature relationships rather than data leakage or evaluation errors.

Feature importance analysis showed that the Decision Tree relied primarily on **`cibil_score`** (approximately **81% feature importance**), with additional decision splits based on **`loan_to_income_ratio`**, **`loan_term`**, and **`asset_to_loan_ratio`**. All of these are legitimate input features, and none are derived from the target variable (`loan_status`), indicating that **no data leakage** is present.

The exceptionally high performance is primarily a characteristic of this particular dataset. The loan approval labels are strongly influenced by **`cibil_score`**, making the decision boundary relatively easy for tree-based algorithms to learn. A simple threshold on `cibil_score` alone achieves approximately **95% accuracy**, while the additional decision rules enable the Decision Tree to classify all test samples correctly.

Random Forest and XGBoost produced nearly identical results, confirming the same underlying pattern using ensemble learning techniques.

In contrast, Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machine (SVM) achieved comparatively lower scores. These algorithms are less effective at modeling the threshold-like decision boundaries that tree-based models naturally capture.

---

## Practical Considerations

Although the evaluation results are excellent, they should be interpreted within the context of this dataset. This Kaggle dataset is widely used for educational purposes and exhibits unusually clear class separation. Consequently, the reported performance should **not** be interpreted as representative of real-world loan approval systems.

In practical applications, credit decisions are influenced by additional financial, behavioral, and economic factors, resulting in noisier and more complex datasets. Models trained on such data would likely achieve lower, although still meaningful, predictive performance.

---

## Why Decision Tree Was Selected

Decision Tree was chosen as the final model for the following reasons:

- It achieved the highest cross-validation and test performance among all evaluated models.
- With a maximum depth of **4**, the model remains highly interpretable, allowing its decision-making process to be easily visualized and explained.
- The model is simple to deploy while maintaining excellent predictive accuracy.
- Although Random Forest and XGBoost demonstrated nearly identical performance, ensemble models are generally more computationally expensive and less interpretable than a single Decision Tree.
- For larger and more complex real-world datasets, Random Forest or XGBoost may provide better generalization because of their robustness to unseen data.

---

## Saved Artifacts

The following files were saved for deployment:

- `models/trained_model.pkl` — Final trained Decision Tree model
- `models/scaler.pkl` — StandardScaler fitted on the training dataset
- `models/feature_columns.pkl` — Feature order required during prediction

---

## Visualizations

The following visualizations are included in the project:

- `images/day4_model_comparison.png` — Model comparison chart
- `images/day4_confusion_matrices.png` — Confusion matrices for all models
- `images/day4_roc_curves.png` — ROC curves comparing all classifiers

---

## Conclusion

The comparison of seven machine learning algorithms demonstrates that **tree-based models consistently outperform linear and distance-based models** for this loan approval dataset. Among all evaluated models, the **Decision Tree** achieved the best balance of predictive performance and interpretability, making it the most suitable choice for this project.

The trained model, preprocessing components, and feature configuration have been saved and are ready to be integrated into a Streamlit application for real-time loan approval prediction.