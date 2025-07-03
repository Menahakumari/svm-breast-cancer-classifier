# svm-breast-cancer-classifier
# 🧠 SVM Binary Classification - Breast Cancer Detection

This project builds a Support Vector Machine (SVM) classifier to detect whether a tumor is malignant or benign using the Breast Cancer dataset. The pipeline includes data preparation, model training (Linear & RBF kernels), hyperparameter tuning, and performance evaluation.

---

## 📁 Dataset

- **Source**: `breast-cancer.csv`
- **Dataset**: Kaggle-('https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset')
- **Type**: Binary Classification (0 = Malignant, 1 = Benign)
- **Features**: Medical attributes of cell nuclei from breast mass images

---

## 📌 Project Tasks

1. Load and prepare the dataset
2. Encode categorical data
3. Train/test split and standardize features
4. Train SVM with:
   - Linear Kernel
   - RBF Kernel
5. Visualize decision boundaries (2D)
6. Tune hyperparameters (`C`, `gamma`)
7. Evaluate using:
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC-AUC
   - Cross-validation

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn

---
✅ Results
Linear Kernel Accuracy: ~96%

RBF Kernel Accuracy: ~98%

Best Parameters: C=1, gamma='scale'

Cross-validation Accuracy: ~91.2%

---

## ▶️ Run This Project

```bash
# Step 1: Clone repo
git clone https://github.com/yourusername/svm-breast-cancer-classifier.git
cd svm-breast-cancer-classifier

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the notebook
jupyter notebook notebooks/svm_breast_cancer_analysis.ipynb


