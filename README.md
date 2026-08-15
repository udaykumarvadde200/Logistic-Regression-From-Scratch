# 📊 Student Placement Prediction — Logistic Regression From Scratch

> **A hands-on machine learning project implementing Perceptron and Logistic Regression from scratch, then benchmarking them against Scikit-learn.**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Implemented%20From%20Scratch-orange?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Model%20Comparison-F7931E?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 🎯 Project Objective

The goal of this project is to understand **binary classification and Logistic Regression at an implementation level**, rather than treating the model as a black box.

The project follows this progression:

```text
Perceptron
    ↓
Linear Classification
    ↓
Logistic Regression
    ↓
Sigmoid Function
    ↓
Binary Cross-Entropy
    ↓
Gradient Descent
    ↓
Scikit-learn Implementation
    ↓
Model Comparison
```

The final models predict whether a student will be **placed or not placed** based on academic and career-related information.

---

## 🧠 Models Implemented

### 1. Perceptron — From Scratch

A Perceptron is implemented manually to establish a baseline linear classifier.

```text
Input Features
      ↓
Weighted Sum
      ↓
Step Function
      ↓
0 / 1 Prediction
```

### 2. Logistic Regression — From Scratch

The complete learning process is implemented manually using:

- Linear combination
- Sigmoid activation
- Binary Cross-Entropy loss
- Analytical gradients
- Gradient Descent
- Probability-based predictions

Core equations:

$$
z = Xw + b
$$

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
L = -\frac{1}{n}\sum[y\log(p)+(1-y)\log(1-p)]
$$

$$
w = w - \alpha \frac{\partial L}{\partial w}
$$

$$
b = b - \alpha \frac{\partial L}{\partial b}
$$

### 3. Logistic Regression — Scikit-learn

The standard `LogisticRegression` implementation from Scikit-learn is trained using the same processed dataset to provide a practical benchmark.

---

## 📂 Dataset

The dataset contains **1,000 student records and 12 columns**.

### Features used

| Feature | Type | Description |
|---|---|---|
| `Branch` | Categorical | Student's engineering branch |
| `CGPA` | Numerical | College CGPA |
| `10th_percentage` | Numerical | 10th standard percentage |
| `12th_percentage` | Numerical | 12th standard percentage |
| `Project Level` | Ordinal | Beginner / Moderate / Advanced |
| `Internships` | Binary | Internship experience |
| `Certifications` | Binary | Certification status |
| `Backlogs` | Numerical | Number of backlogs |

### Removed columns

| Column | Reason |
|---|---|
| `Name` | Identifier |
| `Roll No` | Identifier |
| `Gender` | Excluded from this experiment |
| `Placed` | Target variable |

### Target

```text
0 → Not Placed
1 → Placed
```

---

## ⚙️ Machine Learning Pipeline

```text
Dataset
   │
   ▼
Data Understanding
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Cleaning
   │
   ▼
Feature Selection
   │
   ▼
Categorical Encoding
   │
   ▼
80/20 Train-Test Split
   │
   ▼
Feature Scaling
   │
   ├───────────────┬─────────────────────┐
   ▼               ▼                     ▼
Perceptron     Logistic Scratch     Logistic Sklearn
   │               │                     │
   └───────────────┴─────────────────────┘
                   │
                   ▼
            Model Evaluation
                   │
                   ▼
             Final Comparison
```

---

## 🔧 Preprocessing

### Binary Encoding

Binary categorical features are converted to numerical values:

```text
No  → 0
Yes → 1
```

### Ordinal Encoding

Project experience is represented in increasing order:

```text
Beginner → 0
Moderate → 1
Advanced → 2
```

### One-Hot Encoding

`Branch` is represented using one-hot encoded columns.

### Feature Scaling

Numerical/ordinal features are standardized using statistics calculated from the training set:

```text
CGPA
10th_percentage
12th_percentage
Backlogs
Project Level
```

The scaling formula is:

```text
x_scaled = (x - mean_train) / std_train
```

---

## 📈 Results

All three models were evaluated on the same **200 test samples**.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Perceptron | **88.00%** | **81.82%** | 69.23% | 75.00% | 94.61% |
| Logistic Regression — Scratch | 86.00% | 71.43% | **76.92%** | 74.07% | 94.80% |
| Logistic Regression — Scikit-learn | **88.00%** | 75.00% | **80.77%** | **77.78%** | **95.28%** |

### ⏱️ Training Time

| Model | Training Time |
|---|---:|
| Perceptron | 1.508756 sec |
| Logistic Regression — Scratch | 0.270272 sec |
| Logistic Regression — Scikit-learn | 0.043619 sec |

> Training time can vary depending on the machine and runtime environment.

---

## 🔍 Key Findings

### Perceptron

The Perceptron achieved **88% accuracy** and the highest precision for the `Placed` class.

However, its recall was only **69.23%**, meaning it missed a larger number of students who were actually placed.

### Logistic Regression — Scratch

The scratch implementation achieved **86% accuracy**.

Although its accuracy was slightly lower than the Perceptron, it achieved:

- **76.92% recall**
- **94.80% ROC-AUC**

This demonstrates why evaluating a classification model using only accuracy can be misleading.

### Logistic Regression — Scikit-learn

The Scikit-learn implementation produced the strongest overall results:

- **88% accuracy**
- **80.77% recall**
- **77.78% F1 score**
- **95.28% ROC-AUC**

It also trained significantly faster than the manually implemented models because the library implementation uses highly optimized numerical routines.

---

## 📌 Confusion Matrix Summary

### Perceptron

```text
[[140   8]
 [ 16  36]]
```

### Logistic Regression — Scratch

```text
[[132  16]
 [ 12  40]]
```

### Logistic Regression — Scikit-learn

```text
[[134  14]
 [ 10  42]]
```

For the `Placed` class, the Scikit-learn model correctly identified **42 out of 52** placed students.

---

## 🛠️ Tech Stack

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **Jupyter Notebook**

The core Perceptron and Logistic Regression learning logic is implemented manually using NumPy.

---

## 📁 Project Structure

```text
Logisticregression/
│
├── data/
│   └── placement.csv
│
├── Logistic_Regression_Project_Cleaned.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Logisticregression
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
Logistic_Regression_Project_Cleaned.ipynb
```

Then run the notebook from top to bottom.

---

## 📚 What I Learned

This project helped me understand that Logistic Regression is much more than calling:

```python
LogisticRegression().fit(X, y)
```

I implemented and understood the complete optimization process:

```text
Linear Score
     ↓
Sigmoid
     ↓
Probability
     ↓
Binary Cross-Entropy
     ↓
Gradient
     ↓
Gradient Descent
     ↓
Updated Weights
     ↓
Repeat
```

The project also reinforced the importance of evaluating classification models using multiple metrics rather than relying only on accuracy.

---

## 🔮 Future Work

Possible extensions:

- Hyperparameter tuning
- ROC curve and Precision-Recall curve visualization
- Threshold tuning
- Class imbalance techniques
- Decision Tree implementation from scratch
- Random Forest implementation
- Gradient Boosting / XGBoost from scratch
- Comparison with the original XGBoost paper implementation

---

## 👤 Author

**Uday Kumar**

Machine Learning / AI enthusiast building models from fundamentals and gradually moving toward production-oriented ML systems.

---

⭐ If you found this project useful, consider giving the repository a star!
"# Logistic-Regression-From-Scratch" 
