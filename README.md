# 🚀 Adversarial Robustness in Digit Classification

This project implements an end-to-end machine learning pipeline to investigate how "noisy" data impacts model accuracy and how to programmatically recover lost information. Using the classic **Scikit-Learn Digits Dataset**, we simulate adversarial environments and employ **Principal Component Analysis (PCA)** as a primary remediation strategy.

## 📊 Project Overview
In real-world applications, data is rarely perfect. This analysis explores the **"Garbage In, Garbage Out"** principle by stress-testing three distinct machine learning models against catastrophic noise interference.

### **The Three-Stage Pipeline**
1.  **Baseline Development:** Training on clean, $8\times8$ pixel grayscale images to establish peak performance.
2.  **Adversarial Simulation:** "Poisoning" the dataset with high-scale Gaussian noise (scale 10.0) to bury the digit signal.
3.  **Denoising & Recovery:** Using PCA to retain 80% of variance and filtering out the high-frequency random noise.

---

## 🛠️ Model Architectures Compared
We selected three classifiers with different mathematical philosophies to compare their inherent robustness:

* **Gaussian Naive Bayes (GNB):** A simple probabilistic baseline that treats every pixel independently.
* **K-Nearest Neighbors (KNN):** A distance-based model ($K=5$) relying on visual similarity in 64-dimensional Euclidean space.
* **Multi-Layer Perceptron (MLP):** A feedforward neural network with a (200, 100) hidden layer structure for learning non-linear interactions.

---

## 📈 Key Results
| Condition | GNB Accuracy | KNN Accuracy | MLP Accuracy |
| :--- | :--- | :--- | :--- |
| **Clean Baseline** | ~85-90% | **>95%** | **>95%** |
| **Poisoned (Noise 10.0)** | ~10-20% | ~10-20% | ~10-20% |
| **Denoised (PCA)** | ~82% | **~94%** | **~95%** |



### **Denoising Strategy**
* **PCA Variance Retention:** Strategic choice to keep only the top 80% of variance to filter random noise.
* **Data Clipping:** Applied `np.clip` to force mathematical reconstructions into the valid [0, 16] grayscale range, essential for distance-based KNN stability.

---

## 💻 Installation & Usage
### **Prerequisites**
* Python 3.x
* Scikit-Learn
* NumPy
* Matplotlib

### **Run the Analysis**
Clone the repository and execute the main script to generate the comparison graphs:
```bash
git clone (https://github.com/akshagg/-Adversarial-Robustness-in-Digit-Classification)
cd project-fall2025-akshagg
python main.py
