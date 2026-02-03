# Sampling Assignment: Credit Card Fraud Detection

## Project Overview
This project investigates the impact of various sampling techniques on the performance of machine learning models when dealing with highly imbalanced datasets.The objective is to understand how different sampling strategies influence the accuracy of models like Logistic Regression, Decision Trees, Random Forest, SVM, and KNN.

## Dataset
* **Source:** [Credit Card Fraud Detection dataset](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv)
* **Original Distribution:** The original dataset contained 772 rows and was highly imbalanced.
* **Balancing Technique:** To address the imbalance, the majority class was undersampled to match the minority class, resulting in a perfectly balanced dataset of 18 rows (9 Class 0, 9 Class 1).

## Methodology
The project follows these steps:
1.  **Balancing:** Convert the imbalanced dataset into a balanced one.
2.  **Sampling:** Create five different samples using the following techniques:
    * Simple Random Sampling
    * Systematic Sampling
    * Stratified Sampling
    * Cluster Sampling
    * Bootstrap Sampling
3.  **Modeling:** Train and evaluate five machine learning models on each sample:
    * **M1:** Logistic Regression
    * **M2:** Decision Tree Classifier
    * **M3:** Random Forest Classifier
    * **M4:** Support Vector Machine (SVM)
    * **M5:** K-Nearest Neighbors (KNN)
4.  **Comparison:** Determine which sampling technique provides higher accuracy for each model.

## Results
The following table summarizes the accuracy achieved by each model using different sampling techniques:

| Model | Simple Random | Systematic | Stratified | Cluster | Bootstrap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1 (Logistic Regression)** | 0.75 | 0.50 | 0.50 | 0.50 | **1.00** |
| **M2 (Decision Tree)** | 0.50 | **1.00** | 0.75 | 0.25 | **1.00** |
| **M3 (Random Forest)** | 0.50 | 0.75 | 0.50 | 0.25 | **1.00** |
| **M4 (SVM)** | **0.75** | 0.50 | **0.75** | 0.50 | 0.50 |
| **M5 (KNN)** | **0.75** | 0.50 | 0.25 | 0.50 | 0.50 |

## Discussion
Based on the experimental results, we observed distinct patterns in how sampling affects model performance:

1.  **Bootstrap Sampling Dominance:** Bootstrap sampling proved to be the most effective technique for **Logistic Regression** and **Random Forest**, achieving a perfect accuracy score of **1.0**. This suggests that for small, balanced datasets, the resampling with replacement method helps these models generalize better by emphasizing critical data points.

2.  **Systematic Sampling Success:** The **Decision Tree** model achieved its highest accuracy (1.0) using Systematic Sampling (as well as Bootstrap). This indicates that the structured interval-based selection preserved the decision boundaries well for the tree classifier.

3.  **Simple Random & Stratified for Distance/Margin Models:**
    * **SVM** and **KNN** performed best with **Simple Random Sampling** (0.75).
    * Stratified sampling also performed equally well for SVM (0.75).
    * These models (M4 and M5) struggled significantly with Cluster and Stratified sampling in some instances (dropping as low as 0.25 accuracy), highlighting their sensitivity to how the feature space is represented in the sample.

4.  **Impact of Dataset Size:** It is important to note that the balanced dataset was small (18 instances). This limited size contributes to the high variance in accuracy (ranging from 0.25 to 1.0). In larger datasets, these variations would likely be less extreme, but the relative ranking of sampling techniques remains a valuable insight.

## Conclusion
The analysis demonstrates that **Bootstrap Sampling** and **Systematic Sampling** were the superior techniques for this specific dataset, particularly for tree-based and linear models. Meanwhile, **Simple Random Sampling** provided the most stable baseline for distance-based models like KNN and SVM.

## How to Run
1.  Clone this repository.
2.  Install dependencies: `pip install pandas numpy scikit-learn`
3.  Run the provided Python script or Jupyter Notebook to reproduce the results.
