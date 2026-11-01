# Online Shoppers Purchase Prediction

## 1. Project Overview

This project implements and compares five machine learning classification algorithms to predict whether an online shopping session results in a purchase.

The models are evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

A Streamlit web application is also developed to allow users to upload test data, select a trained model, and view its evaluation results.

## 2. Dataset Information

**Dataset Name:** Online Shoppers Purchasing Intention Dataset

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

- Original records: 12,330
- Original columns: 18
- Input features: 17
- Target variable: Revenue
- Duplicate records removed: 125
- Records after cleaning: 12,205

The target variable `Revenue` indicates whether a shopping session resulted in a purchase.

## 3. Machine Learning Algorithms

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing values.
- Removed duplicate records.
- Separated input features and target variable.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Split the dataset into 80% training data and 20% testing data.
- Used stratified sampling to maintain the target class distribution.

## 5. Model Evaluation Results

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8894 | 0.9022 | 0.7718 | 0.4162 | 0.5408 | 0.5142 |
| Decision Tree | 0.8865 | 0.8619 | 0.6431 | 0.6178 | 0.6302 | 0.5633 |
| KNN | 0.8796 | 0.8110 | 0.6803 | 0.4346 | 0.5304 | 0.4805 |
| Naive Bayes | 0.7108 | 0.8263 | 0.3288 | 0.8141 | 0.4684 | 0.3772 |
| SVM | 0.8931 | 0.8810 | 0.7510 | 0.4738 | 0.5811 | 0.5416 |

## 6. Results and Observations

- **Highest Accuracy:** SVM (0.8931)
- **Highest AUC:** Logistic Regression (0.9022)
- **Highest Precision:** Logistic Regression (0.7718)
- **Highest Recall:** Naive Bayes (0.8141)
- **Highest F1 Score:** Decision Tree (0.6302)
- **Highest MCC:** Decision Tree (0.5633)

Decision Tree provides the strongest balance between precision and recall based on its F1 Score and MCC.

## 7. Streamlit Application

The Streamlit application provides the following features:

- Upload a test dataset in CSV format.
- Select one of the five trained machine learning models.
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- View the confusion matrix.
- View the classification report.

The application loads previously trained models and the saved preprocessing object. It does not retrain models when a user uploads test data.

## 8. Project Structure

```text
Assignment2/
│
├── .ipynb_checkpoints/
│
├── model/
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── preprocessing.joblib
│
├── results/
│   └── model_comparison.csv
│
├── app.py
├── Assignment2.ipynb
├── online_shoppers_intention.csv
├── test_data.csv
├── requirements.txt
└── README.md
```

## 9. Installation and Execution

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal and upload `test_data.csv` to evaluate the selected model.

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 11. Conclusion

Five machine learning classification algorithms were implemented and evaluated using the Online Shoppers Purchasing Intention Dataset.

Their performance was compared using six evaluation metrics. The results demonstrate that different algorithms perform differently across evaluation measures. Decision Tree achieved the best F1 Score and MCC, while SVM achieved the highest accuracy.

The project also includes a Streamlit application that uses saved models to evaluate uploaded test data.