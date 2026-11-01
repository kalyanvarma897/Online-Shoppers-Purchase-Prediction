import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ML Model Comparison", layout="wide")

st.title("Online Shoppers Purchase Prediction")
st.write("Upload test data and select a machine learning model.")

model_options = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_models():
    models = {
        name: joblib.load(f"model/{filename}")
        for name, filename in model_options.items()
    }
    preprocessor = joblib.load("model/preprocessing.joblib")
    return models, preprocessor


models, preprocessor = load_models()

selected_model = st.selectbox("Select a Model", list(model_options.keys()))

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "Revenue" not in data.columns:
        st.error("The uploaded CSV must contain the Revenue column.")
    else:
        X = data.drop(columns=["Revenue"])
        y = data["Revenue"].astype(bool)

        try:
            X_processed = preprocessor.transform(X)
            model = models[selected_model]

            predictions = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)[:, 1]

            from sklearn.metrics import (
                accuracy_score,
                roc_auc_score,
                precision_score,
                recall_score,
                f1_score,
                matthews_corrcoef,
                confusion_matrix,
                classification_report,
            )

            st.subheader("Model Evaluation")

            metrics = {
                "Accuracy": accuracy_score(y, predictions),
                "AUC": roc_auc_score(y, probabilities),
                "Precision": precision_score(y, predictions, zero_division=0),
                "Recall": recall_score(y, predictions, zero_division=0),
                "F1 Score": f1_score(y, predictions, zero_division=0),
                "MCC": matthews_corrcoef(y, predictions),
            }

            columns = st.columns(3)

            for index, (name, value) in enumerate(metrics.items()):
                columns[index % 3].metric(name, f"{value:.4f}")

            st.subheader("Confusion Matrix")
            st.write(
                pd.DataFrame(
                    confusion_matrix(y, predictions),
                    index=["Actual No Purchase", "Actual Purchase"],
                    columns=["Predicted No Purchase", "Predicted Purchase"],
                )
            )

            st.subheader("Classification Report")
            st.text(
                classification_report(
                    y,
                    predictions,
                    target_names=["No Purchase", "Purchase"],
                    zero_division=0,
                )
            )

        except Exception as error:
            st.error(f"Error processing the uploaded file: {error}")
