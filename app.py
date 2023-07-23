import base64
import io
import os
import time

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from models import models_dict

def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = data

def preprocess_data(data, target_col, scaler_type, encoding_columns=None, drop_columns=None, missing_value_handling=None):
    preprocessed_data = data.copy()

    if drop_columns:
        preprocessed_data = preprocessed_data.drop(columns=drop_columns)

    if encoding_columns:
        le = LabelEncoder()
        for col in encoding_columns:
            if col != target_col:
                preprocessed_data[col] = le.fit_transform(preprocessed_data[col])

    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()

    if scaler_type != "None":
        preprocessed_data[preprocessed_data.columns.drop(target_col)] = scaler.fit_transform(
            preprocessed_data[preprocessed_data.columns.drop(target_col)])

    if missing_value_handling:
        preprocessed_data = handle_missing_values(preprocessed_data, missing_value_handling)

    return preprocessed_data
def dataframe_to_csv_bytes(df):
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer



def handle_missing_values(data, method):
    if method == 'drop':
        data = data.dropna()
    elif method == 'mean':
        data = data.fillna(data.mean())
    elif method == 'median':
        data = data.fillna(data.median())
    elif method == 'mode':
        data = data.fillna(data.mode().iloc[0])
    return data

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    try:

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_test == y_pred)
        duration = time.time() - start
        return accuracy, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred), duration
    except:
        return

st.set_page_config(layout="wide")

def reset_session_state():
    st.session_state.uploaded_data = None
    st.session_state.modified_data = None
    st.session_state.preprocessed_data = None

@st.cache_data
def get_scaler_options():
    return ["None", "StandardScaler", "MinMaxScaler"]

def main():
    st.title("Machine Learning Classification App")
    st.sidebar.title("Settings")

    reset_button = st.sidebar.button("Reset")
    if reset_button:
        reset_session_state()

    load_data()
    data = st.session_state.uploaded_data if 'uploaded_data' in st.session_state else None

    if data is not None:
        target_col = st.sidebar.selectbox("Select the target column", options=data.columns, index=len(data.columns) - 1)
        show_pp: bool = st.sidebar.checkbox('Show Pairplot')

        st.sidebar.subheader("Data Preprocessing")

        drop_columns_option = st.sidebar.checkbox('Drop specific columns')
        if drop_columns_option:
            drop_columns = st.sidebar.multiselect("Select columns to drop", options=data.columns, default=None)
        else:
            drop_columns = None

        encoding_option = st.sidebar.checkbox("Encode specific columns")
        if encoding_option:
            encoding_columns = st.sidebar.multiselect("Select columns to encode", options=data.columns, default=None)
        else:
            encoding_columns = None

        scaler_type = st.sidebar.selectbox("Select data scaling method",
                                           options=["None", "StandardScaler", "MinMaxScaler"])

        st.sidebar.subheader("Handle Missing Values")
        missing_value_handling = st.sidebar.selectbox("Select method for handling missing values",
                                                  options=["None", "drop", "mean", "median", "mode"])

        preprocessed_data = preprocess_data(data, target_col, scaler_type, encoding_columns, drop_columns,
                                            missing_value_handling)
        st.session_state.modified_data = preprocessed_data


        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset")
            st.write(data.head())
        with col2:
            st.subheader("Processed Dataset")
            st.write(st.session_state.modified_data.head())

            csv_buffer = dataframe_to_csv_bytes(st.session_state.modified_data)

            with st.container():
                st.write('')
                col1, col2 = st.columns([1, 4])
                with col2:
                    st.download_button(
                        label="Download preprocessed dataset",
                        data=csv_buffer,
                        file_name="preprocessed_dataset.csv",
                        mime="text/csv",
                    )

        if show_pp:
            st.subheader('Pairplot')
            pp_fig = sns.pairplot(st.session_state.modified_data)
            st.pyplot(pp_fig)

        st.sidebar.subheader("Model Selection")
        classifiers = ['Benchmark All Algorithms'] + list(models_dict.keys())
        classifier = st.sidebar.selectbox("Select a classifier", options=classifiers)

        if st.sidebar.button("Train and Evaluate"):
            preprocessed_data = st.session_state.modified_data
            if not np.issubdtype(preprocessed_data[target_col].dtype, np.integer):
                le = LabelEncoder()
                preprocessed_data[target_col] = le.fit_transform(preprocessed_data[target_col])

            X = preprocessed_data.drop(target_col, axis=1)
            y = preprocessed_data[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if classifier != "Benchmark All Algorithms":
                models = {classifier: models_dict[classifier]}
            else:
                models = models_dict

            results_expander = st.expander("Classifier Results", expanded=True)
            with results_expander:
                results = {}

                if classifier == "Benchmark All Algorithms":
                    progress_bar = st.progress(0)
                    total_models = len(models)
                    current_model = 0

                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(train_and_evaluate_model, model, X_train, y_train, X_test, y_test): name
                        for name, model in models.items()
                    }

                    for future in as_completed(futures):
                        if future:
                            name = futures[future]
                            accuracy, report, cm, duration = future.result()
                            results[name] = (accuracy, report, cm, duration)

                            if classifier == "Benchmark All Algorithms":
                                current_model += 1
                                progress_bar.progress(current_model / total_models)

                if classifier == "Benchmark All Algorithms":
                    accuracies = [results[name][0] for name in models.keys()]
                    durations = [results[name][3] for name in models.keys()]

                    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=False)

                    # Plot Accuracies
                    sns.barplot(y=[m for m in models], x=accuracies, ax=ax1)
                    ax1.set_xlim([0, max(accuracies) + 0.1])
                    ax1.set_xlabel("Accuracy")
                    ax1.set_ylabel("Classifier")
                    ax1.set_title("Classifier Accuracy Comparison")

                    for i, accuracy in enumerate(accuracies):
                        ax1.text(accuracy + 0.005, i, f"{accuracy:.4f}", ha="left", va="center", fontsize=10)

                    # Plot Durations
                    sns.barplot(y=[m for m in models], x=durations, ax=ax2)
                    ax2.set_xlim([0, max(durations) + 0.1])
                    ax2.set_xlabel("Duration (s)")
                    ax2.set_ylabel("Classifier")
                    ax2.set_title("Classifier Duration Comparison")

                    for i, duration in enumerate(durations):
                        ax2.text(duration + 0.005, i, f"{duration:.4f}", ha="left", va="center", fontsize=10)

                    fig.tight_layout()
                    st.write(fig)

                for name in models.keys():
                    accuracy, report, cm, duration = results[name]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"Classifier: {name}")
                        st.write("Classification Report:")
                        st.text("\n" + report)
                    with col2:
                        st.subheader("Confusion Matrix:")
                        st.write(cm)



print('cores: ', os.cpu_count())
if __name__ == '__main__':
    print('cores: ', os.cpu_count())
    main()
