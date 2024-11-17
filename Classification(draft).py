import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go

st.sidebar.title("Machine Learning Model Comparison & Tuning")
mode = st.sidebar.radio("Choose Mode", ["Compare Models", "Tune Models"])

if mode == "Compare Models":
    st.subheader("Compare Models")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Dataset for Classification", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

    if uploaded_file:
        st.write("Dataset Preview")
        st.dataframe(data.head())

        # Assuming the last column is the target
        target_col = data.columns[-1]  # Last column as target
        feature_cols = data.columns[:-1]  # All other columns as features

        # Prepare features and target
        X = data[feature_cols]
        y = data[target_col]

        # Check if target variable contains missing values
        if y.isnull().sum() > 0:
            y = y.fillna(y.mean())  # Fill missing values

        # Handle categorical target for classification
        if y.dtype == 'object' or len(np.unique(y)) < 20:  # For classification
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            task = "Classification"

        # Handle missing values in features as well
        X.fillna(X.mean(), inplace=True)

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define classification models
        models = {
            "Decision Tree (CART)": DecisionTreeClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "AdaBoost (Gradient Boosting)": AdaBoostClassifier(),
            "K-Nearest Neighbors (K-NN)": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Multi-Layer Perceptron (MLP)": MLPClassifier(),
            "Perceptron": Perceptron(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machines (SVM)": SVC(),
        }

        metric = accuracy_score
        metric_label = "Accuracy"

        # Evaluate models
        st.subheader(f"{task} Model Performance")
        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = metric(y_test, y_pred)
            results[name] = score

        # Convert results to DataFrame
        results_df = pd.DataFrame(results.items(), columns=["Model", metric_label])

        # Highlight the best and worst rows in the table
        best_model = results_df.loc[results_df[metric_label].idxmax()]
        worst_model = results_df.loc[results_df[metric_label].idxmin()]

        def highlight_best_and_worst(row):
            if row["Model"] == best_model["Model"]:
                return ["background-color: #48bb78; color: white"] * len(row)
            elif row["Model"] == worst_model["Model"]:
                return ["background-color: #f56565; color: white"] * len(row)
            else:
                return [""] * len(row)

        st.write(results_df.style.apply(highlight_best_and_worst, axis=1))

        # Generate the bar chart using Plotly
        def plot_highlighted_bar_chart(df, metric_label, best_model, worst_model):
            colors = [
                "#48bb78" if model == best_model["Model"]  # Green for the best model
                else "#f56565" if model == worst_model["Model"]  # Red for the worst model
                else "#63b3ed"  # Blue for other models
                for model in df["Model"]
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=df["Model"],
                    y=df[metric_label],
                    marker=dict(color=colors),
                    text=[f"{v:.4f}" for v in df[metric_label]],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title=f"{task} Performance Visualization",
                xaxis_title="Machine Learning Algorithm",
                yaxis_title=metric_label,
                xaxis_tickangle=-45,
                template="plotly_white",
            )
            
            return fig

        # Call the function to generate the chart
        st.plotly_chart(plot_highlighted_bar_chart(results_df, metric_label, best_model, worst_model))

        best_model_name = best_model["Model"]
        best_model_instance = models[best_model_name]

        # Train the best model on the full training data
        best_model_instance.fit(X_train, y_train)

        # Save the model to a .joblib file
        joblib_filename = f"{best_model_name}.joblib"
        joblib.dump(best_model_instance, joblib_filename)

        # Provide download link
        st.download_button(
            label="Download the Best Model",
            data=open(joblib_filename, "rb").read(),
            file_name=joblib_filename,
            mime="application/octet-stream",
        )
            
if mode == "Tune Models":
    st.subheader("Hyperparameter Tuning")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Dataset for Classification", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

    if uploaded_file:
        st.write("Dataset Preview")
        st.dataframe(data.head())

        # Assuming the last column is the target
        target_col = data.columns[-1]  # Last column as target
        feature_cols = data.columns[:-1]  # All other columns as features

        # Prepare features and target
        X = data[feature_cols]
        y = data[target_col]

        # Check if target variable contains missing values
        if y.isnull().sum() > 0:
            y = y.fillna(y.mean())  # Fill missing values

        # Handle categorical target for classification
        if y.dtype == 'object' or len(np.unique(y)) < 20:  # For classification
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            task = "Classification"

        # Handle missing values in features as well
        X.fillna(X.mean(), inplace=True)
    
        tuning_results = [] 

        st.write("Decision Tree ML")
        
        # Set collapsible sidebar parameters for hyperparameters
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
            random_seed = st.slider("Random Seed", 1, 100, 50)
            max_depth = st.slider("Max Depth", 1, 20, 5)
            min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Initialize and train the Decision Tree Classifier with hyperparameters
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_seed
        )

        model.fit(X_train, Y_train)
        name= "Decision Tree ML"
        # Evaluate the accuracy
        accuracy = model.score(X_test, Y_test)

        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        # Display the accuracy in the app
        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
        
        
        st.write("Gaussian Naive Bayes")

        # Set collapsible sidebar parameters for hyperparameters with unique keys
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")
            var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1, key="var_smoothing")

        # Convert var_smoothing from log scale to regular scale
        var_smoothing_value = 10 ** var_smoothing

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Initialize the Gaussian Naive Bayes classifier with hyperparameters
        model = GaussianNB(var_smoothing=var_smoothing_value)

        # Train the model on the training data
        model.fit(X_train, Y_train)
        name= "Gaussian Naive Bayes"
        # Evaluate the accuracy
        accuracy = model.score(X_test, Y_test)

        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        # Display the accuracy in the app
        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
        
        st.write("AdaBoost")

        # Set collapsible sidebar parameters for hyperparameters
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size2")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed2")
            n_estimators = st.slider("Number of Estimators", 1, 100, 50)

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Create an AdaBoost classifier
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)

        # Train the model on the training data
        model.fit(X_train, Y_train)
        name="AdaBoost"
        # Evaluate the accuracy
        accuracy = model.score(X_test, Y_test)

        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        # Display the accuracy in the app
        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
        
        st.write("K-Nearest Neighbors")
        # Set collapsible sidebar parameters for hyperparameters
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="keytest3")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed3")
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            weights = st.selectbox("Weights", options=["uniform", "distance"])
            algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"])

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Create a K-Nearest Neighbors (K-NN) classifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

        # Train the model on the training data
        model.fit(X_train, Y_train)
        name="K-Nearest Neighbors"
        # Evaluate the accuracy
        accuracy = model.score(X_test, Y_test)

        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        # Display the accuracy in the app
        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
        
        st.write("Logistic Regression")
        # Set collapsible sidebar parameters for hyperparameters
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test4")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed4")
            max_iter = st.slider("Max Iterations", 100, 500, 200)
            solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
            C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0)

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Create a Logistic Regression model
        model = LogisticRegression(max_iter=max_iter, solver=solver, C=C)

        # Train the model on the training data
        model.fit(X_train, Y_train)
        name="Logistic Regression"
        # Evaluate the accuracy
        accuracy = model.score(X_test, Y_test)

        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        # Display the accuracy in the app
        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
        
        st.write("MLP Classifier")
        # Set collapsible sidebar parameters for hyperparameters
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test5")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed5")
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 65,32)", "65,32")
            activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"])
            max_iter = st.slider("Max Iterations", 100, 500, 200, key="max5")

        # Convert hidden_layer_sizes input to tuple
        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Create an MLP-based model
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                            solver='adam', max_iter=max_iter, random_state=random_seed)

        # Train the model
        model.fit(X_train, Y_train)
        name="MLP Classifier"
        # Evaluate the accuracy
        accuracy = model.score(X_test, Y_test)

        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        # Display the accuracy in the app
        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

        st.write("Perceptron Classifier")
        # Set collapsible sidebar parameters for hyperparameters
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test6")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed6")
            max_iter = st.slider("Max Iterations", 100, 500, 200, key="max6")
            eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0)
            tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3)

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Create a Perceptron classifier
        model = Perceptron(max_iter=max_iter, random_state=random_seed, eta0=eta0, tol=tol)
        
        # Train the model
        model.fit(X_train, Y_train)
        name = "Perceptron"
        # Evaluate the accuracy
        accuracy = model.score(X_test, Y_test)

        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        # Display the accuracy in the app
        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
        
        st.write("Random Forest")
        # Set collapsible sidebar parameters for hyperparameters
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test7")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed7")
            n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100)
            max_depth = st.slider("Max Depth of Trees", 1, 50, None)  # Allows None for no limit
            min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2)
            min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1)

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Create a Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_seed,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        # Train the model
        model.fit(X_train, Y_train)
        name="Random Forest"
        # Evaluate the accuracy
        accuracy = model.score(X_test, Y_test)

        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        # Display the accuracy in the app
        st.write(f"Accuracy: {accuracy * 100.0:.3f}%")
        
        st.write("Support Vector Machine (SVM)")
        # Collapsible sidebar for hyperparameters
        with st.expander("",expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test8")
            random_seed = st.slider("Random Seed", 1, 100, 42, key="seed8")
            C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'])

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Create an SVM classifier
        model = SVC(kernel=kernel, C=C, random_state=random_seed)

        # Train the model
        model.fit(X_train, Y_train)
        name="Support Vector Machine (SVM)"
        # Evaluate the accuracy
        result = model.score(X_test, Y_test)
        
        tuning_results.append({"Model": name, "Accuracy": accuracy * 100.0})
        
        st.write(f"Accuracy: {result * 100.0:.3f}%")
        
        metric = accuracy_score
        metric_label = "Accuracy"
        
        # Convert tuning results to DataFrame
        results_df = pd.DataFrame(tuning_results)
        # Highlight the best and worst models in the table
        best_model = results_df.loc[results_df[metric_label].idxmax()]
        worst_model = results_df.loc[results_df[metric_label].idxmin()]

        def highlight_best_and_worst(row):
            if row["Model"] == best_model["Model"]:
                return ["background-color: #48bb78; color: white"] * len(row)
            elif row["Model"] == worst_model["Model"]:
                return ["background-color: #f56565; color: white"] * len(row)
            else:
                return [""] * len(row)

        # Display the results table with highlighted best and worst models
        st.write(results_df.style.apply(highlight_best_and_worst, axis=1))

        # Generate the bar chart using Plotly
        def plot_highlighted_bar_chart(df, metric_label, best_model, worst_model):
            # Define colors for the bars
            colors = [
                "#48bb78" if model == best_model["Model"]  # Green for the best model
                else "#f56565" if model == worst_model["Model"]  # Red for the worst model
                else "#63b3ed"  # Blue for other models
                for model in df["Model"]
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=df["Model"],
                    y=df[metric_label],
                    marker=dict(color=colors),
                    text=[f"{v:.4f}" for v in df[metric_label]],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title=f"{task} Performance Visualization",
                xaxis_title="Machine Learning Algorithm",
                yaxis_title=metric_label,
                xaxis_tickangle=-45,
                template="plotly_white",
            )
            
            return fig

        # Call the function to generate the chart with highlighted bars for the best and worst models
        st.plotly_chart(plot_highlighted_bar_chart(results_df, metric_label, best_model, worst_model))
        
        best_model_info = max(tuning_results, key=lambda x: x["Accuracy"])
        best_model_name = best_model_info["Model"]

        # Save the best model to a joblib file
        file_name = f"{best_model_name}_model.joblib"
        joblib.dump(model, file_name)

        # Provide a download link for the joblib file
        with open(file_name, "rb") as file:
            btn = st.download_button(
                label=f"Download the Tuned Model",
                data=file,
                file_name=file_name,
                mime="application/octet-stream"
            )