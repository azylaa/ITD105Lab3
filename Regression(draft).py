import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import AdaBoostRegressor
import plotly.graph_objects as go

st.sidebar.title("Machine Learning Model Comparison & Tuning")
mode = st.sidebar.radio("Choose Mode", ["Compare Models", "Tune Models"])

if mode == "Compare Models":
    st.subheader("Compare Models")
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Dataset for Regression", type="csv")

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

        # Handle missing values in the target variable
        if y.isnull().sum() > 0:
            y = y.fillna(y.mean())  # Fill missing values

        # Handle missing values in features as well
        X.fillna(X.mean(), inplace=True)

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Decision Tree (CART)": DecisionTreeRegressor(),
            "Elastic Net": ElasticNet(),
            "AdaBoost (Gradient Boosting)": GradientBoostingRegressor(),
            "K-Nearest Neighbors (K-NN)": KNeighborsRegressor(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge(),
            "Linear Regression": LinearRegression(),
            "Multi-Layer Perceptron (MLP)": MLPRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Machines (SVM)": SVR(),
        }
        metric = mean_absolute_error
        metric_label = "MAE"

        # Evaluate models
        st.write(f"Regression Model Performance")
        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # For regression, round predictions to avoid mismatch errors
            y_pred = np.round(y_pred, 2)
            
            score = metric(y_test, y_pred)
            results[name] = score

        # Convert results to DataFrame
        results_df = pd.DataFrame(results.items(), columns=["Model", metric_label])

        # Highlight the best and worst rows in the table
        best_model = results_df.loc[results_df[metric_label].idxmin()]  # Best model is the one with the lowest MAE
        worst_model = results_df.loc[results_df[metric_label].idxmax()]  # Worst model is the one with the highest MAE

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
                title="Regression Model Performance",
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
        joblib_filename = f"{best_model_name}_best_model.joblib"
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

        # Handle missing values in features as well
        X.fillna(X.mean(), inplace=True)
        tuning_results = [] 
        st.write("Decision Tree Regressor")
        with st.expander("Decision Tree Regressor Hyperparameters", expanded=False):
            max_depth = st.slider("Max Depth", 1, 20, None)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1)
            n_splits = st.slider("Number of Folds (K)", 2, 20, 10)
        # Split the dataset into K-Folds
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Create and train the Decision Tree Regressor
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # Calculate the mean absolute error using cross-validation
        scoring = 'neg_mean_absolute_error'
        results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

        # Display the results in the app
        mae = -results.mean()
        mae_std = results.std()
        name= "Decision Tree Regressor"
        tuning_results.append({"Model": name, "MAE": mae})
        
        st.write(f"Mean Absolute Error (MAE): {mae:.3f} (Standard Deviation: {mae_std:.3f})")
        
        st.write("Elastic Net")
        with st.expander("Elastic Net Hyperparameters", expanded=False):
            alpha = st.slider("Alpha (Regularization Strength)", 0.0, 5.0, 1.0, 0.1)
            l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
            max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)

        # Split the dataset into a 10-fold cross-validation
        kfold = KFold(n_splits=10, random_state=None)

        # Train the Elastic Net model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=None)

        # Calculate the mean absolute error
        scoring = 'neg_mean_absolute_error'
        results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        mae= -results.mean()
        name="Elastic Net"
        tuning_results.append({"Model": name, "MAE": mae })
        
        # Display the results
        st.write(f"Mean Absolute Error (MAE): {-results.mean():.3f} ± {results.std():.3f}")
        
        st.write("AdaBoost Regressor")
        with st.expander("AdaBoost Regressor Hyperparameters", expanded=False):
            n_estimators = st.slider("Number of Estimators", 1, 200, 50, 1)
            learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0, 0.01)

        # Split the dataset into a 10-fold cross-validation
        kfold = KFold(n_splits=10, random_state=None)

        # Train the AdaBoost model
        ada_model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=None)

        # Calculate the mean absolute error with AdaBoost
        scoring = 'neg_mean_absolute_error'
        ada_results = cross_val_score(ada_model, X, y, cv=kfold, scoring=scoring)
        mae= -ada_results.mean()
        name="AdaBoost Regressor"
        tuning_results.append({"Model": name, "MAE": mae})
        
        # Display the results
        st.write(f"AdaBoost Mean Absolute Error (MAE): {-ada_results.mean():.3f} ± {ada_results.std():.3f}")
        
        st.write("K-Nearest Neighbors")
        with st.expander("K-Nearest Neighbors Hyperparameters", expanded=False):
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1)
            weights = st.selectbox("Weights", ["uniform", "distance"])
            algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

        # Split the dataset into a 10-fold cross-validation
        kfold = KFold(n_splits=10, random_state=None)

        # Train the K-NN model
        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

        # Calculate the mean absolute error with K-NN
        scoring = 'neg_mean_absolute_error'
        knn_results = cross_val_score(knn_model, X, y, cv=kfold, scoring=scoring)
        mae=-knn_results.mean()
        name="K-Nearest Neighbors"
        tuning_results.append({"Model": name, "MAE": mae})
        
        # Display the results
        st.write(f"K-NN Mean Absolute Error (MAE): {-knn_results.mean():.3f} ± {knn_results.std():.3f}")
        
        st.write("Lasso and Ridge Regression")
        with st.expander("Lasso and Ridge Regression Hyperparameters", expanded=False):
            alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01)
            max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100)

        # Split the dataset into a 10-fold cross-validation
        kfold = KFold(n_splits=10, random_state=None)

        # Train the data on a Lasso Regression model
        lasso_model = Lasso(alpha=alpha, max_iter=max_iter, random_state=None)

        # Calculate the mean absolute error with Lasso
        scoring = 'neg_mean_absolute_error'
        lasso_results = cross_val_score(lasso_model, X, y, cv=kfold, scoring=scoring)
        mae=-lasso_results.mean()
        name="Lasso and Ridge Regression"
        tuning_results.append({"Model": name, "MAE": mae})
        
        # Display Lasso results
        st.write(f"Lasso Mean Absolute Error (MAE): {-lasso_results.mean():.3f} ± {lasso_results.std():.3f}")

        # Train the data on a Ridge Regression model
        ridge_model = Ridge(alpha=alpha, max_iter=max_iter, random_state=None)

        # Calculate the mean absolute error with Ridge
        ridge_results = cross_val_score(ridge_model, X, y, cv=kfold, scoring=scoring)
        
        # Display Ridge results
        st.write(f"Ridge Mean Absolute Error (MAE): {-ridge_results.mean():.3f} ± {ridge_results.std():.3f}")
        
        st.write("Linear Regression")
        # Split the dataset into a 10-fold cross-validation
        kfold = KFold(n_splits=10, random_state=None)

        # Collapsible sidebar for additional options if needed
        with st.expander("Linear Regression Hyperparameters", expanded=False):
            # Placeholder for future options, if needed
            st.write("N/A")

        # Train the data on a Linear Regression model
        model = LinearRegression()

        # Calculate the mean absolute error
        scoring = 'neg_mean_absolute_error'
        results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

        mae=-results.mean()
        name="Linear Regression"
        tuning_results.append({"Model": name, "MAE": mae})

        # Display results
        st.write(f"Mean Absolute Error (MAE): {-results.mean():.3f} ± {results.std():.3f}")

        st.write("MLP Regressor")
        with st.expander("MLP Regressor Hyperparameters", expanded=False):
            # Define hyperparameters with default values
            hidden_layer_sizes = st.slider("Hidden Layer Sizes", min_value=10, max_value=200, value=(100, 50), step=10)
            activation = st.selectbox("Activation Function", options=['identity', 'logistic', 'tanh', 'relu'], index=3)
            solver = st.selectbox("Solver", options=['adam', 'lbfgs', 'sgd'], index=0)
            learning_rate = st.selectbox("Learning Rate Schedule", options=['constant', 'invscaling', 'adaptive'], index=0)
            max_iter = st.slider("Max Iterations", min_value=100, max_value=2000, value=1000, step=100, key="max1")
            random_state = st.number_input("Random State", value=50)

        # Split the dataset into a 10-fold cross-validation
        kfold = KFold(n_splits=10, random_state=None)

        # Train the data on an MLP Regressor with specified hyperparameters
        mlp_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,  # Use user-defined hidden layer sizes
            activation=activation,                   # Use user-defined activation function
            solver=solver,                          # Use user-defined optimization algorithm
            learning_rate=learning_rate,            # Use user-defined learning rate schedule
            max_iter=max_iter,                      # Use user-defined max iterations
            random_state=random_state                # Use user-defined random state
        )

        # Calculate the mean absolute error with MLP
        scoring = 'neg_mean_absolute_error'
        mlp_results = cross_val_score(mlp_model, X, y, cv=kfold, scoring=scoring)
        mae=-mlp_results.mean()
        name="MLP Regressor"
        
        tuning_results.append({"Model": name, "MAE": mae})
        
        # Display results
        st.write("Mean Absolute Error (MAE): %.3f ± %.3f" % (-mlp_results.mean(), mlp_results.std()))

        st.write("Random Forest Regressor")
        with st.expander("Random Forest Regressor Hyperparameters", expanded=False):
            n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
            max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=None)
            min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
            min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1)
            random_state = st.number_input("Random State", value=42)

        # Split the dataset into a 10-fold cross-validation
        kfold = KFold(n_splits=10, random_state=None)

        # Train the data on a Random Forest Regressor with specified hyperparameters
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,      # Use user-defined number of trees
            max_depth=max_depth,            # Use user-defined max depth
            min_samples_split=min_samples_split,  # Use user-defined min samples split
            min_samples_leaf=min_samples_leaf,    # Use user-defined min samples leaf
            random_state=random_state        # Use user-defined random state
        )

        # Calculate the mean absolute error with Random Forest
        scoring = 'neg_mean_absolute_error'
        rf_results = cross_val_score(rf_model, X, y, cv=kfold, scoring=scoring)
        mae= -rf_results.mean()
        name="Random Forest Regressor"
        tuning_results.append({"Model": name, "MAE": mae})
        
        # Display results
        st.write("Random Forest Mean Absolute Error (MAE): %.3f ± %.3f" % (-rf_results.mean(), rf_results.std()))
        
        st.write("Support Vector Regressor (SVR)")
        with st.expander("Support Vector Regressor (SVR) Hyperparameters", expanded=False):
            kernel = st.selectbox("Kernel", options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
            C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
            epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

        # Split the dataset into a 10-fold cross-validation
        kfold = KFold(n_splits=10, random_state=None)

        # Train the data on a Support Vector Regressor with specified hyperparameters
        svm_model = SVR(
            kernel=kernel,          # Use user-defined kernel
            C=C,                    # Use user-defined regularization parameter
            epsilon=epsilon         # Use user-defined epsilon
        )

        # Calculate the mean absolute error with SVM
        scoring = 'neg_mean_absolute_error'
        svm_results = cross_val_score(svm_model, X, y, cv=kfold, scoring=scoring)
        mae= -svm_results.mean()
        name="Support Vector Regressor (SVR)"
        tuning_results.append({"Model": name, "MAE": mae})
        # Display results
        st.write("SVM Mean Absolute Error (MAE): %.3f ± %.3f" % (-svm_results.mean(), svm_results.std()))
        
        metric = mean_absolute_error
        metric_label = "MAE"
        
        tuning_results_df = pd.DataFrame(tuning_results)

        # Identify the best and worst model based on MAE
        best_tuned_model = tuning_results_df.loc[tuning_results_df['MAE'].idxmin()]
        worst_tuned_model = tuning_results_df.loc[tuning_results_df['MAE'].idxmax()]

        # Function to highlight best and worst models
        def highlight_best_and_worst_tuned(row):
            if row['Model'] == best_tuned_model['Model']:
                return ['background-color: #48bb78; color: white'] * len(row)  # Best model - green
            elif row['Model'] == worst_tuned_model['Model']:
                return ['background-color: #f56565; color: white'] * len(row)  # Worst model - red
            else:
                return [''] * len(row)

        # Display the results with highlighted best and worst models
        st.write(tuning_results_df.style.apply(highlight_best_and_worst_tuned, axis=1))

        # For the bar chart
        def plot_highlighted_tuned_bar_chart(df, best_model, worst_model):
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
                    y=df["MAE"],
                    marker=dict(color=colors),
                    text=[f"{v:.4f}" for v in df["MAE"]],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title="Tuned Model Performance",
                xaxis_title="Machine Learning Algorithm",
                yaxis_title="Mean Absolute Error (MAE)",
                xaxis_tickangle=-45,
                template="plotly_white",
            )

            return fig

        # Call the function to generate the chart and display it
        st.plotly_chart(plot_highlighted_tuned_bar_chart(tuning_results_df, best_tuned_model, worst_tuned_model))

        best_model_info = max(tuning_results, key=lambda x: x["MAE"])
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