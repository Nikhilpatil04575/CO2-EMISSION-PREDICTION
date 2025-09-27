import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# Model imports
from catboost import CatBoostRegressor

# --- Configuration ---
# Set the main page configuration
st.set_page_config(
    page_title="CO2 Emission Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Setup ---

# Load data (using the original notebook name for consistency)
@st.cache_data
def load_data():
    try:
        # NOTE: Make sure 'CO2 Emissions.csv' is in the same directory as this script!
        # Your notebook loaded a file named 'CO2 Emissions.csv'.
        df = pd.read_csv('CO2 Emissions.csv')
        # Clean up column names for easier access and display
        df.columns = df.columns.str.replace(r'\(L\)', '(L)', regex=True).str.replace(r'\(L/100 km\)', '(L/100 km)', regex=True).str.replace(r'\(g/km\)', '(g/km)', regex=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'CO2 Emissions.csv' not found. Please ensure the file is in the same directory as this script.")
        return pd.DataFrame()

df = load_data()

# --- Model Training and Prediction Setup ---
# Only proceed if data is loaded successfully
if not df.empty:
    # Target and Features
    X = df.drop('CO2 Emissions(g/km)', axis=1)
    y = df['CO2 Emissions(g/km)']

    # Define training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Feature Lists for CatBoost Pipeline ---
    # CatBoost uses native handling for categorical features, so we only use StandardScaler on numeric features.
    # CRITICAL: These lists must include ALL 11 features from your X_train/X_test.
    numeric_cols_cb = [
        'Engine Size(L)', 
        'Cylinders', 
        'Fuel Consumption Comb (L/100 km)', 
        'Fuel Consumption Hwy (L/100 km)', 
        'Fuel Consumption City (L/100 km)',
        'Fuel Consumption Comb (mpg)' 
    ]
    categorical_cols_cb = [
        'Make', 
        'Model', 
        'Vehicle Class', 
        'Transmission', 
        'Fuel Type'
    ]

    # Preprocessing: Scale numeric features and pass through categorical features
    preprocessing_cb = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_cols_cb),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    preprocessing_cb.set_output(transform="pandas")


    @st.cache_resource
    def train_best_model(_X_train, _y_train, _prep_pipe, _cat_cols):
        # CatBoost Regressor configuration (based on your notebook)
        cat_model = CatBoostRegressor(
            iterations=1000, 
            learning_rate=0.1, 
            depth=6, 
            verbose=0,
            cat_features=_cat_cols
        )
        
        pipeline_cb = Pipeline(steps=[
            ('preprocess', _prep_pipe),
            ('regressor', cat_model)
        ])
        
        # Fit the pipeline
        pipeline_cb.fit(_X_train, _y_train)
        return pipeline_cb, cat_model

    pipeline_cb, cat_model = train_best_model(X_train, y_train, preprocessing_cb, categorical_cols_cb)
    
    # --- Performance Metrics for Display ---
    y_pred_test = pipeline_cb.predict(X_test)
    r2_final = r2_score(y_test, y_pred_test)
    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_test))
    avg_co2 = df['CO2 Emissions(g/km)'].mean()

# --- Streamlit Layout ---
if not df.empty:
    st.title("🚗 CO2 Emission Prediction Dashboard")

    tab_overview, tab_performance, tab_predictor = st.tabs([
        "📊 Overview & EDA", 
        "📈 Model Performance", 
        "🔮 Interactive Predictor"
    ])

    # --- TAB 1: Overview & EDA ---
    with tab_overview:
        st.header("Exploratory Data Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df)} vehicles")
        with col2:
            st.metric("Average CO2 Emission", f"{avg_co2:.2f} g/km")
        with col3:
            st.metric("Top R-squared Score", f"{r2_final:.4f}")

        st.markdown("---")

        # Scatter Plots (Relationship with Target)
        st.subheader("CO2 Emissions vs. Fuel Consumption (Colored by Vehicle Class)")
        
        features_to_plot = [
            'Fuel Consumption Comb (L/100 km)',
            'Fuel Consumption City (L/100 km)',
            'Fuel Consumption Hwy (L/100 km)'
        ]
        
        selected_x = st.selectbox("Select X-Axis Feature:", features_to_plot)

        fig_scatter = px.scatter(
            df, 
            x=selected_x, 
            y="CO2 Emissions(g/km)", 
            color="Vehicle Class",
            title=f'{selected_x} vs. CO2 Emissions'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


        # Bar Chart (Categorical Breakdown)
        st.subheader("Average CO2 Emissions by Vehicle Class")
        
        df_group = df.groupby('Vehicle Class')['CO2 Emissions(g/km)'].mean().sort_values(ascending=False).reset_index()
        fig_bar = px.bar(
            df_group, 
            x='Vehicle Class', 
            y='CO2 Emissions(g/km)',
            color='Vehicle Class',
            title='Average CO2 Emissions by Vehicle Class'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 2: Model Performance & Comparison ---
    with tab_performance:
        st.header("Best Model Performance (CatBoost)")

        col_met1, col_met2 = st.columns(2)
        with col_met1:
            st.metric("R-squared (Test Set)", f"{r2_final:.4f}", help="Close to 1.0 indicates a great fit.")
        with col_met2:
            st.metric("RMSE (Root Mean Squared Error)", f"{rmse_final:.2f} g/km", help="Average error in prediction units.")

        st.markdown("---")

        # Prediction vs. Actual Plot
        st.subheader("Actual vs. Predicted CO2 Emissions (Test Set)")
        df_test_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred_test,
            'Residuals': y_test - y_pred_test
        })

        fig_perf = px.scatter(
            df_test_results, 
            x='Actual', 
            y='Predicted', 
            title='Actual vs. Predicted CO2 (Ideal: Points on y=x)'
        )
        fig_perf.update_layout(xaxis_title="Actual CO2 Emissions (g/km)", yaxis_title="Predicted CO2 Emissions (g/km)")
        
        # Add a line for perfect prediction (y=x) manually
        fig_perf.add_shape(
            type="line", line=dict(dash='dash', color='red'),
            x0=df_test_results['Actual'].min(), y0=df_test_results['Actual'].min(),
            x1=df_test_results['Actual'].max(), y1=df_test_results['Actual'].max()
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Residuals Plot
        st.subheader("Residuals Distribution")
        fig_res = px.histogram(
            df_test_results, 
            x='Residuals', 
            title='Distribution of Prediction Errors (Residuals)'
        )
        st.plotly_chart(fig_res, use_container_width=True)
        
        # Feature Importance (CatBoost)
        st.subheader("Top Feature Importances")
        
        # Get feature names from X_train to ensure length matches the model's importances
        feature_names = X_train.columns.tolist() 

        # Get feature importance from the CatBoost model
        feature_importances = cat_model.get_feature_importance()
        
        # Create Series with the correct index/names
        feature_scores = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
        
        top_features = feature_scores.head(20)
        
        fig_feat = px.bar(
            top_features, 
            x=top_features.values, 
            y=top_features.index,
            orientation='h',
            title='Top 20 Feature Importances'
        )
        fig_feat.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Importance Score")
        st.plotly_chart(fig_feat, use_container_width=True)


    # --- TAB 3: Interactive Predictor ---
    with tab_predictor:
        st.header("Predict New CO2 Emission")
        
        # Create a form for user input
        with st.form("prediction_form"):
            st.subheader("Vehicle Specifications")
            
            # --- Input Columns ---
            col_a, col_b = st.columns(2)
            
            # Use unique values from the original data for select boxes
            with col_a:
                make = st.selectbox("Make:", df['Make'].unique(), key='make_input')
                v_class = st.selectbox("Vehicle Class:", df['Vehicle Class'].unique(), key='vclass_input')
                
                e_min, e_max = df['Engine Size(L)'].min(), df['Engine Size(L)'].max()
                e_size = st.slider("Engine Size (L):", min_value=float(e_min), max_value=float(e_max), value=float(df['Engine Size(L)'].mean()), step=0.1, key='esize_input')
                cylinders = st.selectbox("Cylinders:", sorted(df['Cylinders'].unique()), key='cyl_input')
                
            with col_b:
                # Filter models based on Make for better UX
                model_options = df[df['Make'] == make]['Model'].unique()
                model = st.selectbox("Model:", model_options, key='model_input')
                fuel_type = st.selectbox("Fuel Type:", df['Fuel Type'].unique(), key='fuel_input')
                transmission = st.selectbox("Transmission:", df['Transmission'].unique(), key='trans_input')
                
                # Fuel Consumption Sliders
                comb_min, comb_max = df['Fuel Consumption Comb (L/100 km)'].min(), df['Fuel Consumption Comb (L/100 km)'].max()
                comb_l100km = st.slider("Fuel Consumption Comb (L/100 km):", min_value=float(comb_min), max_value=float(comb_max), value=float(df['Fuel Consumption Comb (L/100 km)'].mean()), step=0.1, key='comb_input')
                
                city_min, city_max = df['Fuel Consumption City (L/100 km)'].min(), df['Fuel Consumption City (L/100 km)'].max()
                city_l100km = st.slider("Fuel Consumption City (L/100 km):", min_value=float(city_min), max_value=float(city_max), value=float(df['Fuel Consumption City (L/100 km)'].mean()), step=0.1, key='city_input')
                
                hwy_min, hwy_max = df['Fuel Consumption Hwy (L/100 km)'].min(), df['Fuel Consumption Hwy (L/100 km)'].max()
                hwy_l100km = st.slider("Fuel Consumption Hwy (L/100 km):", min_value=float(hwy_min), max_value=float(hwy_max), value=float(df['Fuel Consumption Hwy (L/100 km)'].mean()), step=0.1, key='hwy_input')
                
                # Hidden/Default field for 'Fuel Consumption Comb (mpg)' 
                # This must be included in the input DataFrame to match the training data columns (11 features total).
                # We can calculate a placeholder value based on the combined L/100km input, or just use the mean.
                mpg = df['Fuel Consumption Comb (mpg)'].mean() # Using the dataset average as a safe placeholder
                
            submitted = st.form_submit_button("Predict CO2 Emissions")

        if submitted:
            # Create a DataFrame for the prediction
            new_data = pd.DataFrame({
                'Make': [make],
                'Model': [model],
                'Vehicle Class': [v_class],
                'Engine Size(L)': [e_size],
                'Cylinders': [cylinders],
                'Transmission': [transmission],
                'Fuel Type': [fuel_type],
                'Fuel Consumption City (L/100 km)': [city_l100km],
                'Fuel Consumption Hwy (L/100 km)': [hwy_l100km],
                'Fuel Consumption Comb (L/100 km)': [comb_l100km],
                'Fuel Consumption Comb (mpg)': [mpg]
            })
            
            # Make prediction
            prediction = pipeline_cb.predict(new_data)[0]
            
            # --- Display Prediction ---
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown(f"### Predicted CO2 Emission:")
                st.success(f"## {prediction:.2f} g/km")
            
            with col_res2:
                st.markdown(f"### Comparison to Average:")
                # Determine if the prediction is better or worse than the average
                diff = prediction - avg_co2
                if diff < 0:
                    st.markdown(f"The predicted emission is **{abs(diff):.2f} g/km lower** than the dataset average.")
                elif diff > 0:
                    st.markdown(f"The predicted emission is **{diff:.2f} g/km higher** than the dataset average.")
                else:
                    st.markdown("The predicted emission is exactly the same as the dataset average.")
            
            # Visual comparison
            st.markdown("### Visual Comparison")
            
            # Create a simple comparison DataFrame
            comparison_df = pd.DataFrame({
                'Metric': ['Predicted CO2', 'Dataset Average CO2'],
                'CO2 Emissions (g/km)': [prediction, avg_co2]
            })
            
            fig_comp = px.bar(
                comparison_df,
                x='Metric',
                y='CO2 Emissions (g/km)',
                color='Metric',
                color_discrete_map={
                    'Predicted CO2': 'green',
                    'Dataset Average CO2': 'gray'
                },
                title='Predicted CO2 vs. Dataset Average'
            )
            st.plotly_chart(fig_comp, use_container_width=True)