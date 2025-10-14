import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0;
    }
    .prediction-positive {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-negative {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching for performance"""
    try:
        model = joblib.load('lung_cancer_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'lung_cancer_model.pkl' is in the project directory.")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for visualization"""
    try:
        return pd.read_csv("survey lung cancer.csv")
    except FileNotFoundError:
        return None

def get_exact_model_feature_names():
    """Get the EXACT feature names with correct spacing as expected by the model"""
    # These are the exact feature names from your model output (including trailing spaces!)
    return [
        'GENDER',
        'AGE', 
        'SMOKING',
        'YELLOW_FINGERS',
        'ANXIETY',
        'PEER_PRESSURE',
        'CHRONIC DISEASE',
        'FATIGUE ',      # Note the trailing space!
        'ALLERGY ',      # Note the trailing space!
        'WHEEZING',
        'ALCOHOL CONSUMING',
        'COUGHING',
        'SHORTNESS OF BREATH',
        'SWALLOWING DIFFICULTY',
        'CHEST PAIN'
    ]

def preprocess_input(data):
    """Preprocess user input to match training data format"""
    # Get the exact feature names the model expects
    expected_features = get_exact_model_feature_names()
    
    # Create a DataFrame with the exact column names and order
    processed_data = pd.DataFrame()
    
    for feature in expected_features:
        if feature in data.columns:
            processed_data[feature] = data[feature]
        else:
            st.error(f"Missing feature: '{feature}'")
            return None
    
    # Convert categorical features from 1-2 scale to 0-1 scale (except GENDER and AGE)
    categorical_features = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                          'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',  # Note the spaces!
                          'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
                          'SWALLOWING DIFFICULTY', 'CHEST PAIN']
    
    for feature in categorical_features:
        if feature in processed_data.columns:
            processed_data[feature] = processed_data[feature] - 1
    
    return processed_data

def create_feature_importance_plot(model, feature_names):
    """Create feature importance visualization"""
    if model is None:
        return None
        
    # Clean feature names for display (remove trailing spaces)
    clean_feature_names = [name.strip() for name in feature_names]
    
    importance_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, 
                 x='Importance', 
                 y='Feature',
                 orientation='h',
                 title="Feature Importance Analysis",
                 color='Importance',
                 color_continuous_scale='Blues',
                 height=600)
    
    fig.update_layout(
        showlegend=False,
        title_font_size=20,
        font=dict(size=12)
    )
    return fig

def create_prediction_probability_gauge(probability):
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cancer Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">Lung Cancer Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown('<div class="sub-header">Navigation</div>', unsafe_allow_html=True)
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Information", "Data Visualization"])
    
    # Load model
    model = load_model()
    
    if page == "Prediction":
        prediction_page(model)
    elif page == "Model Information":
        model_info_page(model)
    else:
        visualization_page()

def prediction_page(model):
    st.markdown('<div class="sub-header">Patient Information Input</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 20, 100, 50)
        
        st.markdown("### Lifestyle Factors")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
        
        st.markdown("### Physical Symptoms")
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        allergy = st.selectbox("Allergy", ["No", "Yes"])
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    
    with col2:
        st.markdown("### Medical History")
        chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        
        st.markdown("### Respiratory Symptoms")
        coughing = st.selectbox("Coughing", ["No", "Yes"])
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
        
        st.markdown("### Prediction")
        if st.button("Predict Cancer Risk", type="primary"):
            # Create input data with EXACT feature names including trailing spaces!
            input_data = {
                'GENDER': 1 if gender == "Male" else 0,
                'AGE': age,
                'SMOKING': 2 if smoking == "Yes" else 1,
                'YELLOW_FINGERS': 2 if yellow_fingers == "Yes" else 1,
                'ANXIETY': 2 if anxiety == "Yes" else 1,
                'PEER_PRESSURE': 2 if peer_pressure == "Yes" else 1,
                'CHRONIC DISEASE': 2 if chronic_disease == "Yes" else 1,
                'FATIGUE ': 2 if fatigue == "Yes" else 1,  # Note the trailing space!
                'ALLERGY ': 2 if allergy == "Yes" else 1,  # Note the trailing space!
                'WHEEZING': 2 if wheezing == "Yes" else 1,
                'ALCOHOL CONSUMING': 2 if alcohol == "Yes" else 1,
                'COUGHING': 2 if coughing == "Yes" else 1,
                'SHORTNESS OF BREATH': 2 if shortness_breath == "Yes" else 1,
                'SWALLOWING DIFFICULTY': 2 if swallowing_difficulty == "Yes" else 1,
                'CHEST PAIN': 2 if chest_pain == "Yes" else 1
            }
            make_prediction(model, input_data)

def make_prediction(model, input_data):
    """Make prediction and display results"""
    try:
        # Convert to DataFrame with exact column order
        expected_features = get_exact_model_feature_names()
        
        # Create DataFrame with features in the exact order expected by the model
        input_df = pd.DataFrame([input_data])[expected_features]
        
        # Preprocess
        processed_input = preprocess_input(input_df)
        
        if processed_input is None:
            st.error("Error in preprocessing input data.")
            return
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0]
        
        # Display results
        st.markdown("---")
        st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
        
        # Create three columns for results
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.markdown(
                    '<div class="prediction-positive">High Risk: Cancer Detected</div>', 
                    unsafe_allow_html=True
                )
                st.error("This prediction indicates a high risk of lung cancer. Please consult with a healthcare professional immediately.")
            else:
                st.markdown(
                    '<div class="prediction-negative">Low Risk: No Cancer Detected</div>', 
                    unsafe_allow_html=True
                )
                st.success("This prediction indicates a low risk of lung cancer. Continue regular health check-ups.")
        
        # Probability gauge
        st.markdown("### Risk Assessment")
        fig_gauge = create_prediction_probability_gauge(probability[1])
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Probability breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.metric("No Cancer Probability", f"{probability[0]:.2%}")
        with col2:
            st.metric("Cancer Probability", f"{probability[1]:.2%}")
        
        st.markdown("---")
        st.info("""
        **Disclaimer**: This prediction is based on a machine learning model and should not replace professional medical diagnosis. 
        Please consult with qualified healthcare professionals for accurate diagnosis and treatment.
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check that all input fields are filled correctly.")

def model_info_page(model):
    st.markdown('<div class="sub-header">Model Performance Information</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded.")
        return
    
    # Model metrics using Streamlit's native components instead of HTML
    st.markdown("### Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value="85.7%",
            delta="High Performance"
        )
    
    with col2:
        st.metric(
            label="Precision", 
            value="84.6%",
            delta="Good Precision"
        )
    
    with col3:
        st.metric(
            label="Recall",
            value="100.0%",
            delta="Perfect Recall"
        )
    
    with col4:
        st.metric(
            label="F1-Score",
            value="91.7%",
            delta="Excellent Balance"
        )
    
    # Additional metrics in a nice layout
    st.markdown("---")
    st.markdown("### Additional Performance Metrics")
    
    # Create a DataFrame for additional metrics
    metrics_data = {
        'Metric': ['AUC-ROC', 'Specificity', 'NPV (Negative Predictive Value)', 'Training Samples'],
        'Value': ['97.7%', '33.3%', '100.0%', '276 samples'],
        'Description': [
            'Area Under ROC Curve - Excellent discrimination',
            'True Negative Rate - Model identifies non-cancer cases',
            'Probability of true negative when test is negative', 
            'Total samples after removing duplicates'
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Feature importance
    st.markdown("---")
    feature_names = get_exact_model_feature_names()
    
    fig_importance = create_feature_importance_plot(model, feature_names)
    if fig_importance:
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model details
    st.markdown("---")
    st.markdown("### Model Architecture & Training Details")
    
    # Create expandable sections for detailed info
    with st.expander("🔧 Model Configuration"):
        st.markdown("""
        - **Algorithm**: Random Forest Classifier
        - **Number of Estimators**: 100 trees
        - **Random State**: 42 (for reproducibility)
        - **Max Features**: Auto (sqrt of total features)
        - **Bootstrap**: True
        """)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        - **Original Dataset Size**: 309 samples
        - **After Preprocessing**: 276 samples (33 duplicates removed)
        - **Features**: 15 clinical and behavioral attributes
        - **Target Classes**: Binary (Cancer: YES/NO)
        - **Class Distribution**: Check Data Visualization page
        """)
    
    with st.expander("🎯 Model Features"):
        clean_features = [name.strip() for name in feature_names]
        for i, feature in enumerate(clean_features, 1):
            st.write(f"{i:2d}. **{feature}**")
    
    with st.expander("⚠️ Model Limitations"):
        st.markdown("""
        - **Data Dependency**: Performance depends on data quality and representativeness
        - **Feature Importance**: Some features may be more critical than others
        - **Medical Context**: This is a screening tool, not a diagnostic device
        - **Version Compatibility**: Model trained with scikit-learn 1.3.0
        """)

def visualization_page():
    st.markdown('<div class="sub-header">Data Visualization & Insights</div>', unsafe_allow_html=True)
    
    # Load sample data
    df = load_sample_data()
    
    if df is None:
        st.error("Sample data not found. Please ensure 'survey lung cancer.csv' is in the project directory.")
        return
    
    # Basic statistics
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        cancer_count = df['LUNG_CANCER'].value_counts().get('YES', 0)
        st.metric("Cancer Cases", cancer_count)
    with col4:
        no_cancer_count = df['LUNG_CANCER'].value_counts().get('NO', 0)
        st.metric("Non-Cancer Cases", no_cancer_count)
    
    # Age distribution
    st.markdown("### Age Distribution Analysis")
    fig_age = px.histogram(df, x='AGE', color='LUNG_CANCER', 
                          title="Age Distribution by Cancer Status",
                          nbins=20)
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Gender distribution
    st.markdown("### Gender Distribution")
    gender_cancer = df.groupby(['GENDER', 'LUNG_CANCER']).size().reset_index(name='Count')
    fig_gender = px.bar(gender_cancer, x='GENDER', y='Count', color='LUNG_CANCER',
                       title="Cancer Distribution by Gender")
    st.plotly_chart(fig_gender, use_container_width=True)
    
    # Smoking analysis
    st.markdown("### Smoking Impact Analysis")
    smoking_cancer = df.groupby(['SMOKING', 'LUNG_CANCER']).size().reset_index(name='Count')
    fig_smoking = px.bar(smoking_cancer, x='SMOKING', y='Count', color='LUNG_CANCER',
                        title="Cancer Distribution by Smoking Status")
    st.plotly_chart(fig_smoking, use_container_width=True)

if __name__ == "__main__":
    main()
