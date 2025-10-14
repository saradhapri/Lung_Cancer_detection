"""
Utility functions for the lung cancer detection model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess the raw dataset for training or prediction.
    
    Args:
        df (pandas.DataFrame): Raw dataset
        
    Returns:
        pandas.DataFrame: Preprocessed dataset
    """
    # Remove duplicates
    df_clean = df.drop_duplicates()
    
    # Encode categorical variables
    encoder = LabelEncoder()
    df_clean['LUNG_CANCER'] = encoder.fit_transform(df_clean['LUNG_CANCER'])
    df_clean['GENDER'] = encoder.fit_transform(df_clean['GENDER'])
    
    return df_clean

def prepare_features(df):
    """
    Prepare features for model training by converting categorical variables.
    
    Args:
        df (pandas.DataFrame): Preprocessed dataset
        
    Returns:
        tuple: Features (X) and target (y)
    """
    X = df.drop(['LUNG_CANCER'], axis=1)
    y = df['LUNG_CANCER']
    
    # Convert categorical features from 1-2 scale to 0-1 scale
    categorical_features = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                          'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
                          'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
                          'SWALLOWING DIFFICULTY', 'CHEST PAIN']
    
    for feature in categorical_features:
        if feature in X.columns:
            X[feature] = X[feature] - 1
    
    return X, y

def validate_input(input_data):
    """
    Validate user input data.
    
    Args:
        input_data (dict): User input data
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_features = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                        'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
                        'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                        'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
    
    # Check if all required features are present
    for feature in required_features:
        if feature not in input_data:
            return False
    
    # Check age range
    if not 20 <= input_data['AGE'] <= 100:
        return False
    
    # Check categorical values
    categorical_features = [f for f in required_features if f != 'AGE' and f != 'GENDER']
    for feature in categorical_features:
        if input_data[feature] not in [1, 2]:
            return False
    
    # Check gender
    if input_data['GENDER'] not in [0, 1]:
        return False
    
    return True

def get_risk_level(probability):
    """
    Determine risk level based on probability.
    
    Args:
        probability (float): Cancer probability (0-1)
        
    Returns:
        str: Risk level description
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def format_prediction_result(prediction, probability):
    """
    Format prediction results for display.
    
    Args:
        prediction (int): Binary prediction (0 or 1)
        probability (array): Probability array [no_cancer_prob, cancer_prob]
        
    Returns:
        dict: Formatted results
    """
    return {
        'prediction': "Cancer Detected" if prediction == 1 else "No Cancer Detected",
        'risk_level': get_risk_level(probability[1]),
        'cancer_probability': probability[1],
        'no_cancer_probability': probability[0],
        'confidence': max(probability)
    }
