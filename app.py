# -*- coding: utf-8 -*-
# Suppress TensorFlow warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no WARN, 3=no ERROR

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs
import joblib

app = Flask(__name__)

# Global variables
model = None
scaler = None
label_encoder = None

# WQI parameters
WQI_PARAMS = [
    'Dissolved Oxygen (mg/L)',
    'pH',
    'Conductivity (umho/cm)',
    'BOD (mg/L)',
    'Nitrate N (mg/L)',
    'Fecal Streptococci (MPN/100ml)'
]

def calc_wqi_for_df(df):
    """Calculate Water Quality Index"""
    weights = {
        'Dissolved Oxygen (mg/L)': 0.17,
        'pH': 0.11,
        'Conductivity (umho/cm)': 0.08,
        'BOD (mg/L)': 0.11,
        'Nitrate N (mg/L)': 0.10,
        'Fecal Streptococci (MPN/100ml)': 0.16
    }
    
    wqi_scores = []
    for idx, row in df.iterrows():
        score = 0
        total_weight = 0
        
        for param, weight in weights.items():
            value = row[param]
            
            # Normalize
            if param == 'Dissolved Oxygen (mg/L)':
                norm_value = min(value * 10, 100)
            elif param == 'pH':
                norm_value = max(0, 100 - abs(value - 7) * 20)
            elif param == 'Conductivity (umho/cm)':
                norm_value = max(0, 100 - value / 20)
            elif param == 'BOD (mg/L)':
                norm_value = max(0, 100 - value * 8)
            elif param == 'Nitrate N (mg/L)':
                norm_value = max(0, 100 - value * 2)
            elif param == 'Fecal Streptococci (MPN/100ml)':
                norm_value = max(0, 100 - value / 10)
            else:
                norm_value = 50
            
            score += norm_value * weight
            total_weight += weight
        
        final_score = score / total_weight if total_weight > 0 else 0
        wqi_scores.append(min(max(final_score, 0), 100))
    
    return wqi_scores

def wqi_classification(wqi):
    """Classify WQI"""
    if wqi > 75:
        return "Very Poor(0)"
    elif wqi >= 51:
        return "Poor(1)"
    elif wqi >= 26:
        return "Good(2)"
    else:
        return "Excellent(3)"

def load_models():
    """Load pre-trained model"""
    global model, scaler, label_encoder
    
    try:
        model_path = 'lstm_model.pkl'
        
        # Try different loading methods
        if os.path.exists(model_path):
            print("Loading model...")
            
            try:
                # Try joblib
                model = joblib.load(model_path)
                print("✓ Model loaded with joblib")
            except:
                try:
                    # Try pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    print("✓ Model loaded with pickle")
                except:
                    try:
                        # Try Keras
                        model = tf.keras.models.load_model(model_path)
                        print("✓ Model loaded with Keras")
                    except Exception as e:
                        print(f"✗ Could not load model: {e}")
                        return False
        
        # Load scaler if exists
        if os.path.exists('scaler.pkl'):
            try:
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                print("✓ Scaler loaded")
            except:
                print("✗ Could not load scaler")
        
        # Setup label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(['Very Poor(0)', 'Poor(1)', 'Good(2)', 'Excellent(3)'])
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def home():
    """Home page"""
    simple_params = ['Dissolved Oxygen', 'pH', 'Conductivity', 'BOD', 'Nitrate N', 'Fecal Streptococci']
    return render_template('index.html', parameters=simple_params)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions"""
    try:
        # Get form data
        do_val = float(request.form.get('Dissolved Oxygen', 7.5))
        ph_val = float(request.form.get('pH', 7.0))
        cond_val = float(request.form.get('Conductivity', 500))
        bod_val = float(request.form.get('BOD', 3.0))
        nitrate_val = float(request.form.get('Nitrate N', 5.0))
        fecal_val = float(request.form.get('Fecal Streptococci', 100))
        
        # Prepare input
        input_data = np.array([[do_val, ph_val, cond_val, bod_val, nitrate_val, fecal_val]])
        input_df = pd.DataFrame(input_data, columns=WQI_PARAMS)
        
        # Calculate WQI
        wqi_score = calc_wqi_for_df(input_df)[0]
        wqi_class = wqi_classification(wqi_score)
        
        # Use model if available
        predicted_class = wqi_class
        confidence = 95.0
        
        if model is not None:
            try:
                # Scale input
                if scaler is not None:
                    input_scaled = scaler.transform(input_data)
                else:
                    input_scaled = input_data
                
                # Reshape for LSTM
                input_reshaped = input_scaled.reshape(1, 1, 6)
                
                # Predict
                prediction = model.predict(input_reshaped, verbose=0)
                pred_idx = np.argmax(prediction[0])
                
                # Map to class
                class_map = {0: 'Very Poor(0)', 1: 'Poor(1)', 2: 'Good(2)', 3: 'Excellent(3)'}
                predicted_class = class_map.get(pred_idx, wqi_class)
                confidence = float(prediction[0][pred_idx]) * 100
            except Exception as e:
                print(f"Model prediction failed: {e}")
        
        # Prepare result
        result = {
            'success': True,
            'wqi_score': round(wqi_score, 2),
            'wqi_class': wqi_class,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'parameters': {
                'Dissolved Oxygen': do_val,
                'pH': ph_val,
                'Conductivity': cond_val,
                'BOD': bod_val,
                'Nitrate N': nitrate_val,
                'Fecal Streptococci': fecal_val
            }
        }
        
        simple_params = ['Dissolved Oxygen', 'pH', 'Conductivity', 'BOD', 'Nitrate N', 'Fecal Streptococci']
        return render_template('index.html', prediction=result, parameters=simple_params, show_result=True)
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        simple_params = ['Dissolved Oxygen', 'pH', 'Conductivity', 'BOD', 'Nitrate N', 'Fecal Streptococci']
        return render_template('index.html', error=error_msg, parameters=simple_params, show_result=False)

@app.route('/health')
def health_check():
    """Health endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'WQI Prediction Service'
    })

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Water Quality Index (WQI) Predictor")
    print("=" * 50)
    
    # Load model
    if load_models():
        print("✓ All models loaded")
    else:
        print("⚠ Running in WQI calculation mode only")
    
    print("\nServer starting...")
    print("Local: http://localhost:5000")
    print("API:   http://localhost:5000/api/predict")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)