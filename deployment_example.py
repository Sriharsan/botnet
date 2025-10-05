

import joblib
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

scaler = joblib.load("robust_scaler.joblib")
rf_model = joblib.load("enhanced_rf_model.joblib")
ensemble = joblib.load("novel_adaptive_ensemble.joblib")
cnn_model = tf.keras.models.load_model("novel_multiscale_attention_cnn.h5")

def predict_botnet(new_sample):
   
    # Preprocess
    X_scaled = scaler.transform(new_sample)
    
    # Add graph features (simplified for deployment)
    # In production, maintain a reference dataset for k-NN
    
    # Get predictions from both models
    cnn_pred = cnn_model.predict(X_scaled.reshape(-1, X_scaled.shape[1], 1))
    ensemble_pred = ensemble.predict_proba(X_scaled)
    
    # Apply hybrid strategy
    cnn_confidence = abs(cnn_pred[0][0] - 0.5) * 2
    
    if cnn_confidence > 0.15:
        final_prob = cnn_pred[0][0]
    else:
        final_prob = ensemble_pred[0]
    
    prediction = 1 if final_prob >= 0.5 else 0
    confidence = final_prob if prediction == 1 else (1 - final_prob)
    
    return prediction, confidence

