import time
import numpy as np
import pandas as pd
from joblib import load

# Load the models
model_xgb = load('trained_models/model_xgb.joblib')
model_rf = load('trained_models/model_rf.joblib')

def round_to_next_odd(n):
    rounded = np.ceil(n)
    return np.where(rounded % 2 == 1, rounded, rounded + 1)

def predict_parameters(data_frame):
    # Predict using Model 1
    d_predictions = model_xgb.predict(data_frame)

    # Prepare Input for Model 2
    input_for_rf2 = np.column_stack((d_predictions, data_frame['l']))

    # Predict using Model 2
    r_predictions = model_rf.predict(input_for_rf2)

    return d_predictions, r_predictions

if __name__ == "__main__":

    # Change the data frame according to need
    data_frame = pd.DataFrame({
        # four types physical error rates
    'p_dep': [0.0005, 0.0007, 2.4E-04, 2.0E-4, 2.6E-4], # depolarizing error rate
    'p_gate': [0.0005, 0.0007, 7.7E-03, 6.6E-3, 8.5E-3], # gate error rate
    'p_res': [0.0005, 0.0007, 2.4E-02, 2.0E-2, 2.6E-2], # reset error rate
    'p_read': [0.0005, 0.0007, 2.4E-02, 2.0E-2, 2.7E-2], # readout error rate

    'l': [0.0000015, 0.0000017, 5.26E-08, 5.26E-08, 5.26E-08] # logical error rate
    })
    start_time = time.time()
    d_predictions, r_predictions = predict_parameters(data_frame)
    elapsed_time_model1 = time.time() - start_time

    print("Predictions for 'd':", round_to_next_odd(d_predictions))
    print("Predictions for 'r':", np.ceil(r_predictions))
    print(f"Time taken for Model 1 prediction: {elapsed_time_model1:.4f} seconds")