
# MITS-A-Quantum-Sorcerer-s-Stone-For-Designing-Surface-Codes
MITS, a new methodology to optimize surface code implementations by predicting ideal distance and rounds given target logical error rates and known physical noise levels of the hardware.
=======
# MITS: A Quantum Sorcerer’s Stone For Designing Surface Codes

## Description

MITS accepts the specific noise model of a quantum computer and a target logical error rate as input and outputs the optimal surface code rounds and code distances. This guarantees minimal qubit and gate usage, harmonizing the desired logical error rate with the existing hardware limitations on qubit numbers and gate fidelity. Quantum computers undergo routine calibration, leading to fluctuations in physical error rates and requiring frequent adjustments to surface code parameters. Identifying these optimal parameters through simulations is time-consuming, often taking hours or days. This task is especially challenging due to the need for rapid adjustments following regular calibrations. Our main challenge is quickly finding the optimal mix of code distance and rounds to meet a target logical error rate, considering the quantum system's current physical errors. This is where MITS comes into play, aiming to swiftly find the optimal distance and rounds required for rotated surface codes.

This project is based on the methodologies described in the paper, MITS: A Quantum Sorcerer’s Stone For Designing Surface Codes (https://arxiv.org/abs/2402.11027).

## Packages Needed

The project requires the following Python packages:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- sklearn
- scipy
- math
- time
- joblib

You can install these packages using pip:

## How to Run?

Begin by executing model_training.py to train two separate models: one for predicting distance and another for predicting the rounds of the rotated surface code. Next, navigate to `get_predictions.py` and substitute your own values into data_frame, which accepts the probability of four types of physical errors alongside the target logical error rate. To obtain the predicted parameters, execute `get_predictions.py`. It's possible to predict multiple parameters simultaneously by supplying various error values. For a demonstration of how to run this process, refer to `running_process.ipynb`. The trained models are being saved in the folder `trained_models`.

## Example Data Frame

```
data_frame = pd.DataFrame({
    'p_dep': [0.0005, 0.0007],
    'p_gate': [0.0005, 0.0007],
    'p_res': [0.0005, 0.0007],
    'p_read': [0.0005, 0.0007],
    'l': [0.0000015, 0.0000017]
})
```

## Example Output

```
Predictions for 'd': [7. 7.]
Predictions for 'r': [25. 17.]
Time taken for Model 1 prediction: 0.0099 seconds
```

## Contact

For any questions or inquiries about this project, please contact:

Name: Avimita Chatterjee
Email: amc8313@psu.edu
>>>>>>> 618e6fd (First commit)
