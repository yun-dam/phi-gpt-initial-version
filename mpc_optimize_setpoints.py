import numpy as np
import itertools
import joblib
from tensorflow.keras.models import load_model

ALLOWED_SETPOINTS = [22.0, 23.0, 24.0]

def evaluate(seq, model, scaler_x, scaler_y, past_input,
             w_energy=1.0, w_comfort=100000.0, target_min=22.5, target_max=23.5):
    """
    Evaluate a 4-step setpoint sequence based on predicted energy and comfort penalty.
    """
    x_input = np.concatenate([past_input, seq])
    x_scaled = scaler_x.transform(x_input.reshape(1, -1))
    y_scaled = model.predict(x_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_scaled)[0]

    T_in_pred = y_pred[:4]
    Energy_pred = y_pred[4:]

    # Comfort penalty: sum of squared violations
    comfort_penalty = 0.0
    for T in T_in_pred:
        if T < target_min:
            comfort_penalty += (target_min - T) ** 2
        elif T > target_max:
            comfort_penalty += (T - target_max) ** 2

    total_energy = np.sum(Energy_pred)
    total_score = w_energy * total_energy + w_comfort * comfort_penalty

    return total_score

def find_best_setpoint(model_path="trained_model", w_energy=1.0, w_comfort=100000.0):
    """
    Find the best 4-step setpoint sequence that balances energy and comfort.
    """
    model = load_model(f"{model_path}/model_6f_n_u_p_s.h5", compile=False)
    scaler_x = joblib.load(f"{model_path}/scaler_x_6f_n_u_p_s.pkl")
    scaler_y = joblib.load(f"{model_path}/scaler_y_6f_n_u_p_s.pkl")
    past_input = np.load(f"{model_path}/latest_input_6f_n_u_p_s.npy")

    candidates = list(itertools.product(ALLOWED_SETPOINTS, repeat=4))
    best_seq, best_score = None, float("inf")

    for seq in candidates:
        score = evaluate(seq, model, scaler_x, scaler_y, past_input,
                         w_energy=w_energy, w_comfort=w_comfort)
        if score < best_score:
            best_score = score
            best_seq = seq

    if best_seq is None:
        raise ValueError("❌ No valid setpoint sequence found.")

    print("✅ Best 4-step setpoints:", best_seq)
    print("⚖️  Weighted total score:", best_score)

    return best_seq, best_score

if __name__ == "__main__":
    # Example: prioritize comfort with higher penalty weight
    find_best_setpoint("trained_model", w_energy=1.0, w_comfort=100000.0)
