import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from feedback_simulator import run_feedback_simulation, find_latest_phi_gpt_log_path
import os
import uuid
import shutil

# Allowed discrete setpoint values
ALLOWED_SETPOINTS = [22.0, 23.0, 24.0]
value_map = {0: 22.0, 1: 23.0, 2: 24.0}

def evaluate_simulation(seq, log_path=None, zone_name=None,
                        w_energy=1.0, w_comfort=7.0,
                        target_min=22.5, target_max=23.5,
                        energy_norm_base=1_000_000.0, comfort_max=4.0):
    """
    Evaluate a 4-step setpoint sequence using EnergyPlus simulation with normalized scoring.
    Applies comfort penalty only when outdoor temperature is 20°C or higher.
    """
    df_result = run_feedback_simulation(seq, log_path=log_path, zone_name=zone_name)
    if df_result is None or df_result.empty:
        return float("inf"), None, None

    T_in = df_result["T_in"].values
    Energy_J = df_result["Energy_J"].values
    if "T_out" not in df_result.columns:
        raise KeyError("❌ 'T_out' column is missing from simulation result. Check extract_future_results() or simulation output.")
    T_out = df_result["T_out"].values

    # Comfort penalty only when T_out >= 20°C
    comfort_penalty = 0.0
    for t_in, t_out in zip(T_in, T_out):
        if t_out >= 20.0:
            comfort_penalty += (
                max(0, target_min - t_in) ** 2 + max(0, t_in - target_max) ** 2
            )

    total_energy = np.sum(Energy_J)

    # Normalized scores
    normalized_energy = total_energy / energy_norm_base
    normalized_comfort = comfort_penalty / comfort_max

    score = w_energy * normalized_energy + w_comfort * normalized_comfort

    print(f"🔍 Evaluation Result → Total Score: {score:.4f}, Energy Score: {normalized_energy:.4f}, Comfort Score: {normalized_comfort:.4f}")

    return score, normalized_energy, normalized_comfort

def find_best_setpoint_by_simulation(log_path=None,
                                     zone_name=None,
                                     w_energy=1.0, w_comfort=7.0):
    """
    Use Genetic Algorithm to find the best 4-step setpoint sequence using normalized score.
    Each simulation uses a copied phi_gpt_log_*.csv inside a unique log directory.
    """
    base_log_path = os.path.abspath("./logs") if log_path is None else os.path.abspath(log_path)
    best_result = {"score": float("inf")}

    def fitness(x):
        seq = [value_map[int(i)] for i in x]

        # 1. 고유 폴더 생성
        unique_log_path = os.path.join(base_log_path, str(uuid.uuid4()))
        os.makedirs(unique_log_path, exist_ok=True)

        # 2. 최신 phi_gpt_log_*.csv 찾아서 복사
        source_log_file = find_latest_phi_gpt_log_path(base_log_path)
        target_log_file = os.path.join(unique_log_path, os.path.basename(source_log_file))
        shutil.copy2(source_log_file, target_log_file)

        # 3. 복사된 로그로 시뮬레이션
        score, energy_score, comfort_score = evaluate_simulation(
            seq,
            log_path=target_log_file,
            zone_name=zone_name,
            w_energy=w_energy,
            w_comfort=w_comfort
        )

        # 저장해두기
        if score < best_result["score"]:
            best_result.update({
                "score": score,
                "energy_score": energy_score,
                "comfort_score": comfort_score,
                "seq": seq
            })

        return score

    varbound = np.array([[0, 2]] * 4)

    algorithm_param = {
        'max_num_iteration': 6,
        'population_size': 8,
        'mutation_probability': 0.2,
        'elit_ratio': 0.2,
        'crossover_probability': 0.6,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 3
    }
    model = ga(
        function=fitness,
        dimension=4,
        variable_type='int',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param,
        convergence_curve=False,
        function_timeout=600
    )
    model.run()

    return best_result["seq"], best_result["score"], best_result["energy_score"], best_result["comfort_score"]

if __name__ == "__main__":
    best_seq, score, energy_score, comfort_score = find_best_setpoint_by_simulation()
    print("\n✅ Best GA Setpoints:", best_seq)
    print(f"⚖️  Normalized Total Score: {score:.4f}")
    print(f"🔋 Energy Score: {energy_score:.4f}")
    print(f"🌡️ Comfort Score: {comfort_score:.4f}")
