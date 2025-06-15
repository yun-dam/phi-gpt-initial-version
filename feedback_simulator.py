import os
import shutil
import pandas as pd
import re
from datetime import datetime, timedelta
from eppy.modeleditor import IDF

# Path configuration
idd_file_path    = "./ep-model/feedback_simulator/Energy+.idd"
idf_in_path      = "./ep-model/feedback_simulator/gates_feedback_base.idf"
idf_out_path     = "./ep-model/feedback_simulator/gates_feedback_updated.idf"
epw_path         = "./ep-model/feedback_simulator/USA_MT_Charlie.Stanford.720996_TMYx.2007-2021.epw"
output_directory = "./ep-model/feedback_simulator/output"

def update_setpoints_by_time(idf_path, idd_path, log_csv_path, setpoint_list, output_path=None):
    if output_path is None:
        output_path = idf_path.replace(".idf", "_updated.idf")

    if len(setpoint_list) != 4:
        raise ValueError("setpoint_list must contain exactly 4 values.")

    df_log = pd.read_csv(log_csv_path)
    df_log["datetime"] = pd.to_datetime(
        df_log["hour"].astype(str).str.zfill(2) + ":" + df_log["minute"].astype(str).str.zfill(2),
        format="%H:%M"
    )
    last_time = df_log["datetime"].iloc[-1]

    recent_rows = df_log.tail(4).copy()
    recent_rows["T_set"] = recent_rows["T_set"].astype(float)
    recent_temp_map = {}
    for _, row in recent_rows.iterrows():
        time_str = f"{int(row['hour']):02d}:{int(row['minute']):02d}"
        recent_temp_map[time_str] = row["T_set"]

    future_temp_map = {}
    for i, delta in enumerate([30, 60, 90, 120]):
        ft = last_time + timedelta(minutes=delta)
        time_str = "24:00" if ft.hour == 0 and ft.minute == 0 else ft.strftime("%H:%M")
        future_temp_map[time_str] = setpoint_list[i]

    IDF.setiddname(idd_path)
    idf = IDF(idf_path)
    schedules = idf.idfobjects["SCHEDULE:COMPACT"]
    csp_schedule = next((s for s in schedules if getattr(s, "Name", "") == "csp"), None)
    if not csp_schedule:
        return None

    for i in range(1, 200):
        field_name = f"Field_{i}"
        if not hasattr(csp_schedule, field_name):
            break
        value = getattr(csp_schedule, field_name)
        if isinstance(value, str) and value.strip().startswith("Until:"):
            match = re.search(r"Until:\s*(\d{1,2}):(\d{2})", value)
            if match:
                time_str = f"{int(match.group(1)):02d}:{match.group(2)}"
                set_field = f"Field_{i + 1}"
                if hasattr(csp_schedule, set_field):
                    if time_str in recent_temp_map:
                        setattr(csp_schedule, set_field, f"{recent_temp_map[time_str]:.1f}")
                    if time_str in future_temp_map:
                        setattr(csp_schedule, set_field, f"{future_temp_map[time_str]:.1f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    idf.saveas(output_path)
    return output_path

def run_energyplus_simulation(idf_path, idd_path, epw_path, output_dir):
    IDF.setiddname(idd_path)
    idf = IDF(idf_path, epw_path)
    os.makedirs(output_dir, exist_ok=True)
    idf.run(output_directory=output_dir, readvars=True, output_prefix="gates_feedback_updated_")
    return True

def find_latest_phi_gpt_log(log_dir: str) -> str:
    """
    Finds the most recent phi_gpt_log_*.csv file in the given log directory.
    """
    log_files = [
        f for f in os.listdir(log_dir)
        if f.startswith("phi_gpt_log_") and f.endswith(".csv")
    ]
    if not log_files:
        raise FileNotFoundError("No phi_gpt_log_*.csv file found in log directory.")

    latest_file = max(log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
    return os.path.join(log_dir, latest_file)


def extract_future_results(log_csv_path, sim_csv_path, zone_name):
    import pandas as pd
    from datetime import datetime, timedelta

    # Load log file
    df_log = pd.read_csv(log_csv_path)
    last_hour = int(df_log["hour"].iloc[-1])
    last_min = int(df_log["minute"].iloc[-1])

    # Load simulation CSV
    df_sim = pd.read_csv(sim_csv_path)
    df_sim.columns = [col.strip('"') for col in df_sim.columns]

    # Fix 24:00:00 to 00:00:00 next day
    def fix_24_hour(dt_str):
        dt_str = dt_str.strip()
        if "24:00:00" in dt_str:
            date_part = dt_str.split()[0]
            new_dt = datetime.strptime(date_part + " 00:00:00", "%m/%d %H:%M:%S") + timedelta(days=1)
        else:
            new_dt = datetime.strptime(dt_str, "%m/%d %H:%M:%S")
        return new_dt

    # Parse datetime
    df_sim["datetime"] = df_sim["Date/Time"].apply(fix_24_hour)

    # Find the simulation row closest to last (hour:min)
    target_min = last_hour * 60 + last_min
    df_sim["delta_min"] = df_sim["datetime"].dt.hour * 60 + df_sim["datetime"].dt.minute
    df_sim["delta_to_last"] = abs(df_sim["delta_min"] - target_min)

    anchor_idx = df_sim["delta_to_last"].idxmin()
    target_indices = [anchor_idx + i for i in [1, 2, 3, 4]]  # 30, 60, 90, 120 mins later

    # Extract available rows
    df_future = df_sim.iloc[[i for i in target_indices if i < len(df_sim)]].copy()

    # Extract key columns
    temp_col = f"{zone_name}:Zone Air Temperature [C](TimeStep)"
    setp_col = f"{zone_name}:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"
    terminal_col = "ADU VAV HW RHT 13:Zone Air Terminal Sensible Cooling Energy [J](TimeStep)"

    result_cols = ["datetime"]
    col_map = {}

    for col in [temp_col, setp_col, terminal_col]:
        if col in df_sim.columns:
            label = "T_in" if "Air Temperature" in col else "T_set" if "Cooling Setpoint" in col else "Energy_J"
            col_map[col] = label
            result_cols.append(col)

    df_future = df_future[result_cols].rename(columns=col_map)
    return df_future


def run_feedback_simulation(setpoints, log_path=None, zone_name="THERMAL ZONE: STORY 2 SOUTH PERIMETER SPACE"):
    """
    Main callable function: runs simulation with given setpoints and returns extracted metrics.
    If log_path is None, the latest phi_gpt_log_*.csv file will be used.
    """
    if log_path is None:
        log_path = find_latest_phi_gpt_log("./logs")

    updated_idf_path = update_setpoints_by_time(
        idf_path=idf_in_path,
        idd_path=idd_file_path,
        log_csv_path=log_path,
        setpoint_list=setpoints,
        output_path=idf_out_path
    )

    if not updated_idf_path:
        raise RuntimeError("Failed to update IDF with new setpoints.")

    run_energyplus_simulation(
        idf_path=updated_idf_path,
        idd_path=idd_file_path,
        epw_path=epw_path,
        output_dir=output_directory
    )

    sim_csv = os.path.join(output_directory, "gates_feedback_updated_out.csv")
    return extract_future_results(log_path, sim_csv, zone_name)


# if __name__ == "__main__":
#     test_setpoints = [22.0, 23.0, 23.0, 22.0]
#     df = run_feedback_simulation(test_setpoints)
#     print("\n🔍 Feedback Simulation Result:")
#     print(df.to_string(index=False))
