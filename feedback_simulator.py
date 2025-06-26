import os
import shutil
import pandas as pd
import re
from datetime import datetime, timedelta
from eppy.modeleditor import IDF
import shutil


# Path configuration
idd_file_path    = "./ep-model/feedback_simulator/Energy+.idd"
idf_in_path      = "./ep-model/feedback_simulator/gates_feedback_base.idf"
idf_out_path     = "./ep-model/feedback_simulator/gates_feedback_updated.idf"
epw_path         = "./USA_CA_Palo.Alto.AP.724937_TMYx.2009-2023.epw"
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


import time

def find_latest_phi_gpt_log(log_dir: str) -> str:
    log_files = [
        f for f in os.listdir(log_dir)
        if f.startswith("phi_gpt_log_") and f.endswith(".csv")
    ]
    if not log_files:
        raise FileNotFoundError("No phi_gpt_log_*.csv file found in log directory.")

    latest_file = max(log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
    full_path = os.path.join(log_dir, latest_file)
    temp_path = os.path.join(log_dir, "latest_tmp_log.csv")

    # 🚨 Retry copy up to 5 times
    for attempt in range(5):
        try:
            shutil.copy2(full_path, temp_path)
            return temp_path
        except PermissionError:
            print(f"⚠️ PermissionError: retrying... ({attempt+1}/5)")
            time.sleep(0.5)

    raise PermissionError(f"Failed to copy log file after 5 attempts: {full_path}")


def extract_future_results(log_csv_path, sim_csv_path, zone_name):
    import pandas as pd
    from datetime import datetime, timedelta

    # 🧩 내부에서 zone_to_adu_name 정의
    def zone_to_adu_name(zone_name):
        mapping = {
            "THERMAL ZONE: STORY 1 EAST CORE SPACE": 0,
            "THERMAL ZONE: STORY 1 EAST LOWER PERIMETER SPACE": 1,
            "THERMAL ZONE: STORY 1 EAST UPPER PERIMETER SPACE": 2,
            "THERMAL ZONE: STORY 1 NORTH LOWER PERIMETER SPACE": 3,
            "THERMAL ZONE: STORY 1 NORTH UPPER PERIMETER SPACE": 4,
            "THERMAL ZONE: STORY 1 SOUTH PERIMETER SPACE": 5,
            "THERMAL ZONE: STORY 1 WEST CORE SPACE": 6,
            "THERMAL ZONE: STORY 1 WEST PERIMETER SPACE": 7,
            "THERMAL ZONE: STORY 2 EAST CORE SPACE": 8,
            "THERMAL ZONE: STORY 2 EAST LOWER PERIMETER SPACE": 9,
            "THERMAL ZONE: STORY 2 EAST UPPER PERIMETER SPACE": 10,
            "THERMAL ZONE: STORY 2 NORTH LOWER PERIMETER SPACE": 11,
            "THERMAL ZONE: STORY 2 NORTH UPPER PERIMETER SPACE": 12,
            "THERMAL ZONE: STORY 2 SOUTH PERIMETER SPACE": 13,
            "THERMAL ZONE: STORY 2 WEST CORE SPACE": 14,
            "THERMAL ZONE: STORY 2 WEST PERIMETER SPACE": 15,
            "THERMAL ZONE: STORY 3 EAST CORE SPACE": 16,
            "THERMAL ZONE: STORY 3 EAST LOWER PERIMETER SPACE": 17,
            "THERMAL ZONE: STORY 3 EAST UPPER PERIMETER SPACE": 18,
            "THERMAL ZONE: STORY 3 NORTH LOWER PERIMETER SPACE": 19,
            "THERMAL ZONE: STORY 3 NORTH UPPER PERIMETER SPACE": 20,
            "THERMAL ZONE: STORY 3 SOUTH PERIMETER SPACE": 21,
            "THERMAL ZONE: STORY 3 WEST CORE SPACE": 22,
            "THERMAL ZONE: STORY 3 WEST PERIMETER SPACE": 23,
            "THERMAL ZONE: STORY 4 EAST CORE SPACE": 24,
            "THERMAL ZONE: STORY 4 EAST LOWER PERIMETER SPACE": 25,
            "THERMAL ZONE: STORY 4 EAST UPPER PERIMETER SPACE": 26,
            "THERMAL ZONE: STORY 4 NORTH LOWER PERIMETER SPACE": 27,
            "THERMAL ZONE: STORY 4 NORTH UPPER PERIMETER SPACE": 28,
            "THERMAL ZONE: STORY 4 SOUTH PERIMETER SPACE": 29,
            "THERMAL ZONE: STORY 4 WEST CORE SPACE": 30,
            "THERMAL ZONE: STORY 4 WEST PERIMETER SPACE": 31,
            "THERMAL ZONE: STORY 5 EAST CORE SPACE": 32,
            "THERMAL ZONE: STORY 5 EAST LOWER PERIMETER SPACE": 33,
            "THERMAL ZONE: STORY 5 EAST UPPER PERIMETER SPACE": 34,
            "THERMAL ZONE: STORY 5 NORTH LOWER PERIMETER SPACE": 35,
            "THERMAL ZONE: STORY 5 NORTH UPPER PERIMETER SPACE": 36,
            "THERMAL ZONE: STORY 5 SOUTH PERIMETER SPACE": 37,
            "THERMAL ZONE: STORY 5 WEST CORE SPACE": 38,
            "THERMAL ZONE: STORY 5 WEST PERIMETER SPACE": 39,
            "THERMAL ZONE: STORY 6 EAST CORE SPACE": 40,
            "THERMAL ZONE: STORY 6 EAST LOWER PERIMETER SPACE": 41,
            "THERMAL ZONE: STORY 6 EAST UPPER PERIMETER SPACE": 42,
            "THERMAL ZONE: STORY 6 NORTH LOWER PERIMETER SPACE": 43,
            "THERMAL ZONE: STORY 6 NORTH UPPER PERIMETER SPACE": 44,
            "THERMAL ZONE: STORY 6 SOUTH PERIMETER SPACE": 45,
            "THERMAL ZONE: STORY 6 WEST CORE SPACE": 46,
            "THERMAL ZONE: STORY 6 WEST PERIMETER SPACE": 47,
        }
        index = mapping.get(zone_name.upper())
        if index is None:
            raise ValueError(f"Unknown zone name: {zone_name}")
        return "ADU VAV HW RHT" if index == 0 else f"ADU VAV HW RHT {index}"

    # Set dynamic terminal column name
    adu_name = zone_to_adu_name(zone_name)
    terminal_col = f"{adu_name}:Zone Air Terminal Sensible Cooling Energy [J](TimeStep)"

    # Load log file
    df_log = pd.read_csv(log_csv_path)
    last_hour = int(df_log["hour"].iloc[-1])
    last_min = int(df_log["minute"].iloc[-1])

    # Load simulation CSV
    df_sim = pd.read_csv(sim_csv_path)
    df_sim.columns = [col.strip('"') for col in df_sim.columns]

    # Fix 24:00:00
    def fix_24_hour(dt_str):
        dt_str = dt_str.strip()
        if "24:00:00" in dt_str:
            date_part = dt_str.split()[0]
            new_dt = datetime.strptime(date_part + " 00:00:00", "%m/%d %H:%M:%S") + timedelta(days=1)
        else:
            new_dt = datetime.strptime(dt_str, "%m/%d %H:%M:%S")
        return new_dt

    df_sim["datetime"] = df_sim["Date/Time"].apply(fix_24_hour)

    # Match target time
    target_min = last_hour * 60 + last_min
    df_sim["delta_min"] = df_sim["datetime"].dt.hour * 60 + df_sim["datetime"].dt.minute
    df_sim["delta_to_last"] = abs(df_sim["delta_min"] - target_min)

    anchor_idx = df_sim["delta_to_last"].idxmin()
    target_indices = [anchor_idx + i for i in [1, 2, 3, 4]]  # +30, 60, 90, 120 mins

    df_future = df_sim.iloc[[i for i in target_indices if i < len(df_sim)]].copy()

    # Extract columns
    temp_col = f"{zone_name}:Zone Air Temperature [C](TimeStep)"
    setp_col = f"{zone_name}:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"
    terminal_col = f"{adu_name}:Zone Air Terminal Sensible Cooling Energy [J](TimeStep)"
    outdoor_col = "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"

    result_cols = ["datetime"]
    col_map = {}

    for col in [temp_col, setp_col, terminal_col, outdoor_col]:
        if col in df_sim.columns:
            if col == outdoor_col:
                col_map[col] = "T_out"
            elif col == temp_col:
                col_map[col] = "T_in"
            elif col == setp_col:
                col_map[col] = "T_set"
            elif col == terminal_col:
                col_map[col] = "Energy_J"
            result_cols.append(col)

    df_future = df_future[result_cols].rename(columns=col_map)
    
    return df_future


def find_latest_phi_gpt_log_path(log_dir: str) -> str:
    """
    Find the most recently modified phi_gpt_log_*.csv file in the given directory or its subdirectories.
    """
    candidate_files = []

    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith("phi_gpt_log_") and f.endswith(".csv"):
                full_path = os.path.join(root, f)
                candidate_files.append(full_path)

    if not candidate_files:
        raise FileNotFoundError("No phi_gpt_log_*.csv file found in or under log directory.")

    latest_file = max(candidate_files, key=os.path.getmtime)
    return latest_file


def run_feedback_simulation(setpoints, log_path=None, zone_name=None):
    """
    Main callable function: runs simulation with given setpoints and returns extracted metrics.
    If log_path is None, the latest phi_gpt_log_*.csv file will be used.
    """
    if log_path is None:
        raise ValueError("log_path must be provided (phi_gpt_log_*.csv).")


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
