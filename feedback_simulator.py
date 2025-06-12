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

    df_log = pd.read_csv(log_csv_path, encoding='cp949')
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
    idf.run(output_directory=output_dir)
    return True

def extract_future_results(log_csv_path, sim_csv_path, zone_name):
    df_log = pd.read_csv(log_csv_path, encoding='cp949')
    df_log["datetime"] = pd.to_datetime(
        df_log["hour"].astype(str).str.zfill(2) + ":" + df_log["minute"].astype(str).str.zfill(2),
        format="%H:%M"
    )
    last_time = df_log["datetime"].iloc[-1]

    base_date = datetime.strptime("08/08", "%m/%d")
    full_last_time = base_date.replace(hour=last_time.hour, minute=last_time.minute)
    future_times = [full_last_time + timedelta(minutes=delta) for delta in [30, 60, 90, 120]]
    formatted_times = [f" {dt.strftime('%m/%d')}  {dt.strftime('%H:%M:%S')}" for dt in future_times]

    df_sim = pd.read_csv(sim_csv_path)
    df_sim.columns = [col.strip('"') for col in df_sim.columns]
    df_filtered = df_sim[df_sim["Date/Time"].isin(formatted_times)].copy()
    df_filtered["datetime"] = formatted_times[:len(df_filtered)]

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

    df_filtered = df_filtered[result_cols].rename(columns=col_map)
    return df_filtered

def run_feedback_simulation(setpoints, log_path, zone_name):
    """
    Main callable function: runs simulation with given setpoints and returns extracted metrics.
    """
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

# Optional test block
if __name__ == "__main__":
    test_setpoints = [22.0, 23.0, 23.0, 22.0]
    zone = "THERMAL ZONE: STORY 2 SOUTH PERIMETER SPACE"
    df = run_feedback_simulation(test_setpoints, "./logs/phi_gpt_log_test.csv", zone)
    print("\n🔍 Feedback Simulation Result:")
    print(df.to_string(index=False))
