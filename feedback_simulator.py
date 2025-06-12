# ─── Path Configuration ───────────────────────────────────────
idd_file_path    = "./ep-model/feedback_simulator/Energy+.idd"
idf_in_path      = "./ep-model/feedback_simulator/gates_feedback_base.idf"
idf_out_path     = "./ep-model/feedback_simulator/gates_feedback_updated.idf"
epw_path         = "./ep-model/feedback_simulator/USA_MT_Charlie.Stanford.720996_TMYx.2007-2021.epw"
csv_path         = "./logs/phi_gpt_log_test.csv"
output_directory = "./ep-model/feedback_simulator/output"

import os
import shutil
import pandas as pd
from eppy.modeleditor import IDF
import re

def update_and_run_csp_eppy(idf_in, idf_out, idd_path, epw, csv_file):
    """
    Update temperature values in a Schedule:Compact object using CSV input,
    and run the simulation using eppy.
    """
    # Step 1: Read the last 4 rows from CSV
    df = pd.read_csv(csv_file, encoding='cp949')
    if not {'hour', 'minute', 'T_set'}.issubset(df.columns):
        raise KeyError("CSV must include 'hour', 'minute', and 'T_set' columns.")

    last_rows = df.tail(4).copy()
    time_temp_mappings = {}

    # Build a time-to-temperature mapping (rounded to 30 minutes)
    for _, row in last_rows.iterrows():
        h, m, t = int(row.hour), int(row.minute), float(row.T_set)
        total_minutes = h * 60 + m + 30
        until_hour, until_minute = divmod(total_minutes, 60)

        if until_hour == 24 and until_minute == 0:
            until_str = "24:00"
        else:
            until_str = f"{until_hour % 24:02d}:{until_minute:02d}"

        time_temp_mappings[until_str] = t

    print("=== Time-Temperature Mapping from CSV ===")
    for k, v in time_temp_mappings.items():
        print(f"  • Until: {k} -> Temp: {v}°C")

    # Step 2: Load IDF
    IDF.setiddname(idd_path)
    idf = IDF(idf_in, epw)

    # Step 3: Find 'csp' schedule
    target_schedule = next((s for s in idf.idfobjects['SCHEDULE:COMPACT'] if getattr(s, 'Name', '') == 'csp'), None)
    if not target_schedule:
        print("❌ 'csp' Schedule:Compact not found.")
        print("Available schedules:")
        for i, s in enumerate(idf.idfobjects['SCHEDULE:COMPACT']):
            print(f"  • {getattr(s, 'Name', f'unnamed_{i}')}")
        return
    print("✅ Found 'csp' Schedule:Compact object.")

    # Step 4: Analyze fields
    until_fields = []
    for i in range(1, 100):
        field = f"Field_{i}"
        if hasattr(target_schedule, field):
            value = getattr(target_schedule, field)
            if i <= 10:
                print(f"  • {field}: {value}")
            if isinstance(value, str) and "Until:" in value:
                until_fields.append((i, value))
        else:
            break
    print(f"Total fields found: {len(until_fields)} with 'Until:' pattern.")

    # Step 5: Update temperature values for matched time entries
    changes = 0
    for i, text in until_fields:
        match = re.search(r'Until:\s*(\d{1,2}:\d{2})', text)
        if match:
            time_str = match.group(1)
            if time_str in time_temp_mappings:
                temp_field = f"Field_{i + 1}"
                if hasattr(target_schedule, temp_field):
                    old = getattr(target_schedule, temp_field)
                    new = f"{time_temp_mappings[time_str]:.1f}"
                    setattr(target_schedule, temp_field, new)
                    print(f"  ✅ Updated: Until {time_str} -> {old}°C → {new}°C")
                    changes += 1
    if not changes:
        print("⚠️ No matching 'Until:' times found.")
        for _, v in until_fields[:5]:
            print(f"  • {v}")
    else:
        print(f"\n✅ {changes} temperature values updated.")

    # Step 6: Save updated IDF
    os.makedirs(os.path.dirname(idf_out), exist_ok=True)
    idf.saveas(idf_out)
    print(f"✅ Updated IDF saved to: {idf_out}")
    if os.path.exists(idf_in):
        shutil.copy(idf_in, idf_in + ".bak")
        print(f"✅ Backup created at: {idf_in}.bak")

    # Step 7: Run the simulation
    try:
        os.makedirs(output_directory, exist_ok=True)
        idf.run(output_directory=output_directory)
        print(f"\n🎉 Simulation completed!")
        print(f"  • Output Directory: {output_directory}")
    except Exception as e:
        print(f"\n❌ Error during simulation run: {e}")

def debug_schedule_structure(idf_path, idd_path, schedule_name="csp"):
    """
    Print field structure of a Schedule:Compact object for debugging.
    """
    print(f"=== Debugging Schedule:Compact '{schedule_name}' ===")
    IDF.setiddname(idd_path)
    idf = IDF(idf_path)

    schedules = idf.idfobjects['SCHEDULE:COMPACT']
    schedule = next((s for s in schedules if getattr(s, 'Name', '') == schedule_name), None)

    if not schedule:
        print(f"❌ Schedule '{schedule_name}' not found.")
        print("Available schedules:")
        for s in schedules:
            print(f"  • {getattr(s, 'Name', 'unnamed')}")
        return

    print(f"✅ Found schedule: {schedule_name}\n=== Fields ===")
    until_count = 0
    for i in range(1, 200):
        field = f"Field_{i}"
        if hasattr(schedule, field):
            val = getattr(schedule, field)
            if isinstance(val, str) and "Until:" in val:
                until_count += 1
                print(f"  {field}: {repr(val)} [UNTIL]")
            elif i <= 20:
                print(f"  {field}: {repr(val)}")
        else:
            print(f"Stopped at field {i} (total fields: {i-1})")
            break
    print(f"\n📊 Total 'Until:' patterns found: {until_count}")

def revise_csp_schedule_simple(idf_path, idd_path, time_temp_dict, output_path=None):
    """
    Simplified version of updating 'csp' schedule temperatures.

    Args:
        time_temp_dict (dict): Mapping of time ("HH:MM") to temperature.
    """
    if output_path is None:
        output_path = idf_path.replace(".idf", "_updated.idf")

    IDF.setiddname(idd_path)
    idf = IDF(idf_path)
    csp_schedule = next((s for s in idf.idfobjects['SCHEDULE:COMPACT'] if getattr(s, 'Name', '') == 'csp'), None)

    if not csp_schedule:
        print("❌ 'csp' schedule not found.")
        return

    print("✅ Found 'csp' schedule. Updating values...")
    changes = 0
    for i in range(1, 200):
        field = f"Field_{i}"
        if not hasattr(csp_schedule, field):
            break
        val = getattr(csp_schedule, field)
        if isinstance(val, str) and "Until:" in val:
            match = re.search(r'Until:\s*(\d{1,2}:\d{2})', val)
            if match:
                time_str = match.group(1)
                if time_str in time_temp_dict:
                    next_field = f"Field_{i+1}"
                    if hasattr(csp_schedule, next_field):
                        old_temp = getattr(csp_schedule, next_field)
                        new_temp = f"{time_temp_dict[time_str]:.1f}"
                        setattr(csp_schedule, next_field, new_temp)
                        changes += 1
                        print(f"  • Until: {time_str} -> {old_temp}°C → {new_temp}°C")
    print(f"✅ Total updates: {changes}")
    idf.saveas(output_path)
    print(f"✅ File saved: {output_path}")
    return idf

if __name__ == "__main__":
    print("Step 1: Debug schedule structure")
    debug_schedule_structure(idf_in_path, idd_file_path, "csp")

    print("\n" + "="*60)
    print("Step 2: Update temperatures and run simulation")
    update_and_run_csp_eppy(
        idf_in   = idf_in_path,
        idf_out  = idf_out_path,
        idd_path = idd_file_path,
        epw      = epw_path,
        csv_file = csv_path
    )
