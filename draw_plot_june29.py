import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ✅ CSV 파일 경로
csv_path = "socket_text_time.csv"
filename_base = os.path.splitext(os.path.basename(csv_path))[0] 

# Load and clean datetime
df = pd.read_csv(csv_path)
df["Date/Time"] = df["Date/Time"].str.strip().str.replace(" 24:", " 00:")
mask_24 = df["Date/Time"].str.contains(" 00:") & df.duplicated("Date/Time", keep=False)
df.loc[mask_24, "Date/Time"] = pd.to_datetime(df.loc[mask_24, "Date/Time"], format="%m/%d %H:%M:%S") + pd.Timedelta(days=1)
df.loc[~mask_24, "Date/Time"] = pd.to_datetime(df.loc[~mask_24, "Date/Time"], format="%m/%d %H:%M:%S")
df["datetime"] = pd.to_datetime(df["Date/Time"])

# Target columns
temp_col = "THERMAL ZONE: STORY 6 WEST PERIMETER SPACE:Zone Air Temperature [C](TimeStep)"
setpoint_col = "THERMAL ZONE: STORY 6 WEST PERIMETER SPACE:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"
df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
df[setpoint_col] = pd.to_numeric(df[setpoint_col], errors="coerce")

# Remove 0시 데이터만 제외
df_filtered = df[df["datetime"].dt.time != pd.to_datetime("00:00:00").time()].copy()

# 날짜별 정오 기준 경계선과 라벨용 정보
df_filtered["date_only"] = df_filtered["datetime"].dt.date
day_boundaries = df_filtered.groupby("date_only")["datetime"].min().tolist()

# ❗ Shift setpoint one timestep forward (i.e., T₁ setpoint is drawn at T₀)
df_filtered["setpoint_shifted"] = df_filtered[setpoint_col].shift(-1)

# Plot 전체 기간
plt.figure(figsize=(16, 6))
plt.plot(df_filtered["datetime"], df_filtered[temp_col], color="tab:red", label="Zone Air Temperature")
plt.step(df_filtered["datetime"], df_filtered["setpoint_shifted"], color="tab:blue", linestyle='--', where='post', label="Cooling Setpoint")

for boundary in day_boundaries:
    plt.axvline(x=boundary, color='black', linestyle='--', linewidth=1)

start_time = pd.Timestamp(df_filtered["date_only"].iloc[0]) + pd.Timedelta(hours=6)
plt.axvline(x=start_time, color='black', linestyle='-', linewidth=2.5)
plt.annotate('→ Control starts',
             xy=(start_time, 25), xytext=(start_time + pd.Timedelta(hours=0.5), 25),
             textcoords='data',
             fontsize=16, fontname="Arial", color="black", ha='left', va='center')

plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Hh'))
plt.xlim(df_filtered["datetime"].min(), df_filtered["datetime"].max())

plt.xlabel("Time of Day", fontsize=20, fontname="Arial")
plt.ylabel("Temperature [°C]", fontsize=20, fontname="Arial")
plt.xticks(fontsize=20, fontname="Arial")
plt.yticks(fontsize=20, fontname="Arial")
plt.ylim(19, 25.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=18, loc='upper right')
plt.tight_layout()

plt.savefig(f"{filename_base}.png", dpi=300)
plt.show()

# Plot 둘째 날만
second_day = df_filtered["date_only"].unique()[2]
df_day2 = df_filtered[df_filtered["date_only"] == second_day]

plt.figure(figsize=(16, 6))
plt.plot(df_day2["datetime"], df_day2[temp_col], color="tab:red", label="Zone Air Temperature")
plt.step(df_day2["datetime"], df_day2["setpoint_shifted"], color="tab:blue", linestyle='--', where='post', label="Cooling Setpoint")

plt.xlabel("Time of Day", fontsize=20, fontname="Arial")
plt.ylabel("Temperature [°C]", fontsize=20, fontname="Arial")
plt.xticks(fontsize=20, fontname="Arial")
plt.yticks(fontsize=20, fontname="Arial")
plt.ylim(19, 25.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=18, loc='upper right')
plt.tight_layout()

plt.savefig(f"{filename_base}_day2.png", dpi=300)
plt.show()
