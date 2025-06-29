import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load and clean datetime
df = pd.read_csv("socket_fixed.csv")
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

# Plot
plt.figure(figsize=(16, 6))

# 온도 연속 곡선
plt.plot(df_filtered["datetime"], df_filtered[temp_col], color="tab:red", label="Zone Temperature")

# 세트포인트 점선 스텝
plt.step(df_filtered["datetime"], df_filtered[setpoint_col], color="tab:blue", linestyle='--', where='post', label="Cooling Setpoint")

# 날짜 경계선
for boundary in day_boundaries:
    plt.axvline(x=boundary, color='black', linestyle='--', linewidth=1)

# # 날짜 라벨
# for date in day_boundaries:
#     noon_time = pd.Timestamp(date) + pd.Timedelta(hours=12)
#     if noon_time in df_filtered["datetime"].values:
#         plt.text(noon_time, df_filtered[temp_col].max() + 0.3,
#                  pd.Timestamp(date).strftime('%m-%d'),
#                  ha='center', va='bottom', fontsize=18, fontname="Arial")

# X축 설정
plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Hh'))

# X축 범위 조정 (공백 제거)
plt.xlim(df_filtered["datetime"].min(), df_filtered["datetime"].max())

# 스타일
# plt.title("Zone Air Temperature and Cooling Setpoint", fontsize=24, fontname="Arial")
plt.xlabel("Time of Day", fontsize=20, fontname="Arial")
plt.ylabel("Temperature [°C]", fontsize=20, fontname="Arial")
plt.xticks(fontsize=20, fontname="Arial")
plt.yticks(fontsize=20, fontname="Arial")
plt.ylim(19, 25)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig("fixed_june29.png", dpi=300)
