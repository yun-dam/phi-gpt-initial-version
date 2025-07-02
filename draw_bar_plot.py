import pandas as pd
import matplotlib.pyplot as plt

# 파일 리스트와 레이블 정의 (줄바꿈 포함)
file_paths = [
    "socket_fixed.csv",
    "socket_fixed24.csv",
    "socket_MPC_July1.csv",
    "socket_Text.csv",
    "socket_text_time_july2.csv"
]

labels = [
    "Fixed\n(23℃)",
    "Fixed\n(24℃)",
    "MPC",
    "phiGPT\n(Text only)",
    "phiGPT\n(Text+Time)"
]

patterns = ['.', 'x', '/', '\\', 'o']
colors = ['gray', 'lightgray', 'red', 'skyblue', 'blue']

# 측정 대상 컬럼
datetime_col = "Date/Time"
energy_col = "ADU VAV HW RHT 47:Zone Air Terminal Sensible Cooling Energy [J](TimeStep)"
temp_col = "THERMAL ZONE: STORY 6 WEST PERIMETER SPACE:Zone Air Temperature [C](TimeStep)"
pmv_col = "THERMAL ZONE: STORY 6 WEST PERIMETER SPACE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)"

# 결과 저장
energy_kwh_totals = []
pmv_violation_rates = []

for file in file_paths:
    df = pd.read_csv(file)
    df[datetime_col] = df[datetime_col].str.strip().str.replace(" 24:", " 00:")
    df["datetime"] = pd.to_datetime(df[datetime_col], format="%m/%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["datetime"])

    # 기준 시간: 첫째 날의 06시부터
    start_date = df["datetime"].dt.date.min()
    start_time = pd.Timestamp.combine(start_date, pd.to_datetime("06:00:00").time())
    df = df[df["datetime"] >= start_time]

    # 수치 변환
    df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce')
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df[pmv_col] = pd.to_numeric(df[pmv_col], errors='coerce')

    total_kwh = df[energy_col].sum() / 3600000  # J → kWh
    energy_kwh_totals.append(total_kwh)

    # 🕒 6시~22시까지만 필터링
    df["hour"] = df["datetime"].dt.hour
    df_comfort = df[(df["hour"] >= 6) & (df["hour"] < 22)]

    # PMV comfort violation (|PMV| ≥ 1.0)
    total_steps = df_comfort[pmv_col].notna().sum()
    violations = (df_comfort[pmv_col].abs() >= 1.0).sum()
    violation_rate = (violations / total_steps) * 100 if total_steps > 0 else 0
    pmv_violation_rates.append(violation_rate)

# 기준값 (Case #1 = Fixed 23℃)
base_energy = energy_kwh_totals[0]
base_violation = pmv_violation_rates[0]

# 출력
print("=== Cooling Energy (kWh) from 06:00 ===")
for i, label in enumerate(labels):
    print(f"{label.replace(chr(10), ' ')}: {energy_kwh_totals[i]:.2f} kWh", end='')
    if i > 0:
        delta = (energy_kwh_totals[i] - base_energy) / base_energy * 100
        print(f"   (Δ {delta:+.2f}%)")
    else:
        print()

print("\n=== PMV Violation Rate (%) from 06:00 to 22:00 ===")
for i, label in enumerate(labels):
    print(f"{label.replace(chr(10), ' ')}: {pmv_violation_rates[i]:.2f} %", end='')
    if i > 0:
        delta = (pmv_violation_rates[i] - base_violation) / base_violation * 100 if base_violation != 0 else float('inf')
        print(f"   (Δ {delta:+.2f}%)")
    else:
        print()

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
bar_width = 0.6
x_pos = list(range(len(labels)))

# Cooling Energy Plot
bars1 = axs[0].bar(x_pos, energy_kwh_totals, width=bar_width, color=colors, edgecolor='black')
for bar, pattern in zip(bars1, patterns):
    bar.set_hatch(pattern)
axs[0].set_ylabel("Cooling Energy [kWh]", fontsize=20, fontname="Arial")
axs[0].set_xticks(x_pos)
axs[0].set_xticklabels(labels, fontsize=18, fontname="Arial")
axs[0].tick_params(axis='y', labelsize=18)
axs[0].set_ylim(0, max(energy_kwh_totals) * 1.1)

# PMV Violation Plot
bars2 = axs[1].bar(x_pos, pmv_violation_rates, width=bar_width, color=colors, edgecolor='black')
for bar, pattern in zip(bars2, patterns):
    bar.set_hatch(pattern)
axs[1].set_ylabel("PMV Violation [%]", fontsize=20, fontname="Arial")
axs[1].set_xticks(x_pos)
axs[1].set_xticklabels(labels, fontsize=18, fontname="Arial")
axs[1].tick_params(axis='y', labelsize=18)
axs[1].set_ylim(0, max(pmv_violation_rates) * 1.1)

plt.tight_layout(pad=1.2, w_pad=1.0)
plt.savefig("energy_pmv_violation_6to22.png", dpi=300, bbox_inches='tight')
plt.show()
