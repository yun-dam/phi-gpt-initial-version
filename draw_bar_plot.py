import pandas as pd
import matplotlib.pyplot as plt

# 파일 리스트와 레이블 정의 (줄바꿈 포함)
file_paths = ["socket_fixed.csv", "socket_MPC_JUNE28.csv", "socket_MPC_JUNE28.csv", "socket_fixed.csv"]
labels = [
    "Fixed",
    "MPC",
    "phiGPT\nw/ Text",
    "phiGPT\nw/ Text+Time"
]
patterns = ['.', 'x', '/', '\\']
colors = ['gray', 'red', 'skyblue', 'blue']

# 측정 대상 컬럼
energy_col = "ADU VAV HW RHT 47:Zone Air Terminal Sensible Cooling Energy [J](TimeStep)"
pmv_col = "THERMAL ZONE: STORY 6 WEST PERIMETER SPACE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)"

# 결과 저장
energy_kwh_totals = []
pmv_violation_rates = []

for file in file_paths:
    df = pd.read_csv(file)
    df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce')
    df[pmv_col] = pd.to_numeric(df[pmv_col], errors='coerce')

    total_kwh = df[energy_col].sum() / 3600000  # J -> kWh
    energy_kwh_totals.append(total_kwh)

    total_steps = df[pmv_col].notna().sum()
    violations = (df[pmv_col].abs() >= 1.0).sum()
    violation_rate = (violations / total_steps) * 100 if total_steps > 0 else 0
    pmv_violation_rates.append(violation_rate)

# Plot (1단 폭에 맞게 그림 크기와 bar width 조정)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1단 폭 = 약 6.8 inches

bar_width = 0.6
x_pos = list(range(len(labels)))

# 첫 번째 subplot: Cooling Energy
bars1 = axs[0].bar(x_pos, energy_kwh_totals, width=bar_width, color=colors, edgecolor='black')
for bar, pattern in zip(bars1, patterns):
    bar.set_hatch(pattern)

axs[0].set_ylabel("Cooling Energy [kWh]", fontsize=20, fontname="Arial")
axs[0].set_xticks(x_pos)
axs[0].set_xticklabels(labels, fontsize=18, fontname="Arial")
axs[0].tick_params(axis='y', labelsize=18)
axs[0].set_ylim(0, max(energy_kwh_totals) * 1.1)

# 두 번째 subplot: PMV Comfort Violation Rate
bars2 = axs[1].bar(x_pos, pmv_violation_rates, width=bar_width, color=colors, edgecolor='black')
for bar, pattern in zip(bars2, patterns):
    bar.set_hatch(pattern)

axs[1].set_ylabel("PMV Violation [%]", fontsize=20, fontname="Arial")
axs[1].set_xticks(x_pos)
axs[1].set_xticklabels(labels, fontsize=18, fontname="Arial")
axs[1].tick_params(axis='y', labelsize=18)
axs[1].set_ylim(0, max(pmv_violation_rates) * 1.1)

plt.tight_layout(pad=1.2, w_pad=1.0)
plt.savefig("energy_pmv_subplot_wrapped.png", dpi=300, bbox_inches='tight')
plt.show()
