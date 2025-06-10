import os
import datetime
import shutil
import socket
import json
import csv
from pyenergyplus.plugin import EnergyPlusPlugin
from collections import deque

class phiGPTSimulator(EnergyPlusPlugin):
    def __init__(self):
        super().__init__()
        self.need_handles = True
        self.cooling_handle = None
        self.T_out_handle = None
        self.T_in_handle = None
        self.prev_min = -1
        self.zone = "THERMAL ZONE: STORY 5 SOUTH PERIMETER SPACE"
        self.state_buffer = deque(maxlen=12)

        self.use_fixed_setpoint = False  # 🔵 Fixed Setpoint Mode 여부
        self.fixed_setpoint_value = 25  # 🔵 Fixed Setpoint 값 (°C)

        # 🔵 로그 디렉토리 및 파일 설정
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        base_dir = os.path.abspath(os.path.dirname(__file__))  # 현재 .py 파일 위치
        self.log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        if self.use_fixed_setpoint:
            fixed_str = str(round(self.fixed_setpoint_value, 1)).replace('.', '')  # 예: 23.3 -> '233'
            filename = f"phi_gpt_log_fixed{fixed_str}_{timestamp}.csv"
        else:
            filename = f"phi_gpt_log_{timestamp}.csv"

        self.log_path = os.path.join(self.log_dir, filename)

        with open(self.log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["month", "day", "hour", "minute", "T_out", "T_in", "T_set", "reason"])

        self.socket_src = os.path.join(base_dir, "socket.csv")
        self.socket_meter_src = os.path.join(base_dir, "socketMeter.csv")

    def on_begin_zone_timestep_before_set_current_weather(self, state) -> int:
        if not self.api.exchange.api_data_fully_ready(state):
            return 0

        if self.need_handles:
            self.cooling_handle = self.api.exchange.get_actuator_handle(
                state, "Zone Temperature Control", "Cooling Setpoint", self.zone
            )
            self.T_out_handle = self.api.exchange.get_variable_handle(
                state, "Site Outdoor Air Drybulb Temperature", "Environment"
            )
            self.T_in_handle = self.api.exchange.get_variable_handle(
                state, "Zone Mean Air Temperature", self.zone
            )

            if self.cooling_handle == -1 or self.T_out_handle == -1 or self.T_in_handle == -1:
                self.api.runtime.issue_severe(state, "[phiGPT] ❌ Failed to get one or more handles.")
                return 0

            self.api.runtime.issue_warning(state, "[phiGPT] ✅ Handles acquired.")
            if self.use_fixed_setpoint:
                self.api.runtime.issue_warning(state, f"[phiGPT] 🔵 Fixed Setpoint Mode Activated: {self.fixed_setpoint_value:.2f}°C")
            else:
                self.api.runtime.issue_warning(state, "[phiGPT] 🧠 LLM Reasoning Mode Activated.")
            self.need_handles = False

        hour = self.api.exchange.hour(state)
        minute = self.api.exchange.minutes(state)

        if minute == self.prev_min:
            return 0
        self.prev_min = minute

        if minute not in (0, 30):
            return 0

        T_out = self.api.exchange.get_variable_value(state, self.T_out_handle)
        T_in = self.api.exchange.get_variable_value(state, self.T_in_handle)
        T_set = self.api.exchange.get_actuator_value(state, self.cooling_handle)

        self.state_buffer.append((T_out, T_in, T_set))
        if len(self.state_buffer) < 12:
            self.api.runtime.issue_warning(state, f"[phiGPT] Buffering... ({len(self.state_buffer)}/12)")
            return 0

        if self.use_fixed_setpoint:
            new_setpoint_c = self.fixed_setpoint_value
            reason = "Fixed setpoint mode (no LLM reasoning)"
        else:
            response = self.query_reasoning_server(list(self.state_buffer))
            if response and "optimal_cooling_setpoint" in response:
                new_setpoint_c = response["optimal_cooling_setpoint"]
                reason = response.get("reason", "N/A")
            else:
                self.api.runtime.issue_warning(state, "[phiGPT] ⚠️ No valid response from reasoning server.")
                reason = "No valid response"
                return 0

        self.api.exchange.set_actuator_value(state, self.cooling_handle, new_setpoint_c)
        self.api.runtime.issue_warning(state, f"[phiGPT] {hour:02}:{minute:02} → Setpoint = {new_setpoint_c:.2f}°C")

        # 🟢 EnergyPlus 시뮬레이션 시간 기반 날짜 정보 추출
        month = self.api.exchange.month(state)
        day = self.api.exchange.day_of_month(state)

        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                month,
                day,
                hour,
                minute,
                round(T_out, 2),
                round(T_in, 2),
                round(new_setpoint_c, 2),
                reason.replace("\n", " ")[:500]
            ])

        return 0

    def query_reasoning_server(self, state_buffer):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", 55555))
                message = json.dumps({"state_buffer": state_buffer})
                s.sendall(message.encode('utf-8'))
                response = s.recv(8192)
                return json.loads(response.decode('utf-8'))
        except Exception as e:
            print(f"[phiGPT] ❌ Socket error: {e}")
            return None

    def on_end_of_simulation(self, state) -> int:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", 55555))
                shutdown_message = json.dumps({"shutdown": True})
                s.sendall(shutdown_message.encode('utf-8'))
                response = s.recv(1024)
                print(f"[phiGPTSimulator] 🔵 Reasoning server response: {response.decode('utf-8')}")
        except Exception as e:
            print(f"[phiGPTSimulator] ⚠️ Failed to send shutdown signal: {e}")

        if self.use_fixed_setpoint:
            fixed_str = str(round(self.fixed_setpoint_value, 1)).replace('.', '')
            socket_dst = os.path.join(self.log_dir, f"socket_fixed{fixed_str}_{timestamp}.csv")
            socket_meter_dst = os.path.join(self.log_dir, f"socketMeter_fixed{fixed_str}_{timestamp}.csv")
        else:
            socket_dst = os.path.join(self.log_dir, f"socket_{timestamp}.csv")
            socket_meter_dst = os.path.join(self.log_dir, f"socketMeter_{timestamp}.csv")

        if os.path.exists(self.socket_src):
            shutil.copy2(self.socket_src, socket_dst)
            self.api.runtime.issue_warning(state, f"[phiGPT] ✅ socket.csv saved as {os.path.basename(socket_dst)}")
        else:
            self.api.runtime.issue_warning(state, "[phiGPT] ⚠️ socket.csv not found.")

        if os.path.exists(self.socket_meter_src):
            shutil.copy2(self.socket_meter_src, socket_meter_dst)
            self.api.runtime.issue_warning(state, f"[phiGPT] ✅ socketMeter.csv saved as {os.path.basename(socket_meter_dst)}")
        else:
            self.api.runtime.issue_warning(state, "[phiGPT] ⚠️ socketMeter.csv not found.")

        return 0
