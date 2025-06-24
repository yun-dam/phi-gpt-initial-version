import os
import datetime
import shutil
import socket
import json
import csv
from pyenergyplus.plugin import EnergyPlusPlugin
from collections import deque
import re

def zone_to_adu_name(zone_name):
    """
    Converts a given zone name to the corresponding ADU VAV HW Rht equipment name.

    Example:
        "THERMAL ZONE: STORY 4 EAST LOWER PERIMETER SPACE" -> "ADU VAV HW Rht 25"
    """
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
    if index is not None:
        if index == 0:
            return "ADU VAV HW Rht"
        else:
            return f"ADU VAV HW Rht {index}"
    else:
        raise ValueError(f"Unknown zone name: {zone_name}")
        
class phiGPTSimulator(EnergyPlusPlugin):
    def __init__(self):
        super().__init__()
        self.need_handles = True
        self.cooling_handle = None
        self.T_out_handle = None
        self.T_in_handle = None
        self.pmv_handle = None
        self.prev_min = -1
        self.zone = "THERMAL ZONE: STORY 6 WEST PERIMETER SPACE"
        self.state_buffer = deque(maxlen=12)

        self.use_fixed_setpoint = False
        self.fixed_setpoint_value = 23.0
        self.use_deadband_mode = False
        self.deadband_center = 23.0
        self.deadband_range = 0.3

        self.cooling_energy_handle = None 
        self.last_buffer_update = (-1, -1)
        self.first_day = None

        self.pending_setpoint = None
        self.last_setpoint = None

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        def _shorten_zone_name(zone_name: str) -> str:
            try:
                parts = zone_name.split("STORY ")[1].split()
                floor = parts[0] + "f"
                short_parts = [w[0].lower() for w in parts[1:] if w.isalpha()]
                return "_".join([floor] + short_parts)
            except Exception:
                return "unknownzone"

        zone_suffix = _shorten_zone_name(self.zone)

        if self.use_fixed_setpoint:
            fixed_str = str(round(self.fixed_setpoint_value, 1)).replace('.', '')
            filename = f"phi_gpt_log_fixed{fixed_str}_{zone_suffix}_{timestamp}.csv"
        else:
            filename = f"phi_gpt_log_{zone_suffix}_{timestamp}.csv"

        self.log_path = os.path.join(self.log_dir, filename)

        with open(self.log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["month", "day", "hour", "minute", "T_out", "T_in", "T_set", "PMV", "Cooling_Energy_J", "reason"])

        self.socket_src = os.path.join(base_dir, "socket.csv")
        self.socket_meter_src = os.path.join(base_dir, "socketMeter.csv")

    def on_begin_zone_timestep_before_set_current_weather(self, state) -> int:
        if not self.api.exchange.api_data_fully_ready(state):
            return 0

        if self.need_handles:
            try:
                adu_name = zone_to_adu_name(self.zone)
                self.cooling_energy_handle = self.api.exchange.get_variable_handle(
                    state, "Zone Air Terminal Sensible Cooling Energy", adu_name)
            except ValueError as e:
                self.api.runtime.issue_severe(state, f"[phiGPT] ❌ {e}")
                return 0
            self.cooling_handle = self.api.exchange.get_actuator_handle(
                state, "Zone Temperature Control", "Cooling Setpoint", self.zone)
            self.T_out_handle = self.api.exchange.get_variable_handle(
                state, "Site Outdoor Air Drybulb Temperature", "Environment")
            self.T_in_handle = self.api.exchange.get_variable_handle(
                state, "Zone Mean Air Temperature", self.zone)
            self.pmv_handle = self.api.exchange.get_variable_handle(
                state, "Zone Thermal Comfort Fanger Model PMV", self.zone)

            if (self.cooling_handle == -1 or self.T_out_handle == -1 or 
                self.T_in_handle == -1 or self.cooling_energy_handle == -1 or self.pmv_handle == -1):
                self.api.runtime.issue_severe(state, "[phiGPT] ❌ Failed to get one or more handles.")
                return 0

            self.api.runtime.issue_warning(state, "[phiGPT] ✅ Handles acquired.")
            self.need_handles = False

        sim_time = self.api.exchange.current_sim_time(state)
        if sim_time == self.prev_min:
            return 0
        self.prev_min = sim_time

        month = self.api.exchange.month(state)
        day = self.api.exchange.day_of_month(state)
        if self.first_day is None:
            self.first_day = (month, day)

        hour = self.api.exchange.hour(state)
        minute = self.api.exchange.minutes(state)

        T_out = self.api.exchange.get_variable_value(state, self.T_out_handle)
        T_in = self.api.exchange.get_variable_value(state, self.T_in_handle)
        PMV = self.api.exchange.get_variable_value(state, self.pmv_handle)

        if (hour, minute) != self.last_buffer_update and minute in (0, 30):
            T_set = self.api.exchange.get_actuator_value(state, self.cooling_handle)
            cooling_energy = self.api.exchange.get_variable_value(state, self.cooling_energy_handle)
            self.state_buffer.append((T_out, T_in, T_set, cooling_energy))
            self.last_buffer_update = (hour, minute)

        if minute not in (0, 30):
            return 0

        apply_setpoint = True

        # Control decision
        if (month, day) == self.first_day and hour < 6:
            current_setpoint = self.fixed_setpoint_value
            reason = "Warm-up phase (before 6AM on first simulation day)"
        elif len(self.state_buffer) < 12:
            current_setpoint = self.fixed_setpoint_value
            reason = "Warm-up phase (insufficient state_buffer)"
        elif self.use_fixed_setpoint:
            if self.use_deadband_mode:
                lower = self.deadband_center - self.deadband_range
                upper = self.deadband_center + self.deadband_range
                if T_in < lower:
                    current_setpoint = 23.5
                    reason = f"T_in={T_in:.2f} < {lower:.2f}, setting to 23.5°C"
                elif T_in > upper:
                    current_setpoint = 22.5
                    reason = f"T_in={T_in:.2f} > {upper:.2f}, setting to 22.5°C"
                else:
                    current_setpoint = self.api.exchange.get_actuator_value(state, self.cooling_handle)
                    reason = f"Within deadband ({lower:.2f}–{upper:.2f}°C), setpoint unchanged"
                    apply_setpoint = False
            else:
                current_setpoint = self.fixed_setpoint_value
                reason = "Fixed setpoint mode"
        else:
            # LLM-based mode with 1-step delay
            response = self.query_reasoning_server(list(self.state_buffer))
            if response:
                if "applied_setpoint" in response:
                    self.pending_setpoint = response["applied_setpoint"]
                    reason = response.get("reason", "Applied setpoint from reasoning server")
                elif "optimal_cooling_setpoints" in response:
                    self.pending_setpoint = response["optimal_cooling_setpoints"][0]
                    reason = response.get("reason", "Fallback: used first of optimal_cooling_setpoints")
                elif "optimal_cooling_setpoint" in response:
                    self.pending_setpoint = response["optimal_cooling_setpoint"]
                    reason = response.get("reason", "Fallback: used single optimal_cooling_setpoint")
                else:
                    self.api.runtime.issue_warning(state, "[phiGPT] ⚠️ No valid setpoint key found in response.")
                    return 0

            else:
                self.api.runtime.issue_warning(state, "[phiGPT] ⚠️ No valid response from reasoning server.")
                return 0

            if self.last_setpoint is not None:
                current_setpoint = self.last_setpoint
            else:
                current_setpoint = self.pending_setpoint

        if apply_setpoint:
            self.api.exchange.set_actuator_value(state, self.cooling_handle, current_setpoint)

        # Always update last_setpoint (LLM only)
        if not self.use_fixed_setpoint:
            self.last_setpoint = self.pending_setpoint

        cooling_energy = self.api.exchange.get_variable_value(state, self.cooling_energy_handle)
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                month, day, hour, minute,
                round(T_out, 2), round(T_in, 2), round(current_setpoint, 2),
                round(PMV, 2), round(cooling_energy, 2), reason
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
                s.sendall(json.dumps({"shutdown": True}).encode('utf-8'))
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
