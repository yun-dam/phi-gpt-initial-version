import socket
import json
import traceback
from phigpt import phiGPTRetriever
from phigpt import phiGPTGenerator
import os
from datetime import datetime

zone_name = "THERMAL ZONE: STORY 4 EAST LOWER PERIMETER SPACE"

if not zone_name:
    raise ValueError("Zone name must be defined.")

# 1) Initialize the retriever
retriever = phiGPTRetriever(
    ts_db_path_simulation=r".\data\timeseries\ep_simulation",
    ts_db_path_measurement=r".\data\timeseries\multiindex_energyplus_data",
    pdf_db_path=r".\data\text",
    api_key_env="AI_API_KEY",
    api_base_url="https://aiapi-prod.stanford.edu/v1",
    model_name="o3-mini",
    horizon_hours=3,
    target_zone=zone_name
)

# 2) Initialize the generator
generator = phiGPTGenerator(
    retriever=retriever,
    api_key_env="AI_API_KEY",
    api_base_url="https://aiapi-prod.stanford.edu/v1",
    model_name="o3-mini"
)

# Use TextGrad?
use_textgrad = True

HOST = '127.0.0.1'
PORT = 55555
TIMEOUT_SECONDS = 120  # 2 minutes

def handle_request(conn):
    data = conn.recv(8192)
    if not data:
        return
    message = json.loads(data.decode())

    # Shutdown signal
    if message.get("shutdown", False):
        conn.sendall(json.dumps({"status": "server shutting down"}).encode())
        return

    # Extract state buffer
    state_buffer = message.get("state_buffer")
    current_time = message.get("current_time", None) 
    if state_buffer is None:
        conn.sendall(json.dumps({"error": "No state_buffer provided."}).encode())
        return

    try:
        # 🧠 Build prompt
        prompt, ts_know, pdf_sum = retriever.build_cooling_prompt(state_buffer, current_time=current_time)

        if use_textgrad:
            print("[ReasoningServer] 🚀 Using TextGrad optimization...")
            result_raw = generator.optimize_setpoints_with_textgrad(
                prompt_text=prompt,
                ts_knowledge=ts_know,
                pdf_summary=pdf_sum,
                log_path=None,
                zone_name=zone_name,
                max_iters=5,
                current_states=state_buffer
            )

            raw_reason = result_raw.get("reason", "")
            print(f"[DEBUG] Raw reason from result_raw: {repr(raw_reason)}")
            print(result_raw)

            # 🪵 Save JSONL log
            os.makedirs("./logs/reasoning_jsonl", exist_ok=True)
            jsonl_path = f"./logs/reasoning_jsonl/result_{datetime.now().strftime('%Y%m%d')}.jsonl"

            try:
                parsed_reason = json.loads(raw_reason) if isinstance(raw_reason, str) else raw_reason
                reason_text = parsed_reason.get("reason", parsed_reason)
            except Exception:
                reason_text = raw_reason

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "state_buffer": state_buffer,
                "zone_name": zone_name,
                "optimal_cooling_setpoints": result_raw.get("optimal_cooling_setpoints"),
                "applied_setpoint": result_raw.get("applied_setpoint"),
                "reason": reason_text,
                "log_path": result_raw.get("log_path"),
                "improved_prompt": result_raw.get("improved_prompt"),
                "initial_score": float(result_raw.get("initial_score", -1)),
                "final_score": float(result_raw.get("final_score", -1)),
                "improvement": float(result_raw.get("improvement", 0.0))
            }

            with open(jsonl_path, "a", encoding="utf-8") as f:
                json.dump(log_entry, f)
                f.write("\n")

            result = {
                "optimal_cooling_setpoints": result_raw["optimal_cooling_setpoints"],
                "applied_setpoint": result_raw["applied_setpoint"],
                "reason": reason_text,
                "log_path": result_raw.get("log_path", ""),
                "improved_prompt": result_raw.get("improved_prompt", "")
            }
        else:
            print("[ReasoningServer] ✨ Using single-shot LLM generation...")
            result = generator.generate_response_from_prompt(prompt, ts_know, pdf_sum)

        # 📢 Debug output
        print("\n[ReasoningServer] 🔄 Cooling Control Decision (Multi-step)")
        print(f"> Setpoints (t0~t3): {result['optimal_cooling_setpoints']}")
        print(f"> Reason: {result['reason']}\n")

        # ✅ Send response (to simulator_socket)
        response = {
            "optimal_cooling_setpoints": result["optimal_cooling_setpoints"],
            "applied_setpoint": result["applied_setpoint"],
            "reason": result["reason"]
        }

    except Exception as e:
        print("[ReasoningServer] ⚠️ Error during generation:", e)
        traceback.print_exc()
        response = {"error": str(e)}

    conn.sendall(json.dumps(response).encode())


# Main server loop
with socket.socket() as s:
    s.bind((HOST, PORT))
    s.listen()
    s.settimeout(TIMEOUT_SECONDS)

    print(f"[ReasoningServer] Listening on {HOST}:{PORT} (timeout: {TIMEOUT_SECONDS}s)")

    try:
        while True:
            try:
                conn, _ = s.accept()
                with conn:
                    handle_request(conn)
            except socket.timeout:
                print("[ReasoningServer] ⏳ Timeout: No incoming connection. Shutting down.")
                break
    except KeyboardInterrupt:
        print("[ReasoningServer] 🧹 Manually interrupted. Shutting down.")

    print("[ReasoningServer] 🛑 Server shutdown complete.")
