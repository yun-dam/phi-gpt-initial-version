import socket
import json
import traceback
from phigpt import phiGPTRetriever
from phigpt import phiGPTGenerator

# 1) Initialize the retriever
retriever = phiGPTRetriever(
    ts_db_path_simulation=r".\data\timeseries\multiindex_energyplus_data",
    ts_db_path_measurement=r".\data\timeseries\multiindex_energyplus_data",
    pdf_db_path=r".\data\text",
    api_key_env="AI_API_KEY",
    api_base_url="https://aiapi-prod.stanford.edu/v1",
    model_name="o3-mini",
    horizon_hours=3
)

# 2) Initialize the generator
generator = phiGPTGenerator(
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
    if state_buffer is None:
        conn.sendall(json.dumps({"error": "No state_buffer provided."}).encode())
        return

    try:
        # 🧠 Build prompt
        prompt, ts_know, pdf_sum = retriever.build_cooling_prompt(state_buffer)

        if use_textgrad:
            print("[ReasoningServer] 🚀 Using TextGrad optimization...")
            result_raw = generator.optimize_setpoints_with_textgrad(
                prompt_text=prompt,
                ts_knowledge=ts_know,
                pdf_summary=pdf_sum,
                log_path="./logs/phi_gpt_log_test.csv",
                zone_name="THERMAL ZONE: STORY 2 SOUTH PERIMETER SPACE",
                max_iters=1
            )
            # ← Modified mapping to use the keys returned by optimize_setpoints_with_textgrad
            result = {
                "optimal_cooling_setpoints": result_raw["optimal_cooling_setpoints"],
                "applied_setpoint":           result_raw["applied_setpoint"],
                "reason":                     result_raw.get("reason", "Optimized by TextGrad"),
                "log_path":                   result_raw.get("log_path", ""),
                "improved_prompt":            result_raw.get("improved_prompt", "")
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
            "applied_setpoint":           result["applied_setpoint"],
            "reason":                     result["reason"]
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
