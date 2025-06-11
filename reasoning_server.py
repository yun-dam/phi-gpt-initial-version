import socket
import json
from phigpt import phiGPTRetriever
from phigpt import phiGPTGenerator

# 1) Initialize the retriever with API and database paths
retriever = phiGPTRetriever(
    ts_db_path_simulation=r".\data\timeseries\multiindex_energyplus_data",
    ts_db_path_measurement=r".\data\timeseries\multiindex_energyplus_data",
    pdf_db_path=r".\data\text",
    api_key_env="AI_API_KEY",
    api_base_url="https://aiapi-prod.stanford.edu/v1",
    model_name="o3-mini",
    horizon_hours=3
)

# 2) Initialize the generator with the same API settings
generator = phiGPTGenerator(
    api_key_env="AI_API_KEY",
    api_base_url="https://aiapi-prod.stanford.edu/v1",
    model_name="o3-mini"
)

HOST = '127.0.0.1'
PORT = 55555
TIMEOUT_SECONDS = 120  # Auto-shutdown if no connection for 60 seconds

def handle_request(conn):
    data = conn.recv(8192)
    if not data:
        return
    message = json.loads(data.decode())

    if message.get("shutdown", False):
        conn.sendall(json.dumps({"status": "server shutting down"}).encode())
        return

    state_buffer = message.get("state_buffer")
    if state_buffer is None:
        conn.sendall(json.dumps({"error": "No state_buffer provided."}).encode())
        return

    try:
        # Build prompt and call the LLM to get optimal setpoint
        prompt, ts_know, pdf_sum = retriever.build_cooling_prompt(state_buffer)
        result = generator.generate_response_from_prompt(prompt, ts_know, pdf_sum)

        # Print current control decision to console
        print("\n[ReasoningServer] 🔄 Cooling Control Decision at Current Timestep")
        print(f"> Optimal Setpoint: {result['optimal_cooling_setpoint']}°C")
        print(f"> Reason: {result['reason']}\n")

        response = {
            "optimal_cooling_setpoint": result["optimal_cooling_setpoint"],
            "reason": result["reason"]
        }

    except Exception as e:
        print("[ReasoningServer] ⚠️ Error during generation:", e)
        response = {"error": str(e)}

    conn.sendall(json.dumps(response).encode())

# Main server loop with timeout
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
