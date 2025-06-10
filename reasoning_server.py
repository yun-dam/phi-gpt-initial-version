import socket
import json
from phigpt import phiGPTRetriever
from phigpt import phiGPTGenerator 

# 1) Retriever 인스턴스에 API 정보 넘기기
retriever = phiGPTRetriever(
    ts_db_path_simulation=r".\data\timeseries\multiindex_energyplus_data",
    ts_db_path_measurement=r".\data\timeseries\multiindex_energyplus_data",
    pdf_db_path=r".\data\text",
    api_key_env="AI_API_KEY",
    api_base_url="https://aiapi-prod.stanford.edu/v1",
    model_name="o3-mini",
    horizon_hours=3
)

# 2) Generator 인스턴스에도 동일하게 API 정보 넘기기
generator = phiGPTGenerator(
    api_key_env="AI_API_KEY",
    api_base_url="https://aiapi-prod.stanford.edu/v1",
    model_name="o3-mini"
)

HOST = '127.0.0.1'
PORT = 55555
server_running = True

def handle_request(conn):
    data = conn.recv(8192)
    if not data:
        return
    message = json.loads(data.decode())

    if message.get("shutdown", False):
        global server_running
        server_running = False
        conn.sendall(json.dumps({"status": "server shutting down"}).encode())
        return

    state_buffer = message.get("state_buffer")
    if state_buffer is None:
        conn.sendall(json.dumps({"error": "No state_buffer provided."}).encode())
        return

    try:
        # Prompt 생성 → API 호출 → JSON 파싱
        prompt, ts_know, pdf_sum = retriever.build_cooling_prompt(state_buffer)
        result = generator.generate_response_from_prompt(prompt, ts_know, pdf_sum)
        response = {
            "optimal_cooling_setpoint": result["optimal_cooling_setpoint"],
            "reason": result["reason"]
        }
    except Exception as e:
        print("[ReasoningServer] ⚠️ Error during generation:", e)
        response = {"error": str(e)}

    conn.sendall(json.dumps(response).encode())

with socket.socket() as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"[ReasoningServer] Listening on {HOST}:{PORT}")
    while server_running:
        conn, _ = s.accept()
        with conn:
            handle_request(conn)
    print("[ReasoningServer] 🛑 Server shutdown complete.")
