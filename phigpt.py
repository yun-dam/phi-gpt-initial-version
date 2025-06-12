import os
import re
import json
import time
import ast
import faiss
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv
from feedback_simulator import run_feedback_simulation
import itertools
import random

# LangChain components
from langchain.vectorstores.faiss import FAISS as FAISSBase
from langchain.vectorstores import FAISS  # Only if .load_local is used separately
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory


load_dotenv()

class phiGPTGenerator:
    def __init__(
        self,
        api_key_env: str = "AI_API_KEY",
        api_base_url: str = "https://aiapi-prod.stanford.edu/v1",
        model_name: str = "o3-mini"
    ):
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key in env var '{api_key_env}'")
        self.api_base = api_base_url.rstrip('/')
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.target_temp = 23.0

    def _call_chat_api(self, prompt_text: str) -> str:
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [{"role": "user", "content": prompt_text}]
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def extract_json_from_text(self, text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No JSON object found in text.")

    def generate_response_from_prompt(self, prompt_text: str, ts_knowledge=None, pdf_summary=None):
        while True:
            try:
                reply = self._call_chat_api(prompt_text)
                try:
                    result_json = json.loads(reply)
                except json.JSONDecodeError:
                    result_json = self.extract_json_from_text(reply)

                if (
                    isinstance(result_json, dict)
                    and "optimal_cooling_setpoints" in result_json
                    and isinstance(result_json["optimal_cooling_setpoints"], list)
                    and len(result_json["optimal_cooling_setpoints"]) == 4
                    and all(isinstance(x, (int, float)) for x in result_json["optimal_cooling_setpoints"])
                    and "reason" in result_json
                ):
                    applied_setpoint = result_json["optimal_cooling_setpoints"][0]

                    now = datetime.now()
                    log_path = f"phigpt_json_logs/setpoint_log_{now.strftime('%Y%m%d_%H%M%S')}.jsonl"
                    os.makedirs("phigpt_json_logs", exist_ok=True)

                    log_entry = {
                        "timestamp": now.isoformat(),
                        "prompt": prompt_text,
                        "retrieved_time_series": ts_knowledge or {},
                        "retrieved_text": pdf_summary or "",
                        "optimal_cooling_setpoints": result_json["optimal_cooling_setpoints"],
                        "applied_setpoint": applied_setpoint,
                        "reason": result_json["reason"]
                    }

                    with open(log_path, "w") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    result_json.update({
                        "prompt_text": prompt_text,
                        "retrieved_time_series": ts_knowledge,
                        "retrieved_text": pdf_summary,
                        "applied_setpoint": applied_setpoint,
                        "log_path": log_path
                    })

                    return result_json
                else:
                    raise ValueError("Invalid JSON response format.")

            except Exception as e:
                print(f"⚠️ Error: {e}, retrying in 1s...")
                time.sleep(1)

    def generate_response_with_feedback(
        self, prompt_text: str, ts_knowledge=None, pdf_summary=None,
        log_path: str = "./logs/phi_gpt_log_test.csv",
        zone_name: str = "THERMAL ZONE: STORY 2 SOUTH PERIMETER SPACE"
    ):
        result = self.generate_response_from_prompt(prompt_text, ts_knowledge, pdf_summary)
        setpoints = result["optimal_cooling_setpoints"]
        feedback_df = run_feedback_simulation(setpoints, log_path, zone_name)

        comfort_sum = (feedback_df["T_in"] - self.target_temp).abs().sum() if "T_in" in feedback_df else None
        feedback_metrics = {
            "energy_total_J": feedback_df["Energy_J"].sum() if "Energy_J" in feedback_df else None,
            "comfort_violation_sum": comfort_sum
        }

        return {
            "llm_result": result,
            "feedback_df": feedback_df,
            "metrics": feedback_metrics
        }

    def optimize_setpoints_with_textgrad(
        self,
        prompt_text: str,
        ts_knowledge=None,
        pdf_summary=None,
        log_path="./logs/phi_gpt_log_test.csv",
        zone_name="THERMAL ZONE: STORY 2 SOUTH PERIMETER SPACE",
        max_iters=5
    ):
        def evaluate(setpoints):
            feedback_df = run_feedback_simulation(setpoints, log_path, zone_name)
            energy = feedback_df["Energy_J"].sum() if "Energy_J" in feedback_df else float("inf")
            comfort = (feedback_df["T_in"] - self.target_temp).abs().sum() if "T_in" in feedback_df else float("inf")
            return energy + comfort, feedback_df

        def generate_candidates(base, exploration_n=5):
            allowed = [22.0, 23.0, 24.0]
            neighbors = []

            # depth-1: single-site perturb
            for i in range(4):
                for delta in [-1, 1]:
                    val = base[i] + delta
                    if val in allowed:
                        new = base[:]
                        new[i] = val
                        neighbors.append(new)

            # # depth-2: pair-site perturb
            # for i, j in itertools.combinations(range(4), 2):
            #     for d1 in [-1, 1]:
            #         for d2 in [-1, 1]:
            #             vi, vj = base[i] + d1, base[j] + d2
            #             if vi in allowed and vj in allowed:
            #                 new = base[:]
            #                 new[i] = vi
            #                 new[j] = vj
            #                 neighbors.append(new)

            # random full candidates
            rand_set = random.sample(list(itertools.product(allowed, repeat=4)), exploration_n)
            neighbors.extend([list(r) for r in rand_set])

            # deduplicate
            return [list(t) for t in {tuple(x) for x in neighbors}]

        print("🧠 Initial generation from LLM...")
        result = self.generate_response_from_prompt(prompt_text, ts_knowledge, pdf_summary)
        best_setpoints = result["optimal_cooling_setpoints"]
        best_score, best_feedback = evaluate(best_setpoints)

        print(f"🔍 Initial score: {best_score:.2f} | setpoints: {best_setpoints}")

        for iteration in range(1, max_iters + 1):
            print(f"\n🔁 Iteration {iteration}")
            improved = False
            for candidate in generate_candidates(best_setpoints):
                score, _ = evaluate(candidate)
                print(f"  Candidate {candidate} → Score: {score:.2f}")
                if score < best_score:
                    best_setpoints = candidate
                    best_score = score
                    improved = True
                    print(f"  ✅ New best found!")

            if not improved:
                print("  ❌ No improvement found. Early stopping.")
                break

        feedback_str = best_feedback.to_string(index=False)
        improved_prompt = prompt_text + "\n\n---\n\n## Feedback from simulation:\n" + feedback_str

        return {
            "optimal_cooling_setpoints": best_setpoints,
            "applied_setpoint": best_setpoints[0],
            "reason": result["initial_llm_result"].get("reason", "Optimized by TextGrad"),
            "log_path": result["initial_llm_result"].get("log_path", ""),
            "improved_prompt": improved_prompt
        }

class phiGPTRetriever:
    def __init__(
        self,
        ts_db_path_simulation,
        ts_db_path_measurement,  # 그대로 받아오되, 사용은 안 함
        pdf_db_path,
        api_key_env: str = "AI_API_KEY",
        api_base_url: str = "https://aiapi-prod.stanford.edu/v1",
        model_name: str = "o3-mini",
        horizon_hours: int = 3
    ):
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key in env var '{api_key_env}'")
        self.api_base = api_base_url.rstrip('/')
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Simulation vectorstore만 사용
        self.vectorstore_simulation = FAISSBase.load_local(
            ts_db_path_simulation, self.embeddings, allow_dangerous_deserialization=True
        )
        self.target_temp = 23.0


        # 🔒 현재는 measurement vectorstore 사용 안 함
        # self.vectorstore_measurement = FAISSBase.load_local(
        #     ts_db_path_measurement, self.embeddings, allow_dangerous_deserialization=True
        # )

        pdf_index_path = os.path.join(pdf_db_path, "merged_faiss_may26.index")
        pdf_meta_path = os.path.join(pdf_db_path, "merged_metadata_may26.json")
        self.pdf_vectorstore = (
            self.load_faiss_from_json_and_index(pdf_index_path, pdf_meta_path, self.embeddings)
            if os.path.exists(pdf_index_path) and os.path.exists(pdf_meta_path)
            else None
        )
        self.pdf_retriever = self.pdf_vectorstore.as_retriever(search_kwargs={"k": 3}) if self.pdf_vectorstore else None

        self.retriever = self.vectorstore_simulation.as_retriever()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.generator = phiGPTGenerator(api_key_env=api_key_env, api_base_url=api_base_url, model_name=model_name)
        self.horizon_hours = horizon_hours

    def load_faiss_from_json_and_index(self, index_path, metadata_path, embedding_function):
        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        documents = [
            Document(page_content=entry["text"], metadata={k: v for k, v in entry.items() if k != "text"})
            for entry in metadata_list
        ]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}
        return FAISSBase(embedding_function.embed_query, index, docstore, index_to_docstore_id)

    def generate_query_from_state(self, current_states):
        arr = np.array(current_states)
        num_timesteps = arr.shape[0]

        avg_state = np.round(np.mean(arr, axis=0), 2)
        avg_out, avg_inn, avg_setp = avg_state.tolist()
        avg_delta = round(avg_inn - avg_setp, 2)

        curr = arr[-1]
        curr_out, curr_inn, curr_setp = np.round(curr, 2).tolist()
        curr_delta = round(curr_inn - curr_setp, 2)

        target_indoor = 24.0
        to_target = round(curr_inn - target_indoor, 2)

        if num_timesteps >= 2:
            prev = arr[-2]
            indoor_trend = "increasing" if curr[1] > prev[1] else "decreasing" if curr[1] < prev[1] else "stable"
            indoor_change = round(curr[1] - prev[1], 2)

            outdoor_trend = "increasing" if curr[0] > prev[0] else "decreasing" if curr[0] < prev[0] else "stable"
            outdoor_change = round(curr[0] - prev[0], 2)

            setpoint_trend = "increasing" if curr[2] > prev[2] else "decreasing" if curr[2] < prev[2] else "stable"
        else:
            indoor_trend = outdoor_trend = setpoint_trend = "unknown"
            indoor_change = outdoor_change = 0.0

        query = (
            f"Searching for cooling strategies under the following conditions:\n\n"
            f"→ Current system state (most recent timestep):\n"
            f"- Outdoor temperature: {curr_out}°C ({outdoor_trend}, Δ={outdoor_change})\n"
            f"- Indoor temperature: {curr_inn}°C ({indoor_trend}, Δ={indoor_change})\n"
            f"- Cooling setpoint: {curr_setp}°C ({setpoint_trend})\n"
            f"- Indoor - setpoint difference: {curr_delta}°C\n"
            f"- Target indoor temperature: {target_indoor}°C (Δ={to_target}°C from current)\n\n"
            f"→ Average over last {num_timesteps} timesteps:\n"
            f"- Avg outdoor temp: {avg_out}°C\n"
            f"- Avg indoor temp: {avg_inn}°C\n"
            f"- Avg setpoint: {avg_setp}°C\n"
            f"- Avg indoor-setpoint difference: {avg_delta}°C\n\n"
            f"Looking for cooling control strategies that reduce energy use while maintaining thermal comfort."
        )
        return query

    def get_relevant_pdf_text(self, current_states):
        if not self.pdf_retriever:
            return ""
        query = self.generate_query_from_state(current_states)
        docs = self.pdf_retriever.get_relevant_documents(query)
        return "\n\n".join(d.page_content for d in docs)

    def retrieve_time_series_knowledge(self, current_states, top_k=5):
        current_vector = np.round(np.mean(np.array(current_states), axis=0), 4).tolist()
        sim_docs = self.vectorstore_simulation.similarity_search(str(current_vector), k=top_k)
        parsed_sim = [ast.literal_eval(lst) for lst in "\n\n".join(d.page_content for d in sim_docs).strip().split('\n\n')]
        sim_text = self.format_state_series_table(parsed_sim, label="Simulation", is_nested=True)

        # 🔒 측정 기반 검색은 현재 비활성화
        # meas_docs = self.vectorstore_measurement.similarity_search(str(current_vector), k=top_k)
        # parsed_meas = [ast.literal_eval(lst) for lst in "\n\n".join(d.page_content for d in meas_docs).strip().split('\n\n')]
        # meas_text = self.format_state_series_table(parsed_meas, label="Measurement", is_nested=True)

        return {"simulation": sim_text, "measurement": ""}  # 빈 문자열로 처리

    def format_state_series_table(self, series_list, label="State", is_nested=False):
        def make_table_block(series, block_label):
            header = (
                f"{block_label}\n"
                "Hour | Outdoor Temp (°C) | Indoor Temp (°C) | Cooling Setpoint (°C) | Indoor - Setpoint | Δ Indoor | Δ Outdoor\n"
                "-----|------------------|------------------|------------------------|-------------------|-----------|------------"
            )
            rows = []
            prev_indoor = None
            prev_outdoor = None
            for hour, (out_c, inn_c, setp_c) in enumerate(series):
                delta1 = round(inn_c - setp_c, 2)
                delta2 = round(inn_c - prev_indoor, 2) if prev_indoor is not None else "N/A"
                delta3 = round(out_c - prev_outdoor, 2) if prev_outdoor is not None else "N/A"
                row = f"{hour:>4} | {round(out_c, 2):>17} | {round(inn_c, 2):>17} | {round(setp_c, 2):>24} | {delta1:>17} | {str(delta2):>9} | {str(delta3):>10}"
                rows.append(row)
                prev_indoor = inn_c
                prev_outdoor = out_c
            return header + "\n" + "\n".join(rows)

        if is_nested:
            return "\n\n".join([make_table_block(series, f"{label} {i+1}") for i, series in enumerate(series_list)])
        else:
            return make_table_block(series_list, label)

    def build_cooling_prompt(self, current_states):
        ts_knowledge = self.retrieve_time_series_knowledge(current_states, top_k=5)
        current_states_table = self.format_state_series_table(current_states, label="Current State", is_nested=False)
        retrieved_text = self.get_relevant_pdf_text(current_states)


        # 🔒 측정 기반 패턴은 현재 출력 안 함
        # "### Measurement-based Patterns:\n"
        # f"{ts_knowledge['measurement']}\n\n"

        prompt = (
            f"# COOLING Setpoint Optimizer\n\n"
            f"You are an intelligent agent tasked with optimizing the COOLING setpoints for a building based on time-series HVAC data.\n\n"
            "---\n\n"
            "## Objective\n"
            f"Determine the optimal cooling setpoints for the next 4 time steps (t₀ to t₃), each representing a 30-minute interval (totaling 2 hours), that:\n"
            "- Minimize total cooling energy consumption\n"
            f"- Maintain indoor temperature close to {self.target_temp:.1f}°C\n"
            "- Adapt to current building and environmental conditions using historical system behavior and expert strategies\n"
            "- Take into account the **thermal lag** effect — indoor temperature responds gradually to changes in setpoints.\n\n"
            "---\n\n"
            "## Current System States (Last Few Hours)\n"
            f"{current_states_table}\n\n"
            "---\n\n"
            "## Retrieved Historical Patterns\n"
            "### Simulation-based Patterns:\n"
            f"{ts_knowledge['simulation']}\n\n"
            "---\n\n"
            "## Reference Knowledge from Literature\n"
            "Below is a relevant excerpt from technical documents describing expert-designed cooling setpoint strategies:\n\n"
            f"{retrieved_text.strip() if retrieved_text else 'No relevant documents retrieved.'}\n\n"
            "---\n\n"
            "## Response Instructions\n"
            "Please choose **4 cooling setpoints**, one for each of the next 4 time steps (t₀, t₁, t₂, t₃).\n"
            "- Each time step corresponds to 30 minutes (2-hour control horizon).\n"
            "- Choose each setpoint from the options: **[22°C, 23°C, 24°C]**.\n"
            "- Because of thermal lag, your choices should anticipate future temperature behavior rather than reacting only to the present.\n\n"
            "Output your result in **valid JSON format** exactly as shown below.\n\n"
            "---\n\n"
            "## Output Format\n"
            "{\n"
            "  \"optimal_cooling_setpoints\": [23, 22, 22, 23],\n"
            "  \"reason\": \"Started with strong cooling due to rising indoor temps, then relaxed as trend stabilizes\"\n"
            "}"
        )

        return prompt, ts_knowledge, retrieved_text

    def generate_optimized_setpoint(self, current_states):
        prompt, ts_know, pdf_retrieved = self.build_cooling_prompt(current_states)
        result = self.generator.generate_response_from_prompt(prompt, ts_know, pdf_retrieved)
        return result["applied_setpoint"]  # ✅ t₀만 실제 제어에 사용

