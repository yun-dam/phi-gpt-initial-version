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

    def _call_chat_api(self, prompt_text: str) -> str:
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [ {"role": "user", "content": prompt_text} ]
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate_response_from_prompt(self, prompt_text: str, ts_knowledge=None, pdf_summary=None):
        print(f"📥 Sending prompt to model '{self.model_name}' ({len(prompt_text)} characters)")
        while True:
            try:
                reply = self._call_chat_api(prompt_text)
                print(f"📤 Received reply ({len(reply)} characters)")

                try:
                    result_json = json.loads(reply)
                except json.JSONDecodeError:
                    print("⚠️ Raw response contains extra text. Attempting to extract JSON...")
                    result_json = self.extract_json_from_text(reply)
                    print(f"✅ Extracted JSON: {result_json}")

                if (
                    isinstance(result_json, dict) and
                    "optimal_cooling_setpoint" in result_json and
                    isinstance(result_json["optimal_cooling_setpoint"], (int, float)) and
                    "reason" in result_json and
                    isinstance(result_json["reason"], str)
                ):
                    log_dir = "phigpt_json_logs"
                    os.makedirs(log_dir, exist_ok=True)

                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "prompt": prompt_text,
                        "retrieved_time_series": ts_knowledge,
                        "retrieved_text": pdf_summary,
                        "optimal_cooling_setpoint": result_json["optimal_cooling_setpoint"],
                        "reason": result_json["reason"]
                    }

                    log_path = os.path.join(log_dir, "setpoint_log.jsonl")
                    with open(log_path, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    result_json.update({
                        "prompt_text": prompt_text,
                        "retrieved_time_series": ts_knowledge,
                        "retrieved_text": pdf_summary
                    })

                    return result_json
                else:
                    raise ValueError("Invalid JSON structure")

            except Exception as e:
                print(f"⚠️ Error generating response: {e}. Retrying in 1s...")
                time.sleep(1)

    def extract_json_from_text(self, text: str) -> dict:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError("No JSON object found in text.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")


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
            f"You are an intelligent agent tasked with optimizing the COOLING setpoint for a building based on time-series HVAC data.\n\n"
            "---\n\n"
            "## Objective\n"
            "Determine the optimal cooling setpoint that:\n"
            "- Minimizes energy consumption\n"
            "- Maintains indoor temperature near 25°C\n"
            "- Adapts to current building and environmental conditions using historical system behavior and expert strategies\n\n"
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
            "You must now choose the optimal cooling setpoint (24°C, 25°C, or 26°C) and provide a short justification.\n"
            "Use the historical patterns and the expert strategies above to guide your decision.\n"
            "Output your result in valid JSON format **only**.\n\n"
            "---\n\n"
            "## Output Format\n"
            "{\n"
            "  \"optimal_cooling_setpoint\": 25,\n"
            "  \"reason\": \"Brief explanation of your selection rationale\"\n"
            "}"
        )

        return prompt, ts_knowledge, retrieved_text

    def generate_optimized_setpoint(self, current_states):
        prompt, ts_know, pdf_retrieved = self.build_cooling_prompt(current_states)
        return self.generator.generate_response_from_prompt(prompt, ts_know, pdf_retrieved)
