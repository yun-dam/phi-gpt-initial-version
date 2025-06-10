from datetime import datetime
import os
import json
import re
import time
import ast
import numpy as np
import requests
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

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
        ts_db_path_measurement,
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
        self.vectorstore_simulation = FAISS.load_local(
            ts_db_path_simulation, self.embeddings, allow_dangerous_deserialization=True
        )
        self.vectorstore_measurement = FAISS.load_local(
            ts_db_path_measurement, self.embeddings, allow_dangerous_deserialization=True
        )

        self.pdf_vectorstore = (
            FAISS.load_local(pdf_db_path, self.embeddings, allow_dangerous_deserialization=True)
            if os.path.exists(os.path.join(pdf_db_path, "index.faiss"))
            else None
        )
        self.pdf_retriever = self.pdf_vectorstore.as_retriever(search_kwargs={"k": 3}) if self.pdf_vectorstore else None

        self.retriever = self.vectorstore_simulation.as_retriever()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.generator = phiGPTGenerator(api_key_env=api_key_env, api_base_url=api_base_url, model_name=model_name)
        self.horizon_hours = horizon_hours

    def _call_chat_api(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers=self.headers,
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def get_cooling_strategy_summary(self) -> str:
        if not self.pdf_retriever:
            return ""
        docs = self.pdf_retriever.get_relevant_documents("cooling set point strategies")
        combined = "\n\n".join(d.page_content for d in docs)
        system = "You are an assistant summarizing HVAC cooling strategies."
        return self._call_chat_api(system, combined)

    def retrieve_time_series_knowledge(self, current_states, top_k=5):
        current_vector = np.round(np.mean(np.array(current_states), axis=0), 4).tolist()
        sim_docs = self.vectorstore_simulation.similarity_search(str(current_vector), k=top_k)
        meas_docs = self.vectorstore_measurement.similarity_search(str(current_vector), k=top_k)

        sim_text = self._format_time_series_text(
            "\n\n".join(d.page_content for d in sim_docs), label="Simulation"
        )
        meas_text = self._format_time_series_text(
            "\n\n".join(d.page_content for d in meas_docs), label="Measurement"
        )
        return {"simulation": sim_text, "measurement": meas_text}

    def _format_time_series_text(self, raw_text, label="Simulation"):
        list_strings = raw_text.strip().split('\n\n')
        time_series_list = [
            [[round(value, 2) for value in triplet] for triplet in ast.literal_eval(lst)]
            for lst in list_strings
        ]

        formatted_blocks = []
        for idx, series in enumerate(time_series_list):
            header = (
                f"{label} {idx + 1}\n"
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

            formatted_blocks.append(header + "\n" + "\n".join(rows))

        return "\n\n".join(formatted_blocks)

    def format_current_state_table(self, current_states):
        header = (
            f"Current State\n"
            "Hour | Outdoor Temp (°C) | Indoor Temp (°C) | Cooling Setpoint (°C) | Indoor - Setpoint | Δ Indoor | Δ Outdoor\n"
            "-----|------------------|------------------|------------------------|-------------------|-----------|------------"
        )
        rows = []
        prev_indoor = None
        prev_outdoor = None

        for hour, (out_c, inn_c, setp_c) in enumerate(current_states):
            delta1 = round(inn_c - setp_c, 2)
            delta2 = round(inn_c - prev_indoor, 2) if prev_indoor is not None else "N/A"
            delta3 = round(out_c - prev_outdoor, 2) if prev_outdoor is not None else "N/A"
            row = f"{hour:>4} | {round(out_c, 2):>17} | {round(inn_c, 2):>17} | {round(setp_c, 2):>24} | {delta1:>17} | {str(delta2):>9} | {str(delta3):>10}"
            rows.append(row)
            prev_indoor = inn_c
            prev_outdoor = out_c

        return header + "\n" + "\n".join(rows)

    def build_cooling_prompt(self, current_states):
        ts_knowledge = self.retrieve_time_series_knowledge(current_states, top_k=5)
        pdf_summary = self.get_cooling_strategy_summary()
        current_states_table = self.format_current_state_table(current_states)
        recent_1hr_state = [current_states[-1]]
        recent_1hr_table = self.format_current_state_table(recent_1hr_state)

        prompt = (
            f"# COOLING Setpoint Optimizer\n\n"
            f"You are an intelligent agent tasked with optimizing the COOLING setpoint for a building based on time-series HVAC data.\n\n"
            "---\n\n"
            "## Objective\n"
            "Determine the optimal cooling setpoint that:\n"
            "- Minimizes energy consumption\n"
            "- Maintains indoor temperature at 25°C\n"
            "- Adapts to current building conditions using past system behavior\n\n"
            "---\n\n"
            "## Response Requirements\n"
            "1. Respond only with valid JSON  \n"
            "2. Do not include any explanation outside the JSON  \n"
            "3. Choose from these values only: 24, 25, or 26 (°C)  \n"
            "4. Optimize the COOLING setpoint (not heating)\n\n"
            "---\n\n"
            "## Response Format\n"
            "{\n"
            "  \"optimal_cooling_setpoint\": 25,\n"
            "  \"reason\": \"Brief explanation of your selection rationale\"\n"
            "}"
        )

        return prompt, ts_knowledge, pdf_summary

    def generate_optimized_setpoint(self, current_states):
        prompt, ts_know, pdf_sum = self.build_cooling_prompt(current_states)
        return self.generator.generate_response_from_prompt(prompt, ts_know, pdf_sum)
