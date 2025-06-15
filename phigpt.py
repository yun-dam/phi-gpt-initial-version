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
import itertools
import random
import csv

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
        retriever=None, 
        api_key_env: str = "AI_API_KEY",
        api_base_url: str = "https://aiapi-prod.stanford.edu/v1",
        model_name: str = "o3-mini"
    ):
        self.retriever = retriever
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
        log_path=None,
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
            log_path=None,
            zone_name: str = "THERMAL ZONE: STORY 2 SOUTH PERIMETER SPACE",
            max_iters: int = 5,
            debug_mode: str = "full",
            current_states=None
        ):

        import textgrad as tg
        import pandas as pd
        import os
        import json
        from datetime import datetime
        from feedback_simulator import run_feedback_simulation as run_sim
        from textgrad.engine.openai import StanfordChatAI

        if debug_mode == "quick":
            result = self.generate_response_from_prompt(prompt_text, ts_knowledge, pdf_summary)
            initial_setpoints = result.get("optimal_cooling_setpoints", [23.0] * 4)
            return {
                "optimal_cooling_setpoints": initial_setpoints,
                "applied_setpoint": initial_setpoints[0],
                "reason": result.get("reason", "No reason provided"),
                "log_path": result.get("log_path", None),
                "improved_prompt": prompt_text
            }

        elif debug_mode == "single_iter":
            max_iters = 1

        result = self.generate_response_from_prompt(prompt_text, ts_knowledge, pdf_summary)
        initial_setpoints = result.get("optimal_cooling_setpoints", [23.0] * 4)
        if not isinstance(initial_setpoints, list) or len(initial_setpoints) < 4:
            print("⚠️ Invalid initial setpoints. Falling back to default [23.0]*4")
            initial_setpoints = [23.0] * 4

        engine = StanfordChatAI(
            model_string="o3-mini",
            system_prompt="You are a helpful HVAC optimization assistant.",
            api_key_env="AI_API_KEY",
            api_base_url="https://aiapi-prod.stanford.edu/v1"
        )

        initial_value = ", ".join(str(v) for v in initial_setpoints)
        setpoints_var = tg.Variable(
            initial_value,
            requires_grad=True,
            role_description="cooling setpoints"
        )

        def simulation_loss(setpoints, step_idx=None):
            try:
                val = setpoints_var.value
                vals = []
                for v in val.split(","):
                    v_clean = v.strip().replace("°C", "").strip()
                    try:
                        import re
                        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", v_clean)
                        if match:
                            vals.append(float(match.group()))
                        else:
                            raise ValueError
                    except ValueError:
                        print(f"⚠️ Could not parse value '{v}'. Defaulting to {self.target_temp}")
                        vals.append(self.target_temp)

            except:
                vals = [self.target_temp] * 4

            allowed = [22.0, 23.0, 24.0]
            vals = [min(allowed, key=lambda x: abs(x - v)) for v in vals]
            vals = (vals + [self.target_temp] * 4)[:4]

            if step_idx is not None:
                print(f"[TextGrad] 🧪 Running EnergyPlus simulation at iteration {step_idx}")

            try:
                df = run_sim(vals, log_path, zone_name)
            except Exception as e:
                print(f"⚠️ Simulation failed: {e}")
                df = pd.DataFrame({"Energy_J": [float("inf")], "T_in": [self.target_temp]})

            energy = df["Energy_J"].sum() if "Energy_J" in df else float("inf")
            comfort = (df["T_in"] - self.target_temp).abs().sum() if "T_in" in df else float("inf")

            energy_weight = 1.0
            comfort_weight = 2.0
            score = (energy_weight * energy / 1_000_000) + (comfort_weight * comfort)

            state_summary = ""
            if current_states is not None:
                state_summary = self.retriever.generate_query_from_state(current_states)

            feedback_str = (
                f"Current system state:\n{state_summary}\n\n"
                f"Simulation results:\n{df.to_string(index=False)}\n\n"
                f"Energy: {energy:.2f} J, Comfort: {comfort:.2f}, Score: {score:.2f}"
            )

            return tg.Variable(
                feedback_str,
                role_description="simulation feedback"
            ), score, df

        optimizer = tg.optimizer.TextualGradientDescent(parameters=[setpoints_var], engine=engine)

        best_setpoints = initial_setpoints
        feedback_var, best_score, best_df = simulation_loss(setpoints_var)
        initial_score = best_score

        os.makedirs("./logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"./logs/llm_response_log_{timestamp}.csv"
        is_new_file = not os.path.exists(log_file_path)

        for i in range(max_iters):
            state_context = ""
            if current_states is not None:
                state_context = self.retriever.generate_query_from_state(current_states)

            instruction_var = tg.Variable(
            f"You are an HVAC optimization expert.\n"
            f"You are controlling a **single thermal zone**, and your task is to suggest a sequence of 4 cooling setpoints (°C),\n"
            f"each corresponding to a 30-minute interval, for the upcoming 2-hour control horizon.\n\n"
            f"These 4 setpoints represent the control sequence for **one zone only**, not multiple zones.\n"
            f"You must consider the effects of **thermal lag**, meaning the indoor temperature changes gradually in response to setpoint adjustments.\n"
            f"Your objective is to minimize both future energy use and thermal discomfort over time.\n\n"
            f"Return your answer as a single **comma-separated list** of 4 numeric values using only these options: 22.0, 23.0, or 24.0.\n"
            f"For example:\n"
            f"22.0, 23.0, 23.0, 24.0\n\n"
            f"⚠️ Formatting Rules:\n"
            f"- Do NOT include brackets, quotes, units (°C), or JSON/XML\n"
            f"- Do NOT mention zone numbers or names\n"
            f"- Do NOT include any explanation or surrounding text\n"
            f"- Just output 4 numbers separated by commas",
            requires_grad=False,
            role_description="instruction for generating updated setpoints"
            )



            loss_fn = tg.TextLoss(instruction_var, engine=engine)
            feedback_var, score, _ = simulation_loss(setpoints_var, step_idx=i + 1)
            loss = loss_fn(feedback_var)
            loss.backward(engine=engine)
            optimizer.step()

            log_row = {
            "sim_time": current_states[-1]["time"] if current_states and "time" in current_states[-1] else "unknown",
            "iteration": i + 1,
            "updated_setpoints": setpoints_var.value,
            "score": score,
            "feedback": feedback_var.value.replace("\n", " ")[:500]  # 줄바꿈 제거 및 길이 제한
            }

            with open(log_file_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_row.keys())
                if is_new_file:
                    writer.writeheader()
                    is_new_file = False
                writer.writerow(log_row)

            val = setpoints_var.value
            vals = []
            for v in val.split(","):
                v_clean = v.strip().replace("°C", "").strip()
                try:
                    import re
                    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", v_clean)
                    if match:
                        vals.append(float(match.group()))
                    else:
                        raise ValueError
                except ValueError:
                    print(f"⚠️ Could not parse value '{v}'. Defaulting to {self.target_temp}")
                    vals.append(self.target_temp)


            vals = [min([22.0, 23.0, 24.0], key=lambda x: abs(x - v)) for v in vals]
            vals = (vals + [self.target_temp] * 4)[:4]

            _, new_score, new_df = simulation_loss(
                tg.Variable(
                    ", ".join(map(str, vals)),
                    requires_grad=False,
                    role_description="updated setpoint after iteration"
                ),
                step_idx=f"{i + 1} (post-update)"
            )

            if new_score < best_score:
                best_setpoints = vals
                best_score = new_score
                best_df = new_df
                setpoints_var.set_value(", ".join(map(str, best_setpoints)))

        prompt_with_feedback = prompt_text + "\n\n## Simulation Feedback:\n" + best_df.to_string(index=False)
        reason_prompt = (
            f"You are an HVAC expert evaluating the final cooling setpoints.\n\n"
            f"Context:\n{prompt_text}\n\n"
            f"Simulation Feedback Summary:\n{best_df.to_string(index=False)}\n\n"
            f"Setpoints Proposed: {best_setpoints}\n\n"
            f"Please explain concisely in 1–2 sentences why these setpoints were selected, "
            f"considering energy savings and indoor comfort trends."
        )


        reason_response = self._call_chat_api(reason_prompt).strip()

        try:
            maybe_json = json.loads(reason_response)
            if isinstance(maybe_json, dict) and "reason" in maybe_json:
                reason = maybe_json["reason"]
                print(f"[phiGPT] ✅ Extracted 'reason' from JSON:\n{repr(reason)}")
            else:
                reason = reason_response
                print(f"[phiGPT] ✅ Fallback to raw string reason:\n{repr(reason)}")
        except json.JSONDecodeError:
            reason = reason_response
            print(f"[phiGPT] ✅ Received plain string reason:\n{repr(reason)}")



        return {
            "optimal_cooling_setpoints": best_setpoints,
            "applied_setpoint": best_setpoints[0],
            "reason": reason,
            "log_path": log_path,
            "improved_prompt": prompt_with_feedback,
            "initial_score": initial_score,
            "final_score": best_score,
            "improvement": initial_score - best_score if initial_score and best_score else None
        }





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

