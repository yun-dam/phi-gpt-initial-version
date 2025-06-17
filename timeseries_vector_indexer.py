import pandas as pd
import numpy as np
import faiss
import os
from typing import List
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

class TimeSeriesVectorIndexer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def load_and_transform_csv(self, csv_path: str, sampling_interval: int = 3):
        df = pd.read_csv(csv_path)
        df = df.iloc[::sampling_interval].reset_index(drop=True)

        self.timestamp = df['Date/Time']
        t_out_col = 'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'
        self.outdoor_temp = df[t_out_col]

        # Identify all zone names
        zone_names = set()
        for col in df.columns:
            if ":Zone Air Temperature" in col:
                zone = col.split(":Zone Air Temperature")[0]
                zone_names.add(zone)

        long_data = []
        for zone in zone_names:
            t_in_col = f"{zone}:Zone Air Temperature [C](TimeStep)"
            t_set_col = f"{zone}:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"
            if t_in_col in df.columns and t_set_col in df.columns:
                for i in range(len(df)):
                    row = {
                        "timestamp": self.timestamp[i],
                        "zone": zone,
                        "T_out": self.outdoor_temp[i],
                        "T_in": df[t_in_col][i],
                        "T_set": df[t_set_col][i],
                    }
                    long_data.append(row)

        self.df_sampled = pd.DataFrame(long_data).dropna().reset_index(drop=True)

    def create_windows(self, window_size: int = 12):
        self.series_list = []
        self.metadata_list = []

        for zone, group_df in self.df_sampled.groupby("zone"):
            group_df = group_df.reset_index(drop=True)
            for i in range(len(group_df) - window_size + 1):
                window = group_df.iloc[i:i + window_size]
                values = window[["T_out", "T_in", "T_set"]].values.tolist()
                self.series_list.append(values)
                self.metadata_list.append({
                    "zone": zone,
                    "window_id": len(self.series_list) - 1
                })

    def build_vectorstore(self, save_path: str):
        documents = []
        vectors = []

        for i, series in enumerate(self.series_list):
            text = str(series)
            metadata = self.metadata_list[i]
            documents.append(Document(page_content=text, metadata=metadata))
            emb = self.embedding_model.embed_query(text)
            vectors.append(emb)

        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors).astype("float32"))

        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        vectorstore = FAISS(self.embedding_model.embed_query, index, docstore, index_to_docstore_id)
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)
        print(f"✅ Vectorstore saved to: {save_path}")


if __name__ == "__main__":
    csv_path = "./ep-model/simulation_data/gates_simulation_data_base.csv"
    output_path = "./data/timeseries"

    indexer = TimeSeriesVectorIndexer()
    indexer.load_and_transform_csv(csv_path, sampling_interval=3)
    indexer.create_windows(window_size=12)
    indexer.build_vectorstore(output_path)
