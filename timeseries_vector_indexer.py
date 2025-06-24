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

    def zone_to_adu_name(self, zone_name):
        """
        Converts a given zone name to the corresponding ADU VAV HW RHT equipment name.

        Example:
            "THERMAL ZONE: STORY 4 EAST LOWER PERIMETER SPACE" -> "ADU VAV HW RHT 25"
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
            return "ADU VAV HW RHT" if index == 0 else f"ADU VAV HW RHT {index}"
        else:
            raise ValueError(f"Unknown zone name: {zone_name}")


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
            adu_name = self.zone_to_adu_name(zone)
            energy_col = f"{adu_name}:Zone Air Terminal Sensible Cooling Energy [J](TimeStep)"

            if t_in_col in df.columns and t_set_col in df.columns and energy_col in df.columns:
                for i in range(len(df)):
                    row = {
                        "timestamp": self.timestamp[i],
                        "zone": zone,
                        "T_out": self.outdoor_temp[i],
                        "T_in": df[t_in_col][i],
                        "T_set": df[t_set_col][i],
                        "Energy": df[energy_col][i]
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
                values = window[["T_out", "T_in", "T_set", "Energy"]].values.tolist()

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
