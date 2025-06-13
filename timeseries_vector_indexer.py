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

    def load_csv(self, csv_path: str, columns: List[str] = ["T_out", "T_in", "T_set"], sampling_interval: int = 3):
        """
        Load and preprocess the CSV file.
        Args:
            csv_path: Path to the input CSV.
            columns: Columns to extract.
            sampling_interval: Row interval (e.g., 3 for 30-minute step if data is every 10 minutes).
        """
        df = pd.read_csv(csv_path)
        self.df_sampled = df[columns].dropna().iloc[::sampling_interval].reset_index(drop=True)

    def create_windows(self, window_size: int = 12):
        """
        Create sliding windows from the sampled data.
        Args:
            window_size: Number of steps in each window (e.g., 12 = 6 hours for 30-minute intervals).
        """
        self.series_list = [
            self.df_sampled.iloc[i:i+window_size].values.tolist()
            for i in range(len(self.df_sampled) - window_size + 1)
        ]

    def build_vectorstore(self, save_path: str):
        """
        Build FAISS vectorstore and save it to disk.
        Args:
            save_path: Directory path to store the FAISS vectorstore.
        """
        documents = []
        vectors = []

        for i, series in enumerate(self.series_list):
            text = str(series)
            documents.append(Document(page_content=text, metadata={"window_id": i}))
            emb = self.embedding_model.embed_query(text)
            vectors.append(emb)

        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors).astype("float32"))

        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        vectorstore = FAISS(self.embedding_model.embed_query, index, docstore, index_to_docstore_id)

        # Ensure the save path exists
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)
        print(f"✅ Vectorstore saved to: {save_path}")

if __name__ == "__main__":
    # Input CSV and output directory
    csv_path = "./data/timeseries/ep_simulation_may15_aug15.csv"
    output_path = "./data/timeseries"

    # Process
    indexer = TimeSeriesVectorIndexer()
    indexer.load_csv(csv_path, columns=["T_out", "T_in", "T_set"], sampling_interval=3)
    indexer.create_windows(window_size=12)
    indexer.build_vectorstore(output_path)
