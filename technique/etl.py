"""
ETL Pipeline for BDKT Dataset
Loads, cleans, and preprocesses data from CSV files
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BDKTDataLoader:
    """Load and preprocess BDKT dataset"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.df = None
        self.skills_metadata = None
        self.num_skills = 0
        self.num_students = 0
        self.num_items = 0
        
    def load_data(self) -> pd.DataFrame:
        """Load all data sources from directory"""
        logger.info("Loading data from directory...")
        
        # Load main dataset
        dataset_path = self.data_dir / "synthetic_bdkt_dataset.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        self.df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(self.df)} rows from dataset")
        
        # Load skills metadata
        skills_path = self.data_dir / "skills_metadata.csv"
        if skills_path.exists():
            self.skills_metadata = pd.read_csv(skills_path)
            self.num_skills = len(self.skills_metadata)
            logger.info(f"Loaded {self.num_skills} skills metadata")
        
        # Load stats if available
        stats_path = self.data_dir / "synthetic_bdkt_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
                self.num_students = stats.get("num_students", self.df["student_id"].nunique())
                self.num_items = stats.get("num_items", self.df["item_id"].nunique())
                logger.info(f"Stats: {self.num_students} students, {self.num_items} items")
        
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and validate data"""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(self.df)} duplicate rows")
        
        # Handle missing values
        self.df = self.df.dropna(subset=["student_id", "item_id", "response", "timestamp"])
        logger.info(f"Rows after removing NaN: {len(self.df)}")
        
        # Convert timestamp to datetime
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        
        # Ensure response is binary
        self.df["response"] = self.df["response"].astype(int)
        self.df = self.df[self.df["response"].isin([0, 1])]
        
        # Sort by student and timestamp
        self.df = self.df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)
        logger.info(f"Data cleaned. Final shape: {self.df.shape}")
        
        return self.df
    
    def encode_multi_hot_skills(self) -> np.ndarray:
        """Create multi-hot encoding of skills for each interaction"""
        logger.info("Creating multi-hot skill encodings...")
        
        if self.num_skills == 0:
            self.num_skills = self.df["skill_ids"].str.split("|").apply(lambda x: max(map(int, x))).max() + 1
        
        n_rows = len(self.df)
        skill_matrix = np.zeros((n_rows, self.num_skills), dtype=np.float32)
        
        for idx, skill_str in enumerate(self.df["skill_ids"]):
            if pd.isna(skill_str):
                continue
            skills = [int(s) for s in str(skill_str).split("|")]
            for skill_id in skills:
                if 0 <= skill_id < self.num_skills:
                    skill_matrix[idx, skill_id] = 1.0
        
        logger.info(f"Multi-hot encoding shape: {skill_matrix.shape}")
        return skill_matrix
    
    def apply_log_time_transform(self) -> pd.Series:
        """Apply log(1+x) transformation to time_since_last"""
        logger.info("Applying log(1+x) transformation to time features...")
        
        # Handle missing values
        self.df["time_since_last"] = self.df["time_since_last"].fillna(0)
        
        # Log transform
        time_log = np.log1p(self.df["time_since_last"].values)
        
        logger.info(f"Time transform - min: {time_log.min():.3f}, max: {time_log.max():.3f}, mean: {time_log.mean():.3f}")
        return time_log
    
    def create_sequences(self, window_length: int = 100, stride: int = 80) -> Tuple[List, List, List]:
        """Create windowed sequences with stride
        
        Args:
            window_length: L = 100
            stride: L - 20 = 80
        """
        logger.info(f"Creating sequences with window_length={window_length}, stride={stride}...")
        
        sequences_x = []  # (window_length, num_skills)
        sequences_y = []  # (window_length,) - responses
        sequences_t = []  # (window_length,) - time gaps
        student_ids = []
        
        skill_matrix = self.encode_multi_hot_skills()
        time_log = self.apply_log_time_transform()
        
        for student_id in self.df["student_id"].unique():
            student_mask = self.df["student_id"] == student_id
            student_indices = np.where(student_mask)[0]
            
            if len(student_indices) < window_length:
                continue
            
            # Create windows with stride
            for start_idx in range(0, len(student_indices) - window_length + 1, stride):
                end_idx = start_idx + window_length
                window_indices = student_indices[start_idx:end_idx]
                
                seq_x = skill_matrix[window_indices]  # (window_length, num_skills)
                seq_y = self.df.iloc[window_indices]["response"].values.astype(np.float32)
                seq_t = time_log[window_indices].astype(np.float32)
                
                sequences_x.append(seq_x)
                sequences_y.append(seq_y)
                sequences_t.append(seq_t)
                student_ids.append(student_id)
        
        logger.info(f"Created {len(sequences_x)} sequences")
        return sequences_x, sequences_y, sequences_t, student_ids
    
    def get_processed_data(self, window_length: int = 100, stride: int = 80) -> Dict:
        """Full pipeline: load -> clean -> encode -> transform -> sequence"""
        self.load_data()
        self.clean_data()
        
        seq_x, seq_y, seq_t, student_ids = self.create_sequences(window_length, stride)
        
        return {
            "sequences_x": seq_x,
            "sequences_y": seq_y,
            "sequences_t": seq_t,
            "student_ids": student_ids,
            "num_skills": self.num_skills,
            "num_students": self.num_students,
            "num_items": self.num_items,
        }


if __name__ == "__main__":
    loader = BDKTDataLoader(".")
    data = loader.get_processed_data()
    print(f"\nDataset Summary:")
    print(f"  Sequences: {len(data['sequences_x'])}")
    print(f"  Skills: {data['num_skills']}")
    print(f"  Students: {data['num_students']}")
    print(f"  Items: {data['num_items']}")
