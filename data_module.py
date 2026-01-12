from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Union, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# ==========================================
# 1. SOYUT KATMAN (KURALLAR)
# ==========================================
class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

class BaseProcessor(ABC):
    @abstractmethod
    def clean(self, numeric_strategy: str, cat_strategy: str) -> pd.DataFrame:
        pass
    @abstractmethod
    def scale(self, strategy: str, exclude_cols: List[str]) -> pd.DataFrame:
        pass


# ----------------------------
# DataLoader 
# ----------------------------
class DataLoader(BaseLoader):
    """
    Dosya yolu, Pandas DataFrame veya Dictionary formatındaki verileri 
    yüklemek ve yönetmek için kullanılan sınıf.
    """
    def __init__(self, input_data: Union[str, pd.DataFrame, dict]):
        self.input_data = input_data
        self.df = pd.DataFrame()

    def load(self) -> pd.DataFrame:
        # Girdi zaten bir Pandas DataFrame ise
        if isinstance(self.input_data, pd.DataFrame):
            self.df = self.input_data.copy()
            
        # Girdi bir Dictionary ise
        elif isinstance(self.input_data, dict):
            self.df = pd.DataFrame(self.input_data)
            
        # Girdi bir metin (dosya yolu) ise
        elif isinstance(self.input_data, str):
            if not os.path.exists(self.input_data):
                raise FileNotFoundError(f"Dosya bulunamadı: {self.input_data}")

            path_lower = self.input_data.lower()
            if path_lower.endswith(".csv"):
                self.df = pd.read_csv(self.input_data)
            elif path_lower.endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(self.input_data)
            elif path_lower.endswith(".json"):
                self.df = pd.read_json(self.input_data)
            elif path_lower.endswith(".parquet"):
                self.df = pd.read_parquet(self.input_data)
            else:
                raise ValueError(f"Desteklenmeyen dosya formatı: {os.path.basename(self.input_data)}")
        else:
            raise TypeError("Girdi tipi bir dosya yolu (str), dict veya DataFrame olmalıdır.")
        
        return self.df

# ----------------------------
# DataProcessor
# ----------------------------
class DataProcessor(BaseProcessor):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean(self, numeric_strategy: str = "mean", cat_strategy: str = "mode") -> pd.DataFrame:
        # Sütun tiplerini ayır
        num_cols = self.df.select_dtypes(include=['number']).columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns

        # Sayısal Doldurma
        for col in num_cols:
            if self.df[col].isnull().any():
                if numeric_strategy == "mean":
                    val = self.df[col].mean()
                elif numeric_strategy == "median":
                    val = self.df[col].median()
                else:
                    val = 0
                self.df[col] = self.df[col].fillna(val)

        # Kategorik Doldurma
        for col in cat_cols:
            if self.df[col].isnull().any():
                if cat_strategy == "mode":
                    mode_val = self.df[col].mode()
                    val = mode_val[0] if not mode_val.empty else "Missing"
                else:
                    val = "Unknown"
                self.df[col] = self.df[col].fillna(val)
        return self.df

    def scale(self, strategy: str = "standard", exclude_cols: List[str] = None) -> pd.DataFrame:
        if exclude_cols is None: exclude_cols = []
        
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]
        
        if not cols_to_scale: return self.df

        if strategy == "minmax": scaler = MinMaxScaler()
        elif strategy == "standard": scaler = StandardScaler()
        elif strategy == "robust": scaler = RobustScaler()
        else: raise ValueError(f"Unknown strategy: {strategy}")

        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
        return self.df

# ----------------------------
# DataConfig (Hataları Giderilmiş Helper Sınıfı)
# ----------------------------
class DataConfig:
    """Veri konfigürasyonu ve analiz araçları."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_basic_stats(self):
        return {
            "row_count": self.df.shape[0],
            "column_count": self.df.shape[1],
            "total_cells": self.df.size
        }
    
    def get_null_count(self) -> Dict[str, int]:
        nulls = self.df.isnull().sum()
        return nulls[nulls > 0].to_dict()
    
    def get_zeros_count(self) -> Dict[str, int]:
        zeros = (self.df == 0).sum()
        return zeros[zeros > 0].to_dict()
    
   
    def get_duplicate_rows(self) -> int:
        return int(self.df.duplicated().sum())

   
    def get_numeric_cols(self) -> List[str]:
        return self.df.select_dtypes(include=['number']).columns.tolist()
    
    def get_categorical_cols(self) -> List[str]:
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_columns(self) -> List[str]:
        return self.df.columns.tolist()
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        return self.df.sample(n=min(n, len(self.df))) # Veri küçükse hata vermesin diye min() eklendi
    
    def get_features_target(self, target: str):
        if target not in self.df.columns:
            raise ValueError(f"'{target}' sütunu bulunamadı.")
        X = self.df.drop(columns=[target])
        y = self.df[target]
        return X, y

    def save(self, path: str, file_type: str = "csv"):
        file_type = file_type.lower()
        if file_type == "csv": self.df.to_csv(path, index=False)
        elif file_type in ["xls", "xlsx"]: self.df.to_excel(path, index=False)
        elif file_type == "json": self.df.to_json(path, orient="records")
        elif file_type == "parquet": self.df.to_parquet(path, index=False)
        else: raise ValueError(f"Desteklenmeyen dosya tipi: {file_type}")

