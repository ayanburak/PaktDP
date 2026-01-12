from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Dict, Any, Union, List
import os

# ----------------------------
# BaseComponent
# ----------------------------
class BaseComponent(ABC):
    def __init__(self, df: pd.DataFrame, params: Dict[str, Any] = None):
        self.df = df.copy()  # Orijinali bozmamak için kopya
        self.params = params if params else {}

    @abstractmethod
    def execute(self) -> pd.DataFrame:
        pass

# ----------------------------
# DataCleaner 
# ----------------------------
class DataCleaner(BaseComponent):
    def execute(self) -> pd.DataFrame:
        df_copy = self.df.copy()
        
        # Parametreleri al (Varsayılanlar belirtildi)
        num_strategy = self.params.get("numeric_strategy", "mean")
        cat_strategy = self.params.get("categorical_strategy", "mode")
        
        # Sütun tiplerini ayır
        numeric_cols = df_copy.select_dtypes(include=['number']).columns
        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns

        # 1. Sayısal Doldurma
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                if num_strategy == "mean":
                    val = df_copy[col].mean()
                elif num_strategy == "median":
                    val = df_copy[col].median()
                else:
                    val = 0
                df_copy[col] = df_copy[col].fillna(val)

        # 2. Kategorik Doldurma
        for col in cat_cols:
            if df_copy[col].isnull().any():
                if cat_strategy == "mode":
                    mode_val = df_copy[col].mode()
                    val = mode_val[0] if not mode_val.empty else "Missing"
                else:
                    val = "Unknown"
                df_copy[col] = df_copy[col].fillna(val)
                
        return df_copy

# ----------------------------
# DataScaler 
# ----------------------------
class DataScaler(BaseComponent):
    def execute(self) -> pd.DataFrame:
        df_copy = self.df.copy()
        
        strategy = self.params.get("strategy", "standard")
        exclude_cols = self.params.get("exclude_cols", []) # Hariç tutulacaklar
        
        # Sadece sayısal olanları bul
        numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
        
        # Hariç tutulacakları çıkar
        cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]
        
        if not cols_to_scale:
            return df_copy # Ölçeklenecek sütun yoksa aynen dön

        # Scaler seçimi
        if strategy == "minmax":
            scaler = MinMaxScaler()
        elif strategy == "standard":
            scaler = StandardScaler()
        elif strategy == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # İşlemi uygula
        df_copy[cols_to_scale] = scaler.fit_transform(df_copy[cols_to_scale])
        
        return df_copy

# ----------------------------
# DataLoader
# ----------------------------
class DataLoader:
    def __init__(self, input_data: Union[str, pd.DataFrame, dict]):
        self.df = pd.DataFrame()
        
        if isinstance(input_data, pd.DataFrame):
            self.df = input_data.copy()
        elif isinstance(input_data, dict):
            self.df = pd.DataFrame(input_data)
        elif isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"File not found: {input_data}")
                
            ext = input_data.lower()
            if ext.endswith(".csv"):
                self.df = pd.read_csv(input_data)
            elif ext.endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(input_data)
            elif ext.endswith(".json"):
                self.df = pd.read_json(input_data)
            elif ext.endswith(".parquet"):
                self.df = pd.read_parquet(input_data)
            else:
                raise ValueError(f"Unsupported file type: {os.path.basename(input_data)}")
        else:
            raise TypeError("Input must be DataFrame, dict, or file path (str).")

    def get_data(self) -> pd.DataFrame:
        return self.df.copy()

# ----------------------------
# DataPipeline
# ----------------------------
class DataPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def run(self, steps: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Adımları sırayla çalıştırır.
        steps formatı: { 'cleaner': {...params...}, 'scaler': {...params...} }
        """
        for step_name, params in steps.items():
            print(f"Running step: {step_name}...") # Loglama eklendi
            
            if step_name == "cleaner":
                component = DataCleaner(self.df, params)
            elif step_name == "scaler":
                component = DataScaler(self.df, params)
            else:
                raise ValueError(f"Unknown pipeline step: {step_name}")
            
            # Güncel dataframe'i bir sonraki adım için güncelle
            self.df = component.execute()
            
        return self.df

