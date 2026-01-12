import pandas as pd
import os
from typing import List, Dict

class DataLoader():
    """
    Dosya yolu, Pandas DataFrame veya Dictionary formatındaki verileri 
    yüklemek ve yönetmek için kullanılan sınıf.
    """
    def __init__(self, input_data):
        self.data = pd.DataFrame()
        
        # Eğer girdi zaten bir Pandas DataFrame ise
        if isinstance(input_data, pd.DataFrame):
            self.data = input_data
            
        # Eğer girdi bir Dictionary ise
        elif isinstance(input_data, dict):
            self.data = pd.DataFrame(input_data)
            
        # Eğer girdi bir metin (dosya yolu) ise
        elif isinstance(input_data, str):
            path_lower = input_data.lower()
            if path_lower.endswith(".csv"):
                self.data = pd.read_csv(input_data)
            elif path_lower.endswith((".xlsx", ".xls")):
                self.data = pd.read_excel(input_data)
            elif path_lower.endswith(".json"):
                self.data = pd.read_json(input_data)
            else:
                raise ValueError(f"Desteklenmeyen dosya formatı: {os.path.basename(input_data)}")
        
        else:
            raise TypeError("Girdi tipi bir dosya yolu (str), dict veya DataFrame olmalıdır.")

    def get_data(self):
        return self.data
    
class DataConfig():
    
    """
    Veri konfigürasyonu ve ayarlarını tutar.
    Ayrıca veri özetleme ve küçük analiz fonksiyonlarını içerir.
    
    İçerisine dataframe alır.
    """

    def __init__(self, df = pd.DataFrame()):
        self.df = df
        
    def get_basic_stats(self):
        return {
            "row_count": self.df.shape[0],
            "column_count": self.df.shape[1],
            "total_cells": self.df.size
        }
    
    # Hangi kolonda kaç tane Null değer olduğunu sayıp dict olarak döndürür
    def get_null_count(self):
        return self.df.isnull().sum().to_dict()
    
    # Hangi kolonda kaç tane Sıfır(0) değeri olduğunu sayıp dict olarak döndürür
    def get_zeros_count(self):
        return (self.df == 0).sum().to_dict()
    
    # Hangi kolonda hangi değerlerin tekrar ettiğini Dict formatında gösteren fonksiyon
    def get_duplicates(self):
            duplicate_report = {}
            
            for column in self.df.columns:
                # value_counts() her değerden kaç tane olduğunu sayar
                counts = self.df[column].value_counts()
                
                # Sadece sayısı 1'den büyük olanları (yani tekrar edenleri) filtreliyoruz
                duplicates = counts[counts > 1].to_dict()
                
                # Eğer o kolonda tekrar eden değer varsa rapora ekliyoruz
                if duplicates:
                    duplicate_report[column] = duplicates
                    
            return duplicate_report
    
    # Sadece sayısal kolonları döndürür [Dict]
    def get_numeric(self) -> Dict[str, List[str]]:
        return self.df.select_dtypes(include=['int64','float64']).to_dict()
    
    # Sadece kategorik (metinsel) kolonları döndürür [Dict] 
    def get_categorical(self) -> Dict[str, List[str]]:
        return self.df.select_dtypes(include=['object','category']).to_dict()
    
    # Kolonların isimlerini döndürür [List]
    def get_columns(self) -> List[str]:
        return self.df.columns.tolist()
    
    # Verilerden girilen sayı kadar örnek gösterir, varsayılan 5 [DataFrame]
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        return self.df.sample(n=n)
    
    # Girilen veri setindeki hedefi ve diğer kolonları ayırır, iki çıktı üretir.
    # x = Hedeflenen kolon [DataFrame]
    # y = Geri kalan bütün kolonlar [DataFrame] 
    def get_features_target(self, target: str) -> pd.DataFrame:
        if target not in self.df.columns:
            raise ValueError(f"'{target}' sütunu veri çerçevesinde bulunamadı.")
        X = self.df.drop(columns=[target])
        y = self.df[target]
        return X, y
    
    # Veri setini belirtilen dosya yoluna belirtilen doysa yoluyla kaydeder.
    # Varsayılan olarak csv kaydeder.
    def save(self, path: str, file_type: str = "csv"):
        file_type = file_type.lower()
        if file_type == "csv":
            self.df.to_csv(path, index=False)
        elif file_type in ["xls", "xlsx"]:
            self.df.to_excel(path, index=False)
        elif file_type == "json":
            self.df.to_json(path, orient="records")
        elif file_type == "parquet":
            self.df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Desteklenmeyen dosya tipi: {file_type}")
