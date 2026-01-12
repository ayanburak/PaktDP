from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Dict, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os

# ----------------------------
# DataConfig
# ----------------------------
class DataConfig:
    """
    Veri konfigürasyonu ve ayarlarını tutar.
    Ayrıca veri özetleme ve küçük analiz fonksiyonlarını içerir.
    """
    def __init__(self, file_path: str = "train.csv", file_type: str = "csv"):
        self.file_path = file_path
        self.file_type = file_type.lower()

    # ----------------------------
    # Analiz fonksiyonları
    # ----------------------------
    @staticmethod
    def get_zeros(df: pd.DataFrame) -> pd.Series:
        return (df == 0).sum()

    @staticmethod
    def get_missing(df: pd.DataFrame) -> pd.Series:
        return df.isnull().sum()

    @staticmethod
    def get_numeric(df: pd.DataFrame) -> pd.DataFrame:
        return df.select_dtypes(include=['int64','float64'])

    @staticmethod
    def get_categorical(df: pd.DataFrame) -> pd.DataFrame:
        return df.select_dtypes(include=['object','category'])

    @staticmethod
    def get_columns(df: pd.DataFrame) -> List[str]:
        return df.columns.tolist()

    @staticmethod
    def get_features_target(df: pd.DataFrame, target: str):
        if target not in df.columns:
            raise ValueError(f"'{target}' sütunu veri çerçevesinde bulunamadı.")
        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    @staticmethod
    def get_sample(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        return df.sample(n=n)

    @staticmethod
    def convert_types(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        for col, dtype in mapping.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
            else:
                print(f"Uyarı: '{col}' sütunu bulunamadı.")
        return df

    @staticmethod
    def save(df: pd.DataFrame, path: str, file_type: str = "csv"):
        file_type = file_type.lower()
        if file_type == "csv":
            df.to_csv(path, index=False)
        elif file_type in ["xls", "xlsx"]:
            df.to_excel(path, index=False)
        elif file_type == "json":
            df.to_json(path, orient="records")
        elif file_type == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Desteklenmeyen dosya tipi: {file_type}")


# ----------------------------
# BaseLoader (ABC)
# ----------------------------
class BaseLoader(ABC):
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def summary(self) -> None:
        pass


# ----------------------------
# DataLoader
# ----------------------------
class DataLoader(BaseLoader):
    """
    Dosya yolu, DataFrame veya dict ile çalışabilir.
    """
    def __init__(self, data: Union[str, pd.DataFrame, dict]):
        super().__init__()
        self.input_data = data

    def load(self) -> pd.DataFrame:
        if isinstance(self.input_data, pd.DataFrame):
            self.df = self.input_data
        elif isinstance(self.input_data, dict):
            self.df = pd.DataFrame(self.input_data)
        elif isinstance(self.input_data, str):
            ext = os.path.splitext(self.input_data)[1].lower()
            if ext == ".csv":
                self.df = pd.read_csv(self.input_data)
            elif ext in [".xls", ".xlsx"]:
                self.df = pd.read_excel(self.input_data)
            elif ext == ".json":
                self.df = pd.read_json(self.input_data)
            elif ext == ".parquet":
                self.df = pd.read_parquet(self.input_data)
            else:
                raise ValueError(f"Desteklenmeyen dosya tipi: {ext}")
        else:
            raise TypeError("Girdi tipi DataFrame, dict veya dosya yolu (str) olmalıdır.")
        return self.df

    def summary(self) -> None:
        if self.df is None:
            raise ValueError("Veri yüklenmemiş. Önce load() çağrılmalı.")
        print("=== Veri Özeti ===")
        print("Boyut:", self.df.shape)
        print("\nİlk 5 satır:")
        print(self.df.head())
        print("\nBilgi:")
        print(self.df.info())
        print("\nEksik değerler:")
        print(DataConfig.get_missing(self.df))
        print("\nSıfır değerler:")
        print(DataConfig.get_zeros(self.df))


# ----------------------------
# BaseCleaner (ABC)
# ----------------------------
class BaseCleaner(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @abstractmethod
    def handle_missing_values(self) -> pd.DataFrame:
        pass


# ----------------------------
# DataCleaner
# ----------------------------
class DataCleaner(BaseCleaner):
    def __init__(self, df: pd.DataFrame, strategy: str = "mean"):
        super().__init__(df)
        self.strategy = strategy

    def handle_missing_values(self) -> pd.DataFrame:
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.strategy == "mean":
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif self.strategy == "median":
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif self.strategy == "mode":
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    raise ValueError(f"Bilinmeyen strateji: {self.strategy}")
        return self.df


# ----------------------------
# BaseScaler (ABC)
# ----------------------------
class BaseScaler(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @abstractmethod
    def scale(self) -> pd.DataFrame:
        pass


# ----------------------------
# DataScaler
# ----------------------------
class DataScaler(BaseScaler):
    def __init__(self, df: pd.DataFrame, strategy: str = "standard"):
        super().__init__(df)
        self.strategy = strategy

    def scale(self) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include=['float64','int64']).columns
        if self.strategy == "minmax":
            scaler = MinMaxScaler()
        elif self.strategy == "standard":
            scaler = StandardScaler()
        elif self.strategy == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Bilinmeyen strateji: {self.strategy}")
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self.df
