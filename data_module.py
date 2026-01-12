from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

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

    # Ekstra analiz fonksiyonları
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
    def get_columns(df: pd.DataFrame) -> list:
        return df.columns.tolist()

    @staticmethod
    def get_features_target(df: pd.DataFrame, target: str):
        if target not in df.columns:
            raise ValueError(f"'{target}' sütunu veri çerçevesinde bulunamadı.")
        X = df.drop(columns=[target])
        y = df[target]
        return X, y


# ----------------------------
# BaseLoader (ABC)
# ----------------------------
class BaseLoader(ABC):
    """
    Tüm Loader sınıfları için abstract base class.
    """
    def __init__(self, config: DataConfig):
        self.config = config
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
    Farklı formatlardaki verileri okuyup DataFrame döndüren somut sınıf.
    """
    def load(self) -> pd.DataFrame:
        if self.config.file_type == "csv":
            self.df = pd.read_csv(self.config.file_path)
        elif self.config.file_type in ["xls", "xlsx"]:
            self.df = pd.read_excel(self.config.file_path)
        elif self.config.file_type == "json":
            self.df = pd.read_json(self.config.file_path)
        else:
            raise ValueError(f"Desteklenmeyen dosya tipi: {self.config.file_type}")
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
        print(self.df.isnull().sum())


# ----------------------------
# BaseCleaner (ABC)
# ----------------------------
class BaseCleaner(ABC):
    """
    Temizleme işlemleri için abstract base class.
    """
    def __init__(self, df: pd.DataFrame, strategy: str = "mean"):
        self.df = df
        self.strategy = strategy

    @abstractmethod
    def handle_missing_values(self) -> pd.DataFrame:
        pass


# ----------------------------
# DataCleaner
# ----------------------------
class DataCleaner(BaseCleaner):
    """
    Eksik değerleri doldurma işlemleri.
    """
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
    """
    Ölçekleme işlemleri için abstract base class.
    """
    def __init__(self, df: pd.DataFrame, strategy: str = "standard"):
        self.df = df
        self.strategy = strategy

    @abstractmethod
    def scale(self) -> pd.DataFrame:
        pass


# ----------------------------
# DataScaler
# ----------------------------
class DataScaler(BaseScaler):
    """
    MinMax, Standard, Robust ölçekleme.
    """
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
