from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# ----------------------------
# BaseCleaner
# ----------------------------
class BaseCleaner(ABC):
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
                    raise ValueError(f"Unknown strategy: {self.strategy}")
        return self.df

# ----------------------------
# BaseScaler
# ----------------------------
class BaseScaler(ABC):
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
    def scale(self) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include=['float64','int64']).columns
        if self.strategy == "minmax":
            scaler = MinMaxScaler()
        elif self.strategy == "standard":
            scaler = StandardScaler()
        elif self.strategy == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self.df
