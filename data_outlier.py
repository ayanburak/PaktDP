import pandas as pd
import numpy as np

class OutlierCleaner:
    """
    OutlierCleaner sınıfı, veri setindeki uç değerleri (outlier) tespit edip temizler.

    Kullanıcı iki yöntemden birini seçebilir:
    - IQR (Interquartile Range): Çeyrekler arası mesafeye göre uç değerleri belirler
    - Z-score: Standart skor yöntemine göre uç değerleri belirler
    """

    def __init__(self, df):
        self.df = df 

        self._strategy_dispatch = {
            "IQR": self._remove_outliers_iqr,
            "zscore": self._remove_outliers_zscore,
        }

    def remove_outliers(self, strategy: str, columns: list, z_thresh: float = 3.0):
        """
        Belirtilen kolonlardaki outlier değerleri temizler.

        """
        if strategy not in self._strategy_dispatch:
            raise ValueError(
                f"Geçersiz strateji: {strategy}. "
                f"Kullanılabilir stratejiler: {list(self._strategy_dispatch.keys())}"
            )
        return self._strategy_dispatch[strategy](columns, z_thresh=z_thresh)

    def _remove_outliers_iqr(self, columns: list, **kwargs):
        """
        IQR yöntemi ile outlier'ları temizler.

        Adımlar:
        1. Her kolon için Q1 ve Q3 hesaplanır
        2. IQR = Q3 - Q1
        3. Alt ve üst sınırlar belirlenir: Q1 - 1.5*IQR, Q3 + 1.5*IQR
        4. Bu sınırların dışında kalan satırlar çıkarılır
        """
        df_clean = self.df.copy()
        for col in columns:
            # Kolon yoksa hata 
            if col not in df_clean.columns:
                raise ValueError(f"{col} kolonu DataFrame’de yok")
            # Sayısal değilse hata 
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                raise TypeError(f"{col} sayısal bir kolon değil")
            
            # Q1 ve Q3 hesaplama
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            # Alt ve üst sınırları belirleme
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Sınırlar dışındaki değerleri çıkar
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean

    def _remove_outliers_zscore(self, columns: list, z_thresh: float = 3.0, **kwargs):
        """
        Z-score yöntemi ile outlier'ları temizler.

        Adımlar:
        1. Her kolon için ortalama ve standart sapma hesaplanır
        2. Z-score = (değer - ortalama) / std
        3. Z-score eşik değerini aşan satırlar çıkarılır
        """
        df_clean = self.df.copy()
        for col in columns:
            # Kolon yoksa hata 
            if col not in df_clean.columns:
                raise ValueError(f"{col} kolonu DataFrame’de yok")
            # Sayısal değilse hata 
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                raise TypeError(f"{col} sayısal bir kolon değil")
            
            # Ortalama ve standart sapma
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()

            # Z-score hesaplama
            z_score = (df_clean[col] - mean_val) / std_val

            # Z-score eşik değerini aşanları çıkar
            df_clean = df_clean[np.abs(z_score) <= z_thresh]
        return df_clean
