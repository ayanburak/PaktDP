import pandas as pd

class DataCleaner:
    """
    DataCleaner sınıfı, veri temizleme işlemleri için ana giriş noktasıdır.

    Bu sınıf doğrudan temizlik işlemi yapmaz.
    Sadece alt temizleyicileri (TableCleaner, ColumnCleaner) bir arada tutar.
    Böylece tablo bazlı ve kolon bazlı temizlik fonksiyonlarına
    tek bir sınıf üzerinden erişebiliriz.
    """

    def __init__(self, df):
        self.df = df  # İşlem yapılacak DataFrame'i sınıfta saklıyoruz

        # Tablo bazlı temizlik işlemlerini başlatıyoruz
        # Örnek: tüm tablo üzerindeki sıfır değerleri doldurma
        self.table = TableCleaner(self.df)

        # Kolon bazlı temizlik işlemlerini başlatıyoruz
        # Örnek: tek bir kolondaki null değerleri doldurma
        self.column = ColumnCleaner(self.df)
        self.row = RowCleaner(self.df) 


class TableCleaner:
    """
    TableCleaner sınıfı, tüm tabloyu ilgilendiren temizlik işlemlerini yapar.

    Özellikler:
    - Bütün kolonlar üzerinde işlem yapılabilir
    - Strategy pattern ile farklı doldurma stratejilerini yönetir
    - if/else blokları yerine dictionary dispatch kullanılır
    """

    def __init__(self, df):
        self.df = df  # İşlem yapılacak DataFrame

        # Sıfır doldurma stratejilerini dictionary ile eşleştiriyoruz
        # Kullanıcı "mode", "mean" veya "median" seçebilir
        self._fill_zero_dispatch = {
            "mode": self._fill_zeros_with_mode,
            "mean": self._fill_zeros_with_mean,
            "median": self._fill_zeros_with_median,
        }

    def fill_zeros(self, strategy: str):
        """
        Tablodaki 0 değerleri seçilen stratejiye göre doldurur.

        Bu fonksiyon doğrudan işlem yapmaz, sadece
        strategy parametresine göre ilgili fonksiyonu çağırır.
        """
        try:
            return self._fill_zero_dispatch[strategy]()
        except KeyError:
            # Kullanıcı geçersiz bir strateji verdiğinde hata fırlatılır
            raise ValueError(
                f"Geçersiz strateji: {strategy}. "
                f"Kullanılabilir stratejiler: {list(self._fill_zero_dispatch.keys())}"
            )

    def _fill_zeros_with_mode(self):
        """
        0 değerleri ilgili kolonun modu ile doldurur.

        Notlar:
        - Sadece 0 içeren sayısal kolonlar dikkate alınır
        - 0 değerleri moda hesaplamaya dahil edilmez
        - Mode bulunamazsa kolon atlanır
        """
        for col in self.df.select_dtypes(include="number").columns:
            if (self.df[col] == 0).any():
                mode_val = self.df.loc[self.df[col] != 0, col].mode()
                if not mode_val.empty:
                    # 0 olan hücreleri mod değeri ile değiştir
                    self.df.loc[self.df[col] == 0, col] = mode_val[0]

        return self.df

    def _fill_zeros_with_mean(self):
        """
        0 değerleri kolonların ortalaması ile doldurur.

        Notlar:
        - Sadece sayısal kolonlar dikkate alınır
        - 0 değerleri ortalama hesabına dahil edilmez
        """
        for col in self.df.select_dtypes(include="number").columns:
            mean_val = self.df.loc[self.df[col] != 0, col].mean()
            self.df.loc[self.df[col] == 0, col] = mean_val

        return self.df

    def _fill_zeros_with_median(self):
        """
        0 değerleri kolonların medyanı ile doldurur.

        Notlar:
        - Medyan, uç değerlere karşı daha dayanıklıdır
        - Sadece sayısal kolonlar dikkate alınır
        """
        for col in self.df.select_dtypes(include="number").columns:
            median_val = self.df.loc[self.df[col] != 0, col].median()
            self.df.loc[self.df[col] == 0, col] = median_val

        return self.df


class ColumnCleaner:
    """
    ColumnCleaner sınıfı, tek bir kolona özel temizlik işlemlerini yapar.

    Özellikler:
    - Kolon bazlı null doldurma işlemleri
    - Strategy pattern kullanılır
    - Sayısal ve kategorik kolon ayrımı bilinçli yapılır
    """

    def __init__(self, df):
        self.df = df 

        # Null doldurma stratejilerini dictionary ile eşleştiriyoruz
        # Kullanıcı "mean", "median" veya "mode" seçebilir
        self._null_dispatch = {
            "mean": self._fill_nulls_with_mean,
            "median": self._fill_nulls_with_median,
            "mode": self._fill_nulls_with_mode,
        }

    def fill_nulls(self, column: str, strategy: str):
        """
        Belirtilen kolondaki null değerleri seçilen stratejiye göre doldurur.
        """
        if column not in self.df.columns:
            raise ValueError(f"{column} kolonu DataFrame’de yok")

        try:
            return self._null_dispatch[strategy](column)
        except KeyError:
            # Kullanıcı geçersiz bir strateji verdiğinde hata fırlatılır
            raise ValueError(
                f"Geçersiz strateji: {strategy}. "
                f"Kullanılabilir stratejiler: {list(self._null_dispatch.keys())}"
            )

    def _validate_numeric_column(self, column: str):
        """
        Kolonun sayısal olup olmadığını kontrol eder.

        Mean ve median gibi istatistiksel işlemler sadece sayısal kolonlarda anlamlıdır.
        """
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"{column} sayısal bir kolon değil")

    def _fill_nulls_with_mean(self, column: str):
        """
        Null değerleri kolon ortalaması ile doldurur.
        """
        self._validate_numeric_column(column)
        mean_val = self.df[column].mean()
        self.df[column].fillna(mean_val, inplace=True)
        return self.df

    def _fill_nulls_with_median(self, column: str):
        """
        Null değerleri kolon medyanı ile doldurur.
        """
        self._validate_numeric_column(column)
        median_val = self.df[column].median()
        self.df[column].fillna(median_val, inplace=True)
        return self.df

    def _fill_nulls_with_mode(self, column: str):
        """
        Null değerleri kolonun modu ile doldurur.

        Notlar:
        - Sayısal veya kategorik kolonlarda çalışır
        - Birden fazla mode varsa ilk değer kullanılır
        - Mode bulunamazsa (örn. tüm değerler NaN) kolon değiştirilmez
        """
        mode_val = self.df[column].mode(dropna=True)

        if not mode_val.empty:
            self.df[column].fillna(mode_val[0], inplace=True)

        return self.df

class RowCleaner:
    """
    RowCleaner sınıfı, veri setindeki satır bazlı temizlik işlemlerini yapar.

    Özellikler:
    - Eksik veri oranı yüksek satırları kaldırma
    - Duplicate (çift) satırları kaldırma
    - Mantıksal hatalara sahip satırları filtreleme
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df  # İşlem yapılacak DataFrame

    def drop_rows_with_missing_threshold(self, threshold: float = 0.5):
        """
        Eksik veri oranı threshold'dan fazla olan satırları siler.
        """
        # Eksik değer oranını hesapla
        missing_ratio = self.df.isnull().mean(axis=1)
        # Threshold'u aşan satırları sil
        self.df = self.df[missing_ratio <= threshold]
        return self.df

    def drop_duplicate_rows(self):
        """
        Aynı satırların tekrarını kaldırır.
        """
        self.df = self.df.drop_duplicates()
        return self.df

    def drop_rows_with_condition(self, condition_func):
        """
        Kullanıcının verdiği mantıksal koşula göre satırları siler.

        Parametreler:
        ----------
        condition_func : function
            Her satırı alıp True/False döndüren fonksiyon.
            True dönen satırlar silinir.

        Örnek:
        >>> rc.drop_rows_with_condition(lambda row: row['age'] < 0 or row['salary'] < 0)
        """
        mask = self.df.apply(condition_func, axis=1)
        self.df = self.df[~mask]
        return self.df