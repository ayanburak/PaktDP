import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """
    Visualizer sınıfı, veri setinin görselleştirmelerini yapar.

    Özellikler:
    - Tek kolon bazlı görselleştirme
    - İki kolon / ilişki bazlı görselleştirme
    - Tablo bazlı görselleştirme
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df  # İşlem yapılacak DataFrame

        # Kolon bazlı görselleştirmeler
        self.column = ColumnVisualizer(self.df)

        # Tablo bazlı görselleştirmeler
        self.table = TableVisualizer(self.df)


class ColumnVisualizer:
    """
    Tek bir kolon veya iki kolon arasındaki görselleştirmeleri yapar.
    """

    def __init__(self, df):
        self.df = df

    def plot_histogram(self, column: str, bins: int = 30):
        """Tek kolonun histogramını çizer."""
        if column not in self.df.columns:
            raise ValueError(f"{column} DataFrame'de yok")
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[column], bins=bins, kde=True)
        plt.title(f"{column} Histogram")
        plt.show()

    def plot_boxplot(self, column: str):
        """Tek kolonun boxplot'unu çizer."""
        if column not in self.df.columns:
            raise ValueError(f"{column} DataFrame'de yok")
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=self.df[column])
        plt.title(f"{column} Boxplot")
        plt.show()

    def plot_bar(self, column: str):
        """Kategorik kolonların frekansını çizer."""
        if column not in self.df.columns:
            raise ValueError(f"{column} DataFrame'de yok")
        plt.figure(figsize=(8, 5))
        sns.countplot(x=self.df[column])
        plt.title(f"{column} Countplot")
        plt.show()

    def plot_scatter(self, x_col: str, y_col: str):
        """İki kolon arasındaki scatter plot."""
        if x_col not in self.df.columns or y_col not in self.df.columns:
            raise ValueError(f"{x_col} veya {y_col} DataFrame'de yok")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.df[x_col], y=self.df[y_col])
        plt.title(f"{x_col} vs {y_col}")
        plt.show()


class TableVisualizer:
    """
    Tüm tablo veya birden fazla kolonun görselleştirmelerini yapar.
    """

    def __init__(self, df):
        self.df = df

    def plot_correlation_matrix(self):
        """Sayısal kolonlar arasındaki korelasyon matrisini çizer."""
        numeric_df = self.df.select_dtypes(include="number")
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def plot_pairplot(self, columns: list):
        """Seçilen kolonlar için pairplot çizer."""
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"{col} DataFrame'de yok")
        sns.pairplot(self.df[columns])
        plt.show()
