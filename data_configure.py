import pandas as pd

class TableManager:
    """
    """
    def __init__(self, df = pd.DataFrame):
        self.df = df
    
    # Her kolonun kaç sıfır değerine sahip olduğu bilgisini verir
    # En alt satırda totalde kaç sıfır olduğu bilgisini verir
    # Bir DataFrame döndürür
    def check_zeros(self) -> pd.DataFrame:
        # Her sütundaki sıfırların sayısını alır
        zero_count_by_column = (self.df == 0).sum()
        
        # True ve false olarak indekslemesini bozduğumuz zero_count_by_column
        # DataFrame'ini eski haline çevirip sütunları isimlendirir
        result_df = zero_count_by_column.reset_index()
        result_df.columns = ['Column', 'Zero Count']
        
        # Toplam sıfır sayısını en sona eklemek için bir dict olarak son satıra ekler
        total_zeros = zero_count_by_column.sum()
        total_row = pd.DataFrame([{'Column': 'Total', 'Zero Count': total_zeros}])
        
        result_df = pd.concat([result_df, total_row], ignore_index=True)
        
        # Sonuçları döndürür [DataFrame]
        return result_df

    # Her kolonun kaç null değerine sahip olduğu bilgisini verir
    # En alt satırda totalde kaç null olduğu bilgisini verir
    # Bir dataframe döndürür
    def check_nulls(self):
        pass
    
    def check_duplicate(self):
        pass
    
class ColumnManager:
    def __init__(self, df = pd.DataFrame):
        self.df = df
    def get_stats(self):
        pass
    
class CompareManager:
    def __init__(self, df = pd.DataFrame):
        self.df = df
    def check(self):
        pass
    
class DataConfig:
    def __init__(self, df = pd.DataFrame):
        self.df = df
        
        self.table = TableManager(self.df)
        self.column = ColumnManager(self.df)
        self.compare = CompareManager(self.df)
        
    