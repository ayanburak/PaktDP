"""
Veri Yükleme ve Otomatik Raporlama Modülü
=========================================

Bu modül, veri bilimi projelerinde kullanılmak üzere farklı kaynaklardan 
veri yükleme, dönüştürme ve görselleştirme işlemlerini kolaylaştıran 
araçları içerir.

Temel Özellikler:
-----------------
* Farklı dosya formatlarını (.csv, .xlsx, .json) tek tip DataFrame'e dönüştürme.
* Python sözlüklerini (dict) veri çerçevesine çevirme.
* Sweetviz kütüphanesi entegrasyonu ile otomatik EDA (Keşifçi Veri Analizi) raporu oluşturma.

Sınıflar:
---------
    DataLoader: Veri yükleme ve raporlama işlemlerini yöneten ana sınıf.
"""
import pandas as pd
import os
import sweetviz as sv


class DataLoader():
    """
    Description:
    -
    Çeşitli kaynaklardan veri yüklemek, yönetmek ve otomatik keşifsel veri analizi
    raporu oluşturmak için tasarlanmış yardımcı sınıf.

    Bu sınıf; CSV, Excel, JSON dosyalarını, Python sözlüklerini (dict) veya mevcut 
    Pandas DataFrame'lerini otomatik olarak işleyerek standart bir DataFrame formatına 
    dönüştürür. Ayrıca Sweetviz kütüphanesi aracılığıyla verinin HTML formatında 
    görsel analiz raporunu oluşturur.

    Attributes:
        data (pd.DataFrame): Yüklenen ve işlenen veriyi tutan temel veri çerçevesi.

    Supported Formats:
        - .csv
        - .xlsx / .xls
        - .json
        - dict
        - pd.DataFrame

    Examples:
        >>> loader = DataLoader("satislar.csv")
        >>> loader.get_report("Satis_Analizi")
        >>> df = loader.get_data()
    """
    
    # Girilen verinin uzantısı:
    # .csv / .xlsx / .xls / .json veya
    # pd.DataFrame ya da Dict ise
    # Sınıfın içinde self.data nesnesine DataFrame olarak kaydeder.
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

    # Girilen veri setinin özetini hızlıca görsel olarak görebilmek için çağırılan fonksiyon
    # Çıkarılacak raporun ismi .html ile veya .html olmadan eklenmelidir
    def get_report(self, report_name = "rapor.html"):  
        # Kullanıcı ".html" yazmış mı kontrol et, yazmamışsa ekle
        if not report_name.endswith(".html"):
            report_name += ".html"
            
        report = sv.analyze(self.data)
        report.show_html(report_name)
    
    # Girilen veriyi sadece döndüren fonksiyon
    # DataFrame olarak döndürür
    def get_data(self) -> pd.DataFrame:
        return self.data

