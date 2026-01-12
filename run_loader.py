from data_module import DataConfig, DataLoader, DataCleaner, DataScaler

# 1️⃣ Veri yükleme
config = DataConfig("train.csv")        # Dosya yolu burada
loader = DataLoader(config.file_path)   # Sadece path (str) veriyoruz
df = loader.load()
loader.summary()

# 2️⃣ DataConfig analiz
print("\nSıfır değerler:\n", DataConfig.get_zeros(df))
print("\nEksik değerler:\n", DataConfig.get_missing(df))

# 3️⃣ Temizleme
cleaner = DataCleaner(df, strategy="mean")
df = cleaner.handle_missing_values()

# 4️⃣ Ölçekleme
scaler = DataScaler(df, strategy="standard")
df_scaled = scaler.scale()

# 5️⃣ Örnek veri
print("\nÖlçeklenmiş veriden ilk 5 satır:\n", df_scaled.head())
