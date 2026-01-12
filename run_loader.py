from data_module import DataConfig, DataLoader, DataCleaner, DataScaler

# 1️⃣ Veri yükleme
config = DataConfig("train.csv")
loader = DataLoader(config)
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
