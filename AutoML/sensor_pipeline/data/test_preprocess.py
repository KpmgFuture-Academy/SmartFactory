from data.preprocessor import load_csv, extract_features, extract_targets, generate_timestamp

df = load_csv("data/ai4i2020.csv")
features = extract_features(df)
binary, multi = extract_targets(df)
timestamps = generate_timestamp(len(df))

print("🧪 Sample row:", features[0])
print("⏱ Sample timestamp:", timestamps[0])
print("🎯 Sample target:", binary[0], multi[0])
