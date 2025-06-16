from db.db_ops import insert_upload_metadata, insert_sensor_data, insert_target_data
from data.preprocessor import load_csv, extract_features, extract_targets, generate_timestamp

# === Load + preprocess
df = load_csv("data/ai4i2020.csv")
features = extract_features(df)
binary, multi = extract_targets(df)
timestamps = generate_timestamp(len(df))

# === Upload metadata
upload_id = insert_upload_metadata(
    file_key="ai4i2020/ai4i2020.csv",
    data_type="sensor_csv",
    has_target=True,
    task_type="binary_classification",
    company_id="11111111-1111-1111-1111-111111111111",
    uploader_id="22222222-2222-2222-2222-222222222222"
)

# === Sensor & Target
insert_sensor_data(upload_id, timestamps, features)
insert_target_data(upload_id, binary, multi)
