import os
import boto3
from config import S3_CONFIG

def download_file(s3_key: str, local_filename: str = None):
    """
    Downloads a single file from S3 to local /data folder.
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=S3_CONFIG["aws_access_key"],
        aws_secret_access_key=S3_CONFIG["aws_secret_key"]
    )

    if not local_filename:
        local_filename = os.path.join("data", os.path.basename(s3_key))
    
    s3.download_file(S3_CONFIG["bucket_name"], s3_key, local_filename)
    print(f"✅ Downloaded {s3_key} → {local_filename}")
    return local_filename
