# s3_utils.py
import boto3
import os

session = boto3.Session(profile_name='s3-uploader')
s3 = boto3.client("s3")

def upload_to_s3(local_path, bucket_name, s3_prefix):
    uploaded = []
    if os.path.isfile(local_path):
        filename = os.path.basename(local_path)
        s3_key = f"{s3_prefix}/{filename}"
        s3.upload_file(local_path, bucket_name, s3_key)
        uploaded.append(s3_key)
    else:
        for root, _, files in os.walk(local_path):
            for file in files:
                full_path = os.path.join(root, file)
                relative = os.path.relpath(full_path, local_path)
                s3_key = f"{s3_prefix}/{relative.replace(os.sep, '/')}"
                s3.upload_file(full_path, bucket_name, s3_key)
                uploaded.append(s3_key)
    return uploaded