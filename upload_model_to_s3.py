import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def upload_model_to_s3(local_file_path, bucket_name, s3_file_key):
    """
    Uploads a file from the local file system to an S3 bucket.
    
    :param local_file_path: Path to the local model file
    :param bucket_name: Name of the S3 bucket
    :param s3_file_key: S3 key for the uploaded file (the file name in the bucket)
    """
    # Initialize an S3 client
    s3 = boto3.client('s3')
    
    try:
        # Upload the file   
        s3.upload_file(local_file_path, bucket_name, s3_file_key)
        print(f"File {local_file_path} uploaded to S3 bucket {bucket_name} as {s3_file_key}.")
    except FileNotFoundError:
        print(f"The file {local_file_path} was not found.")
    except NoCredentialsError:
        print("Credentials not available for AWS.")
    except ClientError as e:
        print(f"An error occurred: {e}")
        
# Example usage
local_file_path = "path/to/your/model.pt"  # Path to your model file
bucket_name = "your-s3-bucket-name"
s3_file_key = "models/latest_model.pt"     # S3 key, e.g., the path in the bucket

upload_model_to_s3(local_file_path, bucket_name, s3_file_key)
