from google.cloud import storage
import os
import sys
BUCKET_NAME = os.getenv('BUCKET_NAME','us-central2-storage')
USER_NAME = os.getenv('USER_NAME', 'boyang-ckpt')
def ensure_data_safety(blob_or_dir_name):
    # assure the data is a subdir/file of USER_NAME/ or USER_NAME
    global USER_NAME
    if not blob_or_dir_name.startswith(USER_NAME):
        raise ValueError(f"Data {blob_or_dir_name} is not a subdir/file of {USER_NAME}")
def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    ensure_data_safety(blob_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    # check whether the blob exists
    blobs = list(bucket.list_blobs(prefix=blob_name)) if blob_name.endswith('/') else [bucket.blob(blob_name)]
    if not blobs:
        raise ValueError(f"FAIL; No blobs found with prefix {blob_name}")
    for blob in blobs:
        blob: storage.Blob
        generation_match_precondition = None
        # Optional: set a generation-match precondition to avoid potential race conditions
        # and data corruptions. The request to delete is aborted if the object's
        # generation number does not match your precondition.
        blob.reload()  # Fetch blob metadata to use in generation_match_precondition.
        generation_match_precondition = blob.generation
        blob.delete(if_generation_match=generation_match_precondition)
        print(f"Blob {blob.name} deleted.")
def get_blobs_list(bucket_name, prefix):
    """Lists all the blobs in the bucket."""
    ensure_data_safety(prefix)
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    print("Blobs:")
    for blob in blobs:
        print(blob.name)
    return blobs
def upload_dir(bucket_name, source_blob_name, destination_blob_name):
    """Uploads a directory to a bucket."""
    ensure_data_safety(destination_blob_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if not os.path.exists(source_blob_name):
        raise ValueError(f"Directory {source_blob_name} does not exist.")
    if os.path.isfile(source_blob_name):
        # see if destination_blob_name is a directory
        if destination_blob_name.endswith('/'):
            destination_blob_name = os.path.join(destination_blob_name, os.path.basename(source_blob_name))
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_blob_name)
        print(f"File {source_blob_name} uploaded to {destination_blob_name}.")
        return
    base_directory = os.path.basename(source_blob_name)
    # Iterate over the files in the directory
    for dirpath, _, filenames in os.walk(source_blob_name):
        for filename in filenames:
            # Upload the file to the destination
            destination = os.path.join(destination_blob_name, base_directory, os.path.relpath(dirpath, source_blob_name), filename)
            # merge useless dir like ./
            destination = os.path.normpath(destination)
            blob = bucket.blob(destination)
            blob.upload_from_filename(os.path.join(dirpath, filename))
            print(f"File {os.path.join(dirpath, filename)} uploaded to {destination}.")
def download_dir(bucket_name, source_blob_name, destination_dir_name):
    """Downloads a directory from a bucket."""
    ensure_data_safety(source_blob_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if not source_blob_name.endswith('/'):  
        # see if destination_blob_name is a directory
        if destination_dir_name.endswith('/'):
            destination_dir_name = os.path.join(destination_dir_name, os.path.basename(source_blob_name))
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_dir_name)
        print(f"File {source_blob_name} downloaded to {destination_dir_name}.")
        return
    # List all objects that satisfy the filter
    blobs = list(bucket.list_blobs(prefix=source_blob_name))
    print(f"Found {len(blobs)} objects in {source_blob_name}. blob names:", [blob.name for blob in blobs])
    if not blobs:
        print(f"No objects found in {source_blob_name}.")
    # remove all blobs that are directories
    blobs = [blob for blob in blobs if not blob.name.endswith('/')]
    # Iterate over the blobs and download them
    for blob in blobs:
        # Get the full path to the file
        destination = os.path.join(destination_dir_name, os.path.relpath(blob.name, source_blob_name))
        destination = os.path.normpath(destination)
        # Ensure the directory that the file is going to be saved in exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        # Download the file
        blob.download_to_filename(destination)
        print(f"File {blob.name} downloaded to {destination}.")
instr_to_func = {
    'upload': upload_dir,
    'download': download_dir,
    'remove': delete_blob,
    'lsdir': get_blobs_list
}
def main():
    instr = sys.argv[1]
    possible_file_or_dir_name = sys.argv[2] if len(sys.argv) > 2 else None
    possible_destination = sys.argv[3] if len(sys.argv) > 3 else None
    func = instr_to_func[instr]
    #only send in non-None args
    args = [BUCKET_NAME, possible_file_or_dir_name, possible_destination]
    args = [arg for arg in args if arg is not None]
    print(f"Calling {func.__name__} with args {args}")
    func(*args)
if __name__ == "__main__":
    main()