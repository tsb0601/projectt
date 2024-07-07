from google.cloud import storage


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket('us-central2-storage')
    blob = bucket.blob(blob_name)
    generation_match_precondition = None

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to delete is aborted if the object's
    # generation number does not match your precondition.
    blob.reload()  # Fetch blob metadata to use in generation_match_precondition.
    generation_match_precondition = blob.generation

    blob.delete(if_generation_match=generation_match_precondition)

    print(f"Blob {blob_name} deleted.")
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket('us-central2-storage')

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket('us-central2-storage')
    blob = bucket.blob('boyang-ckpt/test.txt')

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )
import os
def upload_dir(bucket_name, source_dir_name, destination_blob_name):
    """Uploads a directory to a bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_dir_name = "local/path/to/directory"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    base_directory = os.path.basename(source_dir_name)
    # Iterate over the files in the directory
    for dirpath, _, filenames in os.walk(source_dir_name):
        for filename in filenames:
            # Upload the file to the destination
            destination = os.path.join(destination_blob_name, base_directory, os.path.relpath(dirpath, source_dir_name), filename)
            # merge useless dir like ./
            destination = os.path.normpath(destination)
            print(f"Uploading {filename} to {destination}")
            blob = bucket.blob(destination)
            blob.upload_from_filename(os.path.join(dirpath, filename))

            print(f"File {os.path.join(dirpath, filename)} uploaded to {destination}.")
def download_dir(bucket_name, source_blob_name, destination_dir_name):
    """Downloads a directory from a bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of your GCS object
    # source_blob_name = "storage-object-name"
    # The path to the directory to download the file to
    # destination_dir_name = "local/path/to/directory"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

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
        print(f"Downloading {blob.name} to {destination}")
        blob.download_to_filename(destination)

        print(f"File {blob.name} downloaded to {destination}.")
def main():
    #upload_dir('us-central2-storage', '/home/bytetriper/model_zoo/mae_base_256_ft', 'boyang-ckpt/')
    download_dir('us-central2-storage', 'boyang-ckpt', '/home/bytetriper/test/')
    #upload_blob('us-central2-storage', '../test.txt', 'boyang-ckpt/')
    #delete_blob('', 'boyang-ckpt/test.txt')
    #upload_blob('us-central2-storage', '../test.txt', 'boyang-ckpt/')
    #download_blob('us-central2-storage', 'boyang-ckpt/test.txt', 'test.txt')
if __name__ == "__main__":
    main()