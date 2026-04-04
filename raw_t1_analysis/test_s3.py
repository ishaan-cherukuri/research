
import s3fs

def test_s3_connection():
    try:
        print("Initializing S3 connection...")
        fs = s3fs.S3FileSystem(anon=False)

        bucket_name = "ishaan-research"
        print(f"\nListing contents of s3://{bucket_name}/")

        try:
            contents = fs.ls(bucket_name)
            print(f"\nFound {len(contents)} items:")
            for item in contents[:20]:
                item_type = "DIR" if fs.isdir(item) else "FILE"
                print(f"  [{item_type}] {item}")

            if len(contents) > 20:
                print(f"\n  ... and {len(contents) - 20} more items")

        except Exception as e:
            print(f"Error listing bucket: {e}")
            return False

        data_path = f"{bucket_name}/data"
        print(f"\n\nListing contents of s3://{data_path}/")

        try:
            data_contents = fs.ls(data_path)
            print(f"Found {len(data_contents)} items:")
            for item in data_contents[:10]:
                item_type = "DIR" if fs.isdir(item) else "FILE"
                print(f"  [{item_type}] {item}")

            if len(data_contents) > 10:
                print(f"\n  ... and {len(data_contents) - 10} more items")
        except Exception as e:
            print(f"Error listing {data_path}: {e}")

        print("\n S3 connection successful!")
        return True

    except Exception as e:
        print(f"\n S3 connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check AWS credentials: aws configure")
        print("2. Verify bucket access permissions")
        print("3. Install s3fs: uv add s3fs")
        return False

if __name__ == "__main__":
    test_s3_connection()
