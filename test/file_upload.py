from dotenv import load_dotenv
from pathlib import Path
from services.file_base import storage
from logger.logger import get_logger
logger = get_logger(__name__)

load_dotenv()


def main():
    logger.info("Initializing storage")
    storage.initialize()

    file_path = r"E:\Agentic_RAG\AmritaSync_A_Deep_Learning_based_Anaemia_Tracker_for_Non-Invasive_Detection.pdf"
    object_name = "documents/sample.pdf"
    download_path = Path("downloads/sample.pdf")

    try:
        storage.upload_document(
            file_name=file_path,
            object_name=object_name,
        )
        print("Upload Test: PASSED")
    except Exception as e:
        print(f"Upload Test: FAILED\n{e}")

    try:
        storage.download_document(
            object_name=object_name,
            destination=download_path,
        )
        print("Download Test: PASSED")
    except Exception as e:
        print(f"Download Test: FAILED\n{e}")

    try:
        storage.delete_document(
            object_name=object_name,
        )
        print("Delete Test: PASSED")
    except Exception as e:
        print(f"Delete Test: FAILED\n{e}")


if __name__ == "__main__":
    main()