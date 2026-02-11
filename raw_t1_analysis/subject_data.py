import os
import glob
import pandas as pd
import json
import tempfile
import s3fs

from typing import List, Optional
from pydantic import BaseModel

from nibabel.loadsave import load


class SessionMeta(BaseModel):
    visit_code: str
    image_path: Optional[str] = None
    json_metadata: Optional[dict] = None
    acq_date: Optional[str] = None
    s3_path: Optional[str] = None


class SubjectMeta(BaseModel):
    subject: str
    sessions: List[SessionMeta]
    labels: List[Optional[int]]


class SubjectData:
    def __init__(self):
        self.sessions: List[str] = []  # List of session names
        self.session_data: List[SessionMeta] = []  # List of SessionMeta
        self.labels: List[Optional[int]] = []  # Label for each session (from CSV)

    @staticmethod
    def load(
        label_csv: str,
        subject_id: str,
        json_ext: str = ".json",
        image_ext: str = ".nii.gz",
    ) -> "SubjectData":
        fs = s3fs.S3FileSystem(anon=False)
        subject = SubjectData()
        # Load label CSV with tab delimiter
        label_df = pd.read_csv(label_csv, delimiter="\t")
        # Columns: subject, visit_code, acq_date, path, diagnosis
        subject_rows = label_df[label_df["subject"] == subject_id]
        for _, row in subject_rows.iterrows():
            visit_code = str(row["visit_code"])
            acq_date = str(row["acq_date"])
            s3_path = str(row["path"])
            label = int(row["diagnosis"]) if not pd.isna(row["diagnosis"]) else None
            json_metadata = None
            image_path = None
            if s3_path.startswith("s3://"):
                try:
                    files = fs.ls(s3_path)
                    json_files = [f for f in files if f.endswith(json_ext)]
                    nii_files = [f for f in files if f.endswith(image_ext)]
                    image_path = nii_files[0] if nii_files else None
                    if json_files:
                        with fs.open(json_files[0], "r") as f:
                            try:
                                json_metadata = json.load(f)
                            except Exception as e:
                                print(f"Error parsing {json_files[0]} from S3: {e}")
                except Exception as e:
                    print(f"Error accessing S3 path {s3_path}: {e}")
            else:
                json_files = glob.glob(os.path.join(s3_path, f"*{json_ext}"))
                nii_files = glob.glob(os.path.join(s3_path, f"*{image_ext}"))
                image_path = nii_files[0] if nii_files else None
                if json_files:
                    with open(json_files[0], "r") as f:
                        try:
                            json_metadata = json.load(f)
                        except Exception as e:
                            print(f"Error parsing {json_files[0]}: {e}")
            session_meta = SessionMeta(
                visit_code=visit_code,
                image_path=image_path,
                json_metadata=json_metadata,
                acq_date=acq_date,
                s3_path=s3_path,
            )
            subject.sessions.append(visit_code)
            subject.session_data.append(session_meta)
            subject.labels.append(label)
        return subject

    def load_image(self, session_idx: int):
        """Load the NIfTI image for a session using nibabel. Returns the image object or None."""
        image_path = self.session_data[session_idx].image_path
        if image_path:
            try:
                if image_path.startswith("s3://"):
                    fs = s3fs.S3FileSystem(anon=False)
                    with fs.open(image_path, "rb") as f:
                        data = f.read()
                        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
                            if isinstance(data, str):
                                data = data.encode()
                            tmp.write(data)
                            tmp.flush()
                            img = load(tmp.name)
                else:
                    img = load(image_path)
                return img
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    data_folder = "data"
    subject_id = "002_S_0782"
    label_csv = "subject_metadata.csv"  # Path to your CSV
    subject = SubjectData.load(data_folder, subject_id, label_csv)
    print("Sessions:", subject.sessions)
    print(
        "Session data (first):",
        subject.session_data[0] if subject.session_data else None,
    )
    print("Labels:", subject.labels)
    img = subject.load_image(0)
    # Use get_fdata().shape to get the shape of the image data
    if img is not None:
        try:
            shape = img.get_fdata().shape  # type: ignore
        except Exception as e:
            print(f"Error getting image shape: {e}")
            shape = None
    else:
        shape = None
    print("Loaded image shape:", shape)
