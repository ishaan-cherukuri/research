
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import nibabel as nib
import s3fs
import tempfile
from segmentation import BrainSegmenter

class FeatureExtractor:

    def __init__(
        self,
        data_dir: str = "data",
        use_s3: bool = True,
        enable_segmentation: bool = True,
        segmentation_method: str = "simple",
    ):
        self.data_dir = Path(data_dir)
        self.metadata_file = self.data_dir / "subject_metadata.tsv"
        self.use_s3 = use_s3
        self.fs = s3fs.S3FileSystem(anon=False) if use_s3 else None
        self.enable_segmentation = enable_segmentation
        self.segmenter = (
            BrainSegmenter(method=segmentation_method, use_s3=use_s3)
            if enable_segmentation
            else None
        )

    def extract_subject_features(self, subject_id: str) -> Optional[Dict]:
        if not self.metadata_file.exists():
            print(f"Metadata file {self.metadata_file} not found")
            return None

        metadata_df = pd.read_csv(self.metadata_file, sep="\t")
        subject_metadata = metadata_df[metadata_df["subject"] == subject_id]

        if subject_metadata.empty:
            print(f"No metadata found for {subject_id}")
            return None

        sessions = self._get_sorted_sessions_from_metadata(subject_metadata)
        if not sessions:
            print(f"No sessions found for {subject_id}")
            return None

        features = {"subject_id": subject_id}

        baseline_session = self._get_baseline_mci_session(subject_id, sessions)
        if not baseline_session:
            print(f"No baseline MCI scan found for {subject_id}")
            return None

        survival_features = self._extract_survival_labels(
            subject_id, sessions, baseline_session
        )
        features.update(survival_features)

        cohort_features = self._extract_cohort_features(
            subject_id,
            sessions,
            baseline_session,
            survival_features.get("event_datetime"),
        )
        features.update(cohort_features)

        baseline_features = self._extract_baseline_features(baseline_session)
        features.update(baseline_features)

        longitudinal_features = self._extract_longitudinal_features(
            sessions, baseline_session, survival_features.get("censor_datetime")
        )
        features.update(longitudinal_features)

        return features

    def _get_sorted_sessions_from_metadata(
        self, subject_metadata: pd.DataFrame
    ) -> List[Dict]:
        sessions = []

        for _, row in subject_metadata.iterrows():
            acq_date = row["acq_date"]
            s3_path = row["path"]
            diagnosis = row["diagnosis"]

            try:
                session_datetime = datetime.strptime(acq_date, "%Y-%m-%d")
            except Exception:
                print(f"Could not parse date: {acq_date}")
                continue

            if s3_path.startswith("s3://") and self.fs is not None:
                dir_path = "/".join(s3_path.split("/")[:-1])
                print(f"Loading from S3: {s3_path}")

                json_path = None
                metadata = {}
                nii_path = s3_path

                try:
                    files = self.fs.ls(dir_path)
                    json_files = [f for f in files if f.endswith(".json")]

                    if json_files:
                        json_path = (
                            f"s3://{json_files[0]}"
                            if not json_files[0].startswith("s3://")
                            else json_files[0]
                        )
                        with self.fs.open(json_path, "r") as f:
                            metadata = json.load(f)
                        print(f"  Loaded JSON from S3: {json_path}")
                except Exception as e:
                    print(f"Error accessing S3 path {dir_path}: {e}")
            else:
                dir_path = Path(s3_path).parent
                json_files = list(dir_path.glob("*.json"))
                nii_path = s3_path

                metadata = {}
                if json_files:
                    json_path = json_files[0]
                    with open(json_path, "r") as f:
                        metadata = json.load(f)
                else:
                    json_path = None

            sessions.append(
                {
                    "datetime": session_datetime,
                    "json_path": json_path,
                    "nii_path": nii_path,
                    "metadata": metadata,
                    "diagnosis": diagnosis,
                    "is_s3": s3_path.startswith("s3://"),
                }
            )

        sessions.sort(key=lambda x: x["datetime"])

        sessions_by_date = {}
        for session in sessions:
            date_key = session["datetime"].date()
            if (
                date_key not in sessions_by_date
                or session["datetime"] > sessions_by_date[date_key]["datetime"]
            ):
                sessions_by_date[date_key] = session

        filtered_sessions = list(sessions_by_date.values())
        filtered_sessions.sort(key=lambda x: x["datetime"])
        return filtered_sessions

    def _parse_datetime_from_foldername(self, folder_name: str) -> datetime:
        datetime_str = folder_name.replace("_", " ").replace("-", " ")
        parts = datetime_str.split()

        try:
            if len(parts) >= 6:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                hour, minute = int(parts[3]), int(parts[4])
                second = int(float(parts[5]))
                return datetime(year, month, day, hour, minute, second)
        except Exception:
            pass

        return datetime.now()

    def _get_baseline_mci_session(
        self, subject_id: str, sessions: List[Dict]
    ) -> Optional[Dict]:
        mci_sessions = [s for s in sessions if s.get("diagnosis") == 2.0]

        if not mci_sessions:
            print(f"  No MCI scans found for {subject_id}")
            return None

        return mci_sessions[0]

    def _extract_survival_labels(
        self, subject_id: str, sessions: List[Dict], baseline_session: Dict
    ) -> Dict:
        features = {}

        baseline_datetime = baseline_session["datetime"]
        baseline_diagnosis = baseline_session.get("diagnosis")
        features["mci_bl_datetime"] = baseline_datetime.isoformat()
        features["baseline_diagnosis"] = baseline_diagnosis

        ad_sessions = [
            s
            for s in sessions
            if s.get("diagnosis") == 3.0 and s["datetime"] > baseline_datetime
        ]

        if ad_sessions:
            features["event_observed"] = 1
            first_ad_datetime = ad_sessions[0]["datetime"]
            features["event_datetime"] = first_ad_datetime.isoformat()
            features["censor_datetime"] = first_ad_datetime.isoformat()

            time_delta = (first_ad_datetime - baseline_datetime).total_seconds() / (
                365.25 * 24 * 3600
            )
            features["event_time_years"] = round(time_delta, 3)
        else:
            features["event_observed"] = 0
            features["event_datetime"] = None
            last_scan_datetime = sessions[-1]["datetime"]
            features["censor_datetime"] = last_scan_datetime.isoformat()

            time_delta = (last_scan_datetime - baseline_datetime).total_seconds() / (
                365.25 * 24 * 3600
            )
            features["event_time_years"] = round(time_delta, 3)

        features["aft_y_lower"] = features["event_time_years"]
        features["aft_y_upper"] = (
            features["event_time_years"]
            if features["event_observed"] == 1
            else float("inf")
        )

        return features

    def _extract_cohort_features(
        self,
        subject_id: str,
        sessions: List[Dict],
        baseline_session: Dict,
        event_datetime: Optional[str],
    ) -> Dict:
        features = {}

        features["n_visits_total"] = len(sessions)

        censor_datetime = (
            datetime.fromisoformat(event_datetime)
            if event_datetime
            else sessions[-1]["datetime"]
        )
        visits_used = [s for s in sessions if s["datetime"] <= censor_datetime]
        features["n_visits_used"] = len(visits_used)

        baseline_datetime = baseline_session["datetime"]
        last_datetime = sessions[-1]["datetime"]
        followup_years = (last_datetime - baseline_datetime).total_seconds() / (
            365.25 * 24 * 3600
        )
        features["followup_years"] = round(followup_years, 3)

        manufacturers = [
            s["metadata"].get("Manufacturer", "Unknown")
            for s in visits_used
            if s["metadata"]
        ]
        models = [
            s["metadata"].get("ManufacturersModelName", "Unknown")
            for s in visits_used
            if s["metadata"]
        ]
        field_strengths = [
            s["metadata"].get("MagneticFieldStrength", 0)
            for s in visits_used
            if s["metadata"]
        ]

        features["manufacturer_mode"] = (
            self._get_mode(manufacturers) if manufacturers else "Unknown"
        )
        features["model_mode"] = self._get_mode(models) if models else "Unknown"
        features["field_strength_mode_t"] = (
            self._get_mode(field_strengths) if field_strengths else 0
        )
        features["site_mode"] = "Unknown"

        return features

    def _get_mode(self, values: List) -> Any:
        if not values:
            return None
        return max(set(values), key=values.count)

    def _load_nifti(self, nii_path, is_s3: bool = False):
        if is_s3 and self.fs is not None:
            print(f"  Loading NIfTI from S3: {nii_path}")
            with self.fs.open(nii_path, "rb") as f:
                data = f.read()

            if isinstance(data, str):
                data = data.encode()

            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(data)
                tmp.flush()
                img = nib.load(tmp.name)
            print("  Successfully loaded NIfTI from S3")
            return img
        else:
            print(f"  Loading NIfTI from local: {nii_path}")
            return nib.load(str(nii_path))

    def _extract_baseline_features(self, baseline_session: Dict) -> Dict:
        features = {}
        metadata = baseline_session["metadata"]

        features["meta_field_strength_t_bl"] = metadata.get("MagneticFieldStrength")
        features["meta_tr_s_bl"] = metadata.get("RepetitionTime")
        features["meta_te_s_bl"] = metadata.get("EchoTime")
        features["meta_ti_s_bl"] = metadata.get("InversionTime")
        features["meta_flip_angle_deg_bl"] = metadata.get("FlipAngle")

        if baseline_session["nii_path"]:
            is_s3 = baseline_session.get("is_s3", False)
            nii_features = self._extract_nifti_features(
                baseline_session["nii_path"], is_s3=is_s3
            )
            features.update(nii_features)

            qc_features = self._extract_qc_features(
                baseline_session["nii_path"], is_s3=is_s3
            )
            features.update(qc_features)

            if self.enable_segmentation and self.segmenter:
                try:
                    seg_features = self._extract_segmentation_features(
                        baseline_session["nii_path"], is_s3=is_s3, suffix="_bl"
                    )
                    features.update(seg_features)
                except Exception as e:
                    print(f"  ️  Segmentation failed for baseline: {e}")

        return features

    def _extract_nifti_features(
        self, nii_path, prefix: str = "hdr_", suffix: str = "_bl", is_s3: bool = False
    ) -> Dict:
        features = {}

        try:
            img = self._load_nifti(nii_path, is_s3)
            header = img.header

            shape = img.shape
            features[f"{prefix}dim_x{suffix}"] = (
                int(shape[0]) if len(shape) > 0 else None
            )
            features[f"{prefix}dim_y{suffix}"] = (
                int(shape[1]) if len(shape) > 1 else None
            )
            features[f"{prefix}dim_z{suffix}"] = (
                int(shape[2]) if len(shape) > 2 else None
            )

            pixdim = header.get_zooms()
            features[f"{prefix}vox_x_mm{suffix}"] = (
                float(pixdim[0]) if len(pixdim) > 0 else None
            )
            features[f"{prefix}vox_y_mm{suffix}"] = (
                float(pixdim[1]) if len(pixdim) > 1 else None
            )
            features[f"{prefix}vox_z_mm{suffix}"] = (
                float(pixdim[2]) if len(pixdim) > 2 else None
            )

            if len(pixdim) >= 3:
                vox_vol = pixdim[0] * pixdim[1] * pixdim[2]
                features[f"{prefix}voxvol_mm3{suffix}"] = float(vox_vol)

                if len(shape) >= 3:
                    features[f"{prefix}fov_x_mm{suffix}"] = float(shape[0] * pixdim[0])
                    features[f"{prefix}fov_y_mm{suffix}"] = float(shape[1] * pixdim[1])
                    features[f"{prefix}fov_z_mm{suffix}"] = float(shape[2] * pixdim[2])

        except Exception as e:
            print(f"Error extracting NIfTI features from {nii_path}: {e}")

        return features

    def _extract_qc_features(
        self, nii_path, prefix: str = "qc_", suffix: str = "_bl", is_s3: bool = False
    ) -> Dict:
        features = {}

        try:
            img = self._load_nifti(nii_path, is_s3)
            data = img.get_fdata()

            brain_mask = data > data.mean() * 0.1
            brain_voxels = data[brain_mask]
            bg_mask = ~brain_mask
            bg_voxels = data[bg_mask]

            voxel_vol = np.prod(img.header.get_zooms()[:3])
            features[f"{prefix}brain_mask_vol_mm3{suffix}"] = float(
                brain_mask.sum() * voxel_vol
            )

            if len(brain_voxels) > 0:
                features[f"{prefix}brain_mean{suffix}"] = float(brain_voxels.mean())
                features[f"{prefix}brain_std{suffix}"] = float(brain_voxels.std())
                features[f"{prefix}brain_p01{suffix}"] = float(
                    np.percentile(brain_voxels, 1)
                )
                features[f"{prefix}brain_p50{suffix}"] = float(
                    np.percentile(brain_voxels, 50)
                )
                features[f"{prefix}brain_p99{suffix}"] = float(
                    np.percentile(brain_voxels, 99)
                )

            if len(bg_voxels) > 0:
                features[f"{prefix}bg_mean{suffix}"] = float(bg_voxels.mean())
                features[f"{prefix}bg_std{suffix}"] = float(bg_voxels.std())

            if len(brain_voxels) > 0 and len(bg_voxels) > 0 and bg_voxels.mean() > 0:
                features[f"{prefix}brain_bg_ratio{suffix}"] = float(
                    brain_voxels.mean() / bg_voxels.mean()
                )

            if len(brain_voxels) > 0 and len(bg_voxels) > 0 and bg_voxels.std() > 0:
                features[f"{prefix}snr{suffix}"] = float(
                    brain_voxels.mean() / bg_voxels.std()
                )

        except Exception as e:
            print(f"Error extracting QC features from {nii_path}: {e}")

        return features

    def _extract_segmentation_features(
        self, nii_path: str, is_s3: bool = False, suffix: str = "_bl"
    ) -> Dict:
        features = {}

        if not self.segmenter:
            return features

        try:
            seg_features = self.segmenter.segment(nii_path, is_s3=is_s3)

            for key, value in seg_features.items():
                features[f"{key}{suffix}"] = value

        except Exception as e:
            print(f"  Error in segmentation: {e}")

        return features

    def _extract_longitudinal_features(
        self,
        sessions: List[Dict],
        baseline_session: Dict,
        censor_datetime: Optional[str],
    ) -> Dict:
        features = {}

        if censor_datetime:
            censor_dt = datetime.fromisoformat(censor_datetime)
        else:
            censor_dt = sessions[-1]["datetime"] if sessions else datetime.now()

        used_sessions = [s for s in sessions if s["datetime"] <= censor_dt]

        if len(used_sessions) < 2:
            return self._get_empty_longitudinal_features()

        baseline_dt = baseline_session["datetime"]

        qc_timeseries = []
        for session in used_sessions:
            if not session.get("nii_path"):
                continue

            try:
                is_s3 = session.get("is_s3", False)
                qc_features = self._extract_qc_features(
                    session["nii_path"], prefix="qc_", suffix="", is_s3=is_s3
                )

                time_delta = (session["datetime"] - baseline_dt).total_seconds() / (
                    365.25 * 24 * 3600
                )

                qc_timeseries.append(
                    {
                        "time_years": time_delta,
                        "brain_mask_vol_mm3": qc_features.get("qc_brain_mask_vol_mm3"),
                        "brain_mean": qc_features.get("qc_brain_mean"),
                        "brain_std": qc_features.get("qc_brain_std"),
                        "snr": qc_features.get("qc_snr"),
                        "brain_bg_ratio": qc_features.get("qc_brain_bg_ratio"),
                    }
                )
            except Exception as e:
                print(
                    f"  Error extracting longitudinal features for session {session['datetime']}: {e}"
                )
                continue

        features.update(
            self._compute_longitudinal_stats(
                qc_timeseries, "brain_mask_vol_mm3", "long_brain_vol"
            )
        )
        features.update(
            self._compute_longitudinal_stats(
                qc_timeseries, "brain_mean", "long_brain_intensity"
            )
        )
        features.update(
            self._compute_longitudinal_stats(qc_timeseries, "snr", "long_snr")
        )
        features.update(
            self._compute_longitudinal_stats(
                qc_timeseries, "brain_bg_ratio", "long_brain_bg_ratio"
            )
        )

        features["long_n_scans"] = len(qc_timeseries)

        return features

    def _get_empty_longitudinal_features(self) -> Dict:
        features = {}
        biomarkers = ["brain_vol", "brain_intensity", "snr", "brain_bg_ratio"]

        for biomarker in biomarkers:
            features[f"long_{biomarker}_last"] = None
            features[f"long_{biomarker}_delta"] = None
            features[f"long_{biomarker}_pctchg"] = None
            features[f"long_{biomarker}_slope_yr"] = None
            features[f"long_{biomarker}_mean"] = None
            features[f"long_{biomarker}_std"] = None

        features["long_n_scans"] = 0
        return features

    def _compute_longitudinal_stats(
        self, timeseries: List[Dict], value_key: str, output_prefix: str
    ) -> Dict:
        features = {}

        valid_points = [
            (tp["time_years"], tp[value_key])
            for tp in timeseries
            if tp.get(value_key) is not None
        ]

        if len(valid_points) < 2:
            features[f"{output_prefix}_last"] = None
            features[f"{output_prefix}_delta"] = None
            features[f"{output_prefix}_pctchg"] = None
            features[f"{output_prefix}_slope_yr"] = None
            features[f"{output_prefix}_mean"] = None
            features[f"{output_prefix}_std"] = None
            return features

        times = np.array([p[0] for p in valid_points])
        values = np.array([p[1] for p in valid_points])

        features[f"{output_prefix}_last"] = float(values[-1])

        features[f"{output_prefix}_delta"] = float(values[-1] - values[0])

        if values[0] != 0:
            features[f"{output_prefix}_pctchg"] = float(
                100 * (values[-1] - values[0]) / values[0]
            )
        else:
            features[f"{output_prefix}_pctchg"] = None

        if len(valid_points) >= 2:
            slope, intercept = np.polyfit(times, values, 1)
            features[f"{output_prefix}_slope_yr"] = float(slope)
        else:
            features[f"{output_prefix}_slope_yr"] = None

        features[f"{output_prefix}_mean"] = float(values.mean())
        features[f"{output_prefix}_std"] = float(values.std())

        return features

    def extract_all_subjects(self, output_file: str = "features.csv") -> pd.DataFrame:
        all_features = []

        if not self.metadata_file.exists():
            print(f"Metadata file {self.metadata_file} not found")
            return pd.DataFrame()

        metadata_df = pd.read_csv(self.metadata_file, sep="\t")
        unique_subjects = metadata_df["subject"].unique()

        print(f"Found {len(unique_subjects)} subjects in metadata")

        for subject_id in unique_subjects:
            print(f"Processing {subject_id}...")

            features = self.extract_subject_features(subject_id)
            if features:
                all_features.append(features)

        df = pd.DataFrame(all_features)

        df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")

        return df

def main():
    extractor = FeatureExtractor(data_dir="data")

    subject_id = "002_S_0729"
    features = extractor.extract_subject_features(subject_id)

    if features:
        print(f"\nFeatures for {subject_id}:")
        for key, value in features.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("Extracting features for all subjects...")
    df = extractor.extract_all_subjects()
    print(f"\nExtracted {len(df)} subjects")
    print(f"\nFeature columns ({len(df.columns)}):")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()
