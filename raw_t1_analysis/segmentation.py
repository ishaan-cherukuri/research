
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional
import subprocess
import tempfile
import json

class BrainSegmenter:

    def __init__(self, method: str = "synthseg", use_s3: bool = False):
        self.method = method
        self.use_s3 = use_s3
        if use_s3:
            try:
                import s3fs

                self.fs = s3fs.S3FileSystem(anon=False)
            except ImportError:
                print("️  s3fs not installed. Install with: uv add s3fs")
                self.fs = None
        else:
            self.fs = None
        """Check if required dependencies are available."""
        if self.method == "synthseg":
            try:
                result = subprocess.run(
                    ["mri_synthseg", "--help"], capture_output=True, timeout=5
                )
                self.synthseg_available = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self.synthseg_available = False
                print("️  SynthSeg not found. Install with: uv add freesurfer")
                print("   Falling back to simple segmentation method.")
                self.method = "simple"

    def segment(
        self, nii_path: str, output_dir: Optional[str] = None, is_s3: bool = False
    ) -> Dict:
        if self.method == "synthseg" and self.synthseg_available:
            return self._segment_synthseg(nii_path, output_dir, is_s3)
        else:
            return self._segment_simple(nii_path, is_s3)

    def _segment_synthseg(
        self, nii_path: str, output_dir: Optional[str] = None, is_s3: bool = False
    ) -> Dict:
        features = {}

        try:
            if is_s3 and self.fs:
                with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                    with self.fs.open(nii_path, "rb") as f:
                        tmp.write(f.read())
                    input_file = tmp.name
            else:
                input_file = nii_path

            with tempfile.TemporaryDirectory() as tmpdir:
                seg_file = Path(tmpdir) / "seg.nii.gz"
                vol_file = Path(tmpdir) / "volumes.csv"
                qc_file = Path(tmpdir) / "qc.csv"

                cmd = [
                    "mri_synthseg",
                    "--i",
                    str(input_file),
                    "--o",
                    str(seg_file),
                    "--vol",
                    str(vol_file),
                    "--qc",
                    str(qc_file),
                    "--robust",
                ]

                print(f"  Running SynthSeg on {Path(nii_path).name}...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode != 0:
                    print(f"  ️  SynthSeg failed: {result.stderr}")
                    return self._segment_simple(nii_path)

                if vol_file.exists():
                    features = self._parse_synthseg_volumes(vol_file)

                if qc_file.exists():
                    qc_features = self._parse_synthseg_qc(qc_file)
                    features.update(qc_features)

                if output_dir and seg_file.exists():
                    output_path = Path(output_dir) / Path(nii_path).name.replace(
                        ".nii", "_seg.nii"
                    )
                    subprocess.run(["cp", str(seg_file), str(output_path)])

                print(f"   SynthSeg completed: {len(features)} features extracted")

        except Exception as e:
            print(f"  ️  SynthSeg error: {e}")
            return self._segment_simple(nii_path)

        return features

    def _parse_synthseg_volumes(self, vol_file: Path) -> Dict:
        features = {}

        try:
            with open(vol_file, "r") as f:
                lines = f.readlines()

            if len(lines) >= 2:
                headers = lines[0].strip().split(",")
                values = lines[1].strip().split(",")

                volume_map = {
                    "Left-Hippocampus": "seg_hippocampus_l_mm3",
                    "Right-Hippocampus": "seg_hippocampus_r_mm3",
                    "Left-Amygdala": "seg_amygdala_l_mm3",
                    "Right-Amygdala": "seg_amygdala_r_mm3",
                    "Left-Lateral-Ventricle": "seg_ventricle_lat_l_mm3",
                    "Right-Lateral-Ventricle": "seg_ventricle_lat_r_mm3",
                    "3rd-Ventricle": "seg_ventricle_3rd_mm3",
                    "4th-Ventricle": "seg_ventricle_4th_mm3",
                    "Left-Cerebral-White-Matter": "seg_wm_l_mm3",
                    "Right-Cerebral-White-Matter": "seg_wm_r_mm3",
                    "Left-Cerebral-Cortex": "seg_gm_l_mm3",
                    "Right-Cerebral-Cortex": "seg_gm_r_mm3",
                    "CSF": "seg_csf_mm3",
                }

                for i, header in enumerate(headers):
                    if header in volume_map and i < len(values):
                        try:
                            features[volume_map[header]] = float(values[i])
                        except ValueError:
                            features[volume_map[header]] = None

                if (
                    "seg_hippocampus_l_mm3" in features
                    and "seg_hippocampus_r_mm3" in features
                ):
                    l = features["seg_hippocampus_l_mm3"]
                    r = features["seg_hippocampus_r_mm3"]
                    if l is not None and r is not None:
                        features["seg_hippocampus_total_mm3"] = l + r
                        if l + r > 0:
                            features["seg_hippocampus_asym"] = abs(l - r) / (l + r)

                vent_keys = [
                    "seg_ventricle_lat_l_mm3",
                    "seg_ventricle_lat_r_mm3",
                    "seg_ventricle_3rd_mm3",
                    "seg_ventricle_4th_mm3",
                ]
                vent_vols = [features.get(k) for k in vent_keys]
                if all(v is not None for v in vent_vols):
                    features["seg_ventricles_total_mm3"] = sum(vent_vols)

                if "seg_gm_l_mm3" in features and "seg_gm_r_mm3" in features:
                    l = features["seg_gm_l_mm3"]
                    r = features["seg_gm_r_mm3"]
                    if l is not None and r is not None:
                        features["seg_gm_total_mm3"] = l + r

                if "seg_wm_l_mm3" in features and "seg_wm_r_mm3" in features:
                    l = features["seg_wm_l_mm3"]
                    r = features["seg_wm_r_mm3"]
                    if l is not None and r is not None:
                        features["seg_wm_total_mm3"] = l + r

                gm = features.get("seg_gm_total_mm3")
                wm = features.get("seg_wm_total_mm3")
                csf = features.get("seg_csf_mm3")
                vent = features.get("seg_ventricles_total_mm3")

                if all(v is not None for v in [gm, wm, csf]):
                    features["seg_tiv_mm3"] = gm + wm + csf
                    features["seg_brain_mm3"] = gm + wm

                    tiv = features["seg_tiv_mm3"]
                    if tiv > 0:
                        features["seg_bpf"] = (gm + wm) / tiv

                        if "seg_hippocampus_total_mm3" in features:
                            features["seg_hippocampus_norm"] = (
                                features["seg_hippocampus_total_mm3"] / tiv
                            )
                        if vent is not None:
                            features["seg_ventricles_norm"] = vent / tiv

        except Exception as e:
            print(f"  ️  Error parsing SynthSeg volumes: {e}")

        return features

    def _parse_synthseg_qc(self, qc_file: Path) -> Dict:
        features = {}

        try:
            with open(qc_file, "r") as f:
                lines = f.readlines()

            if len(lines) >= 2:
                headers = lines[0].strip().split(",")
                values = lines[1].strip().split(",")

                for i, header in enumerate(headers):
                    if i < len(values):
                        try:
                            features[f"seg_qc_{header.lower()}"] = float(values[i])
                        except ValueError:
                            features[f"seg_qc_{header.lower()}"] = None

        except Exception as e:
            print(f"  ️  Error parsing SynthSeg QC: {e}")

        return features

    def _segment_simple(self, nii_path: str, is_s3: bool = False) -> Dict:
        features = {}

        try:
            print(f"  Using simple segmentation for {Path(nii_path).name}...")

            if is_s3 and self.fs:
                with self.fs.open(nii_path, "rb") as f:
                    data = f.read()
                with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    img = nib.load(tmp.name)
            else:
                img = nib.load(str(nii_path))

            data = img.get_fdata()
            voxel_vol = np.prod(img.header.get_zooms()[:3])

            brain_mask = data > (data.mean() * 0.1)
            brain_voxels = data[brain_mask]

            if len(brain_voxels) > 0:
                p25, p50, p75, p90 = np.percentile(brain_voxels, [25, 50, 75, 90])

                csf_mask = (data > p25 * 0.5) & (data < p50)
                gm_mask = (data >= p50) & (data < p75)
                wm_mask = data >= p75

                features["seg_csf_mm3"] = float(csf_mask.sum() * voxel_vol)
                features["seg_gm_total_mm3"] = float(gm_mask.sum() * voxel_vol)
                features["seg_wm_total_mm3"] = float(wm_mask.sum() * voxel_vol)

                brain_vol = features["seg_gm_total_mm3"] + features["seg_wm_total_mm3"]
                tiv = brain_vol + features["seg_csf_mm3"]

                features["seg_brain_mm3"] = brain_vol
                features["seg_tiv_mm3"] = tiv

                if tiv > 0:
                    features["seg_bpf"] = brain_vol / tiv

                ventricle_mask = brain_mask & (data < p25)
                features["seg_ventricles_total_mm3"] = float(
                    ventricle_mask.sum() * voxel_vol
                )

                if tiv > 0:
                    features["seg_ventricles_norm"] = (
                        features["seg_ventricles_total_mm3"] / tiv
                    )

                print(
                    f"   Simple segmentation: brain={brain_vol:.0f}mm³, TIV={tiv:.0f}mm³, BPF={features['seg_bpf']:.3f}"
                )

        except Exception as e:
            print(f"  ️  Simple segmentation error: {e}")

        return features

def test_segmentation():
    data_dir = Path("data")
    nii_files = list(data_dir.glob("**/*.nii.gz"))

    if not nii_files:
        print("No .nii.gz files found in data directory")
        return

    test_file = nii_files[0]
    print(f"\n=== Testing segmentation on {test_file.name} ===\n")

    print("1. Testing simple segmentation method:")
    segmenter_simple = BrainSegmenter(method="simple")
    features_simple = segmenter_simple.segment(str(test_file))

    print(f"\nExtracted {len(features_simple)} features:")
    for key, value in sorted(features_simple.items()):
        if value is not None:
            print(f"  {key}: {value:.2f}")

    print("\n2. Testing SynthSeg method:")
    segmenter_synth = BrainSegmenter(method="synthseg")
    if segmenter_synth.synthseg_available:
        features_synth = segmenter_synth.segment(str(test_file))
        print(f"\nExtracted {len(features_synth)} features:")
        for key, value in sorted(features_synth.items()):
            if value is not None:
                print(f"  {key}: {value:.2f}")
    else:
        print("SynthSeg not available. To install:")
        print("  uv add freesurfer")

if __name__ == "__main__":
    test_segmentation()
