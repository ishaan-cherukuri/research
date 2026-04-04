
from features import FeatureExtractor
import json

extractor = FeatureExtractor(data_dir="data", use_s3=False)

subject_id = "002_S_0729"
print(f"Testing feature extraction for {subject_id}...")
features = extractor.extract_subject_features(subject_id)

if features:
    print(f"\n Successfully extracted features for {subject_id}")
    print(f"\nTotal features: {len(features)}")

    print("\n=== Survival Labels ===")
    print(f"  baseline_diagnosis: {features.get('baseline_diagnosis')}")
    print(f"  event_observed: {features.get('event_observed')}")
    print(f"  event_time_years: {features.get('event_time_years')}")
    print(f"  mci_bl_datetime: {features.get('mci_bl_datetime')}")
    print(f"  event_datetime: {features.get('event_datetime')}")

    print("\n=== Cohort Features ===")
    print(f"  n_visits_total: {features.get('n_visits_total')}")
    print(f"  n_visits_used: {features.get('n_visits_used')}")
    print(f"  followup_years: {features.get('followup_years')}")

    print("\n=== Longitudinal Features ===")
    print(f"  long_n_scans: {features.get('long_n_scans')}")
    print(f"  long_brain_vol_slope_yr: {features.get('long_brain_vol_slope_yr')}")
    print(f"  long_brain_vol_pctchg: {features.get('long_brain_vol_pctchg')}")
    print(f"  long_snr_slope_yr: {features.get('long_snr_slope_yr')}")

    print("\n=== All features ===")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value}")
else:
    print(f" Failed to extract features for {subject_id}")
