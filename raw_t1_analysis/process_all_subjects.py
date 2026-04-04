
import pandas as pd
from features import FeatureExtractor
from datetime import datetime
import logging
from pathlib import Path

def setup_logger(output_file: str):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = Path(output_file).stem + ".log"
    log_file = log_dir / log_filename

    logger = logging.getLogger("ADNI_Processing")
    logger.setLevel(logging.INFO)

    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file

def process_all_subjects(
    metadata_file: str = "data/subject_metadata.tsv",
    output_file: str = "features_with_segmentation.csv",
    use_s3: bool = True,
    enable_segmentation: bool = True,
    segmentation_method: str = "simple",
    max_subjects: int | None = None,
):
    logger, log_file = setup_logger(output_file)

    logger.info("=" * 70)
    logger.info("ADNI Feature Extraction with Segmentation")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    logger.info("")

    logger.info(f"Loading metadata from {metadata_file}...")
    metadata_df = pd.read_csv(metadata_file, sep="\t")
    unique_subjects = metadata_df["subject"].unique()

    if max_subjects:
        unique_subjects = unique_subjects[:max_subjects]

    logger.info(f"   Found {len(unique_subjects)} subjects to process")
    logger.info(f"   S3 enabled: {use_s3}")
    logger.info(f"   Segmentation enabled: {enable_segmentation}")
    logger.info(f"   Segmentation method: {segmentation_method}")
    logger.info("")

    extractor = FeatureExtractor(
        data_dir="data",
        use_s3=use_s3,
        enable_segmentation=enable_segmentation,
        segmentation_method=segmentation_method,
    )

    all_features = []
    successful = 0
    failed = 0
    start_time = datetime.now()

    for i, subject_id in enumerate(unique_subjects, 1):
        logger.info(f"[{i}/{len(unique_subjects)}] Processing {subject_id}...")

        try:
            features = extractor.extract_subject_features(subject_id)

            if features:
                all_features.append(features)
                successful += 1

                if enable_segmentation:
                    hippo = features.get("seg_hippocampus_total_mm3_bl")
                    vent = features.get("seg_ventricles_total_mm3_bl")
                    bpf = features.get("seg_bpf_bl")
                    if hippo or vent or bpf:
                        logger.info(
                            f"   Segmentation: Hippo={hippo:.0f if hippo else 'N/A'}mm³, "
                            f"Vent={vent:.0f if vent else 'N/A'}mm³, BPF={bpf:.3f if bpf else 'N/A'}"
                        )

                event = features.get("event_observed")
                event_time = features.get("event_time_years")
                if event == 1:
                    logger.info(f"   MCI->AD conversion at {event_time:.2f} years")
                else:
                    logger.info(
                        f"   No conversion (censored at {event_time:.2f} years)"
                    )
            else:
                failed += 1
                logger.warning("   Failed to extract features")

        except Exception as e:
            failed += 1
            logger.error(f"   Error: {e}")

        if len(all_features) > 0 and len(all_features) % 50 == 0:
            df_temp = pd.DataFrame(all_features)
            temp_file = output_file.replace(".csv", f"_temp_{len(all_features)}.csv")
            df_temp.to_csv(temp_file, index=False)
            logger.info(f"\n   Saved intermediate results to {temp_file}\n")

        logger.info("")

    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_file, index=False)

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Successful: {successful}/{len(unique_subjects)}")
        logger.info(f"Failed: {failed}/{len(unique_subjects)}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Time elapsed: {elapsed / 60:.1f} minutes")
        logger.info(f"Average: {elapsed / successful:.1f} seconds per subject")

        logger.info("\nFeature Categories:")
        seg_features = [c for c in df.columns if c.startswith("seg_")]
        qc_features = [c for c in df.columns if c.startswith("qc_")]
        long_features = [c for c in df.columns if c.startswith("long_")]
        meta_features = [c for c in df.columns if c.startswith("meta_")]

        logger.info(f"   Segmentation: {len(seg_features)}")
        logger.info(f"   QC metrics: {len(qc_features)}")
        logger.info(f"   Longitudinal: {len(long_features)}")
        logger.info(f"   Metadata: {len(meta_features)}")
        logger.info(
            f"   Other: {len(df.columns) - len(seg_features) - len(qc_features) - len(long_features) - len(meta_features)}"
        )

        if "event_observed" in df.columns:
            conversions = df["event_observed"].sum()
            logger.info("\nSample Statistics:")
            logger.info(
                f"   MCI→AD conversions: {conversions}/{len(df)} ({100 * conversions / len(df):.1f}%)"
            )

            if "event_time_years" in df.columns:
                logger.info(
                    f"   Mean follow-up: {df['event_time_years'].mean():.2f} years"
                )
                if conversions > 0:
                    conversion_times = df[df["event_observed"] == 1]["event_time_years"]
                    logger.info(
                        f"   Mean conversion time: {conversion_times.mean():.2f} years"
                    )

        logger.info("=" * 70 + "\n")

        return df
    else:
        logger.error("\nNo features extracted successfully")
        return None

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process ADNI subjects with segmentation"
    )
    parser.add_argument(
        "--metadata",
        default="data/subject_metadata.tsv",
        help="Path to metadata TSV file",
    )
    parser.add_argument(
        "--output", default="features_with_segmentation.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--no-s3", action="store_true", help="Disable S3 access (use local files only)"
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable segmentation feature extraction",
    )
    parser.add_argument(
        "--method",
        choices=["simple", "synthseg"],
        default="simple",
        help="Segmentation method (default: simple)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)",
    )

    args = parser.parse_args()

    process_all_subjects(
        metadata_file=args.metadata,
        output_file=args.output,
        use_s3=not args.no_s3,
        enable_segmentation=not args.no_segmentation,
        segmentation_method=args.method,
        max_subjects=args.max_subjects,
    )

if __name__ == "__main__":
    main()
