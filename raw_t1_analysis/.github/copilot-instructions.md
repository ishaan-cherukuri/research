# AI Coding Agent Instructions for the ADNI Project

Welcome to the ADNI project! This document provides essential guidance for AI coding agents to be productive in this codebase. It outlines the architecture, workflows, conventions, and integration points specific to this project.

## Project Overview

The ADNI project is focused on processing and analyzing Alzheimer’s Disease Neuroimaging Initiative (ADNI) data. The codebase includes utilities for handling subject metadata, visualizing MRI data, and implementing machine learning workflows, with a focus on **Survival Analysis** using XGBoost.

### Key Components

- **Data Directory (`data/`)**: Contains subject metadata and JSON files organized by subject and timestamp. These files represent imaging data and associated metadata.
- **Scripts**:
  - `features.py`: Extracts features from the data.
  - `mri_visualizer.py`: Provides tools for visualizing MRI data.
  - `subject_data.py`: Handles subject metadata processing.
- **Documentation**:
  - `README.md`: High-level project description.
  - `shallow-ml.md`: Details feature engineering for survival analysis.
  - `features.md`: Documents the final flattened schema for survival modeling.

## Developer Workflows

### Running Scripts

- Use Python 3.9 or higher.
- Example: To extract features, run:
  ```bash
  python features.py
  ```

### Data Organization

- Data is stored in `data/`.
- Each subject has a unique folder (e.g., `002_S_0729/`), containing subfolders for different timestamps.
- JSON files include imaging metadata and are named with timestamps or IDs.

### Debugging

- Use print statements or Python debuggers (e.g., `pdb`) to inspect data processing.
- Example: To debug `features.py`, add:
  ```python
  import pdb; pdb.set_trace()
  ```

## Project-Specific Conventions

- **Feature Engineering**:
  - Refer to `shallow-ml.md` for a detailed feature menu, including per-scan and longitudinal features.
  - Focus on **scanner/protocol covariates**, **NIfTI header geometry**, and **MRIQC-style QC metrics**.
- **Survival Modeling**:
  - Follow the schema in `features.md` for time-to-conversion analysis.
  - Key columns include `event_observed`, `event_time_years`, and baseline scan features.
- **File Naming**: JSON files follow the pattern `<timestamp>.json` or `<subject_id>.json`.
- **Data Parsing**: Use `subject_data.py` for consistent metadata handling.
- **Visualization**: Use `mri_visualizer.py` for MRI-related tasks.

## Integration Points

- **External Dependencies**:
  - Ensure `pyproject.toml` lists all required dependencies.
  - Install dependencies using:
    ```bash
    pip install -r requirements.txt
    ```
- **Cross-Component Communication**:
  - Scripts interact via shared data in `data/`.
  - Maintain consistent data formats to avoid processing errors.

## Examples

### Adding a New Subject

1. Create a new folder in `data/` with the subject ID.
2. Add JSON files for each timestamp.
3. Update `subject_metadata.tsv` if necessary.

### Visualizing MRI Data

Run the `mri_visualizer.py` script with the appropriate arguments:

```bash
python mri_visualizer.py --input data/002_S_0729/2006-08-02_07_02_00.0/2006-08-02_07_02_00.0.json
```

### Survival Analysis Workflow

1. Extract features using `features.py`.
2. Follow the feature engineering guidelines in `shallow-ml.md`.
3. Use the schema in `features.md` to prepare the dataset.
4. Train an XGBoost survival model using the processed data.

---

For further questions or clarifications, refer to the `README.md` or contact the project maintainers.
