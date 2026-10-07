# TabPFN-HRV-Sleep-Apnea

HRV-based sleep apnea classification using TabPFN under low-sample and cross-database settings.

## Overview

This repository contains the code, analysis scripts, configuration files, and derived summary outputs used in a study of ECG-derived heart rate variability (HRV) and sleep apnea classification with TabPFN.

The project evaluates patient-independent development performance, low-sample label efficiency, and multi-center transfer across PhysioNet-based cohorts.

## Data access

The original physiological recordings are **not redistributed** in this repository.

Source datasets were accessed from PhysioNet and remain subject to their own licenses, access conditions, and citation requirements. Users must obtain the original datasets directly from the source providers and comply with all applicable terms.

## TabPFN and third-party components

The MIT License in this repository applies only to the original code authored for this project. It does **not** apply to:

- PhysioNet source datasets
- TabPFN model weights or checkpoints
- Third-party libraries, frameworks, or external tools
- Any material whose license is separately defined by its original authors

Users are responsible for reviewing and complying with the licenses and usage terms of external dependencies.

## Reproducibility

To reproduce the analyses:

1. Obtain the relevant source datasets from PhysioNet.
2. Install the Python dependencies used by the analysis scripts.
3. Obtain TabPFN access or a checkpoint under the provider’s current terms.
4. Run the study pipelines as described in the manuscript.

## License

The original source code in this repository is licensed under the MIT License.

This license does not apply to PhysioNet source datasets, third-party software, or TabPFN model weights. Those resources remain subject to their respective licenses and terms of use.
