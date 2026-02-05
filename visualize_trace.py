#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from subject_analysis import Dataset

RESEARCH_DIR = Path("~/Documents/UCSF")
COHORT = "Combined/"
EXPERIMENT = "All trials"


def setup_output_dir():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def load_dataset():
    input_path = RESEARCH_DIR / "datasets" / COHORT
    ld = Dataset(
        str(input_path / "LDPatients.mat"), "LD_P50_Patients_12_5_24"
    ) + Dataset(str(input_path / "ADLDP50Controls.mat"), "AD_LD_P50_Controls_12_5_24")
    return ld


def plot_trial(subject, trial_idx, output_dir=None, show_plot=True):
    time_mask = (subject.taxis >= 0) & (subject.taxis <= 0.2)
    taxis_cropped = subject.taxis[time_mask]
    trials_cropped = subject.trials[:, time_mask]
    median_pitch = np.median(trials_cropped, axis=0)
    trial_pitch = trials_cropped[trial_idx, :]
    cents = 1200 * np.log2(trial_pitch / median_pitch)

    early_mask = (taxis_cropped >= 0) & (taxis_cropped <= 0.05)
    late_mask = (taxis_cropped >= 0.15) & (taxis_cropped <= 0.2)
    baseline_mean = np.mean(np.concatenate([cents[early_mask], cents[late_mask]]))

    plt.axvspan(0, 0.05, color="red", alpha=0.1, zorder=0)
    plt.axvspan(0.15, 0.2, color="red", alpha=0.1, zorder=0)
    plt.plot([0, 0.05], [baseline_mean, baseline_mean], color="red", linewidth=2)
    plt.plot([0.15, 0.2], [baseline_mean, baseline_mean], color="red", linewidth=2)
    plt.plot(taxis_cropped, cents, color="blue", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (cents)")
    plt.title(f"Trial {trial_idx} - {subject.name}")
    plt.tight_layout()

    if output_dir:
        filename = f"trace_{subject.name}_trial_{trial_idx}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved {filename}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize single trial pitch traces")
    parser.add_argument(
        "--subject", type=str, help="Subject name (default: first control subject)"
    )
    parser.add_argument("--trial", type=int, default=3, help="Trial index (default: 3)")
    parser.add_argument(
        "--save", action="store_true", help="Save plot to outputs directory"
    )
    parser.add_argument("--no-show", action="store_true", help="Don't display plot")

    args = parser.parse_args()

    ld = load_dataset()

    if args.subject:
        subject = None
        for cohort_name, cohort in ld.cohorts.items():
            for subj in cohort.subjects:
                if subj.name == args.subject:
                    subject = subj
                    break
            if subject:
                break
        if not subject:
            print(f"Subject {args.subject} not found")
            sys.exit(1)
    else:
        subject = ld.cohorts["Controls"].subjects[0]

    output_dir = setup_output_dir() if args.save else None

    plot_trial(subject, args.trial, output_dir, not args.no_show)


if __name__ == "__main__":
    main()
