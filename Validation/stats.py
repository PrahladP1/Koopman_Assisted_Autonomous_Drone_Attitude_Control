from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ValSummary:
    num_runs: int
    mean_err: float
    median_err: float
    std_err: float
    min_err: float
    max_err: float
    rmse: float
    success_rate: float
    success_threshold: float

def mean(values):
    return float(np.mean(values))

def median(values):
    return float(np.median(values))

def std(values):
    return float(np.std(values))

def minimum(values):
    return float(np.min(values))

def maximum(values):
    return float(np.max(values))

def rmse(values):
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2)))

def success_rate(success_flags):
    success_flags = np.asarray(success_flags)
    return float(np.mean(success_flags))

def failure_rate(success_flags):
    return 1.0-success_rate(success_flags)

def percentile(values, q):
    return float(np.percentile(values, q))

def conf_int(values, confidence=0.95):
    values = np.asarray(values)
    n = len(values)
    mu = np.mean(values)
    sigma = np.std(values)
    z = 1.96 if confidence == 0.95 else 2.58
    half_width = (z*sigma)/(np.sqrt(n))

    return mu-half_width, mu+half_width

def summarize_res(init_err, final_err, success_flags, success_threshold,):
    summary = ValSummary(num_runs=len(final_err),
                         mean_err=mean(final_err),
                         median_err=median(final_err),
                         std_err=std(final_err),
                         min_err=minimum(final_err),
                         max_err=maximum(final_err),
                         rmse=rmse(final_err),
                         success_rate=success_rate(success_flags),
                         success_threshold=success_threshold,)
    return summary

def print_summary(summary):
    print()
    print("="*60)
    print("Validation Summary")
    print("="*60)

    print(f"Runs: {summary.num_runs}")
    print()
    print(f"Mean Error: {summary.mean_err:.4f}")
    print(f"Median Error: {summary.median_err:.4f}")
    print(f"Standard Deviation: {summary.std_err:.4f}")
    print(f"RMSE: {summary.rmse:.4f}")
    print(f"Minimum Error: {summary.min_err:.4f}")
    print(f"Maximum Error: {summary.max_err:.4f}")
    print()
    print(f"Success Rate (< {summary.success_threshold:.2f} m"
          f": {100*summary.success_rate:.2f}%)")
    print("="*60)

def summary2dict(summary):
    return vars(summary)

def summary2csv(summary, filename,):
    df = pd.DataFrame([summary2dict(summary)])
    df.to_csv(filename, index=False,)