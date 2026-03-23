#!/usr/bin/env python3
"""Quick test to verify submitit works on HEC SLURM."""

import os
import sys
from submitit import AutoExecutor

print("[DEBUG] Starting test_submitit.py", flush=True)

folder = "./test_submitit"
os.makedirs(folder, exist_ok=True)
print(f"[DEBUG] Created folder: {folder}", flush=True)

print("[DEBUG] Creating executor...", flush=True)
executor = AutoExecutor(folder=folder, cluster="slurm")
executor.update_parameters(timeout_min=5, cpus_per_task=1)  # 1 core for serial partition
print("[DEBUG] Executor created", flush=True)

print("[DEBUG] Submitting test job...", flush=True)
job = executor.submit(lambda: 1 + 1)
print(f"[DEBUG] Job submitted: {job}", flush=True)

print("[DEBUG] Waiting for result...", flush=True)
result = job.result()

print(f"[SUCCESS] Result: {result}", flush=True)
