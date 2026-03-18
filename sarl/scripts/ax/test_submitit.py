# test_submitit.py
from submitit import AutoExecutor
import os

folder = "./test_submitit"
os.makedirs(folder, exist_ok=True)

executor = AutoExecutor(folder=folder, cluster="slurm")
executor.update_parameters(timeout_min=5, cpus_per_task=2)

job = executor.submit(lambda: 1 + 1)
result = job.result()
print(f"Result: {result}")
