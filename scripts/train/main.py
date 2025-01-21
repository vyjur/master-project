import os
import subprocess

### INFO: Change to "ner" or "tre" depends on what is needed.
PROCESS = "ner"

with open(f"./scripts/train/output/{PROCESS}.txt", "w") as output:
    subprocess.call(["python", f"./scripts/train/{PROCESS}.py"], stdout=output)
