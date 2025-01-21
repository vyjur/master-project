import subprocess

### INFO: Change to "ner" or "tre" depends on what is needed.
process = "ner"

import os

print(os.getcwd())

with open(f"(./scripts/train/{process}.txt", "w") as output:
    subprocess.call(["python", "/scripts/train/{process}.py"], stdout=output)
