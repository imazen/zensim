#!/usr/bin/env python3
"""Run the feature permutations owned by CI, using its exact ordered list."""
import ast
import subprocess
from pathlib import Path

workflow = Path(".github/workflows/ci.yml").read_text()
matrix = workflow.split("features_list=(", 1)[1].split("\n          )", 1)[0]
features = [ast.literal_eval(line.strip()) for line in matrix.splitlines()
            if line.strip().startswith('"')]
assert features and features[0] == "", "CI feature matrix changed shape"
failures = []
for index, feature in enumerate(features):
    print(f"feature permutation {index + 1}/{len(features)}: {feature or 'none'}", flush=True)
    flags = ["--no-default-features"]
    if feature:
        flags += ["--features", feature]
    for command in (["cargo", "clippy", "-p", "zensim", *flags, "--lib", "--", "-D", "warnings"],
                    ["cargo", "test", "-p", "zensim", *flags, "--lib", "--", "--nocapture"]):
        print("command:", " ".join(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append((feature, command[1], result.returncode))
print(f"CI feature permutations: {len(features)}; failures: {failures}", flush=True)
raise SystemExit(bool(failures))
