import subprocess
import sys

mruby_binary = sys.argv[1]

for idx in range(1, len(sys.argv), 2):
    bin = sys.argv[idx]
    bin_bytecode = sys.argv[idx + 1]
    rb_name = bin_bytecode.split("/")[-1].split(".")[0]
    print("Benchmarking " + rb_name)
    r = subprocess.run(
        [
            "hyperfine",
            "--warmup",
            "1",
            # Calling compiled binary
            "-n",
            "lightstorm " + rb_name,
            bin,
            # Calling compiled binary (mruby bytecode)
            "-n",
            "mruby " + rb_name,
            bin_bytecode,
        ],
        capture_output=True,
    )
    summary = r.stdout.decode("utf8").split("Summary")[-1].splitlines()
    summary = [s.strip() for s in summary]
    print(" ".join(summary).strip())
