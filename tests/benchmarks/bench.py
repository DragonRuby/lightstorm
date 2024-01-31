import subprocess
import sys

mruby_binary = sys.argv[1]

for idx in range(2, len(sys.argv), 2):
    bin = sys.argv[idx]
    rb_full = sys.argv[idx + 1]
    rb_name = rb_full.split('/')[-1]
    print("Benchmarking " + rb_name)
    r = subprocess.run([
        'hyperfine',
        '--warmup', '1',
        # Calling compiled binary
        '-n', 'lightstorm ' + rb_name, bin,
        # Calling mruby against the original file
        '-n', 'mruby ' + rb_name, mruby_binary + ' ' + rb_full
    ], capture_output=True)
    summary = r.stdout.decode('utf8').split("Summary")[-1].splitlines()
    summary = [s.strip() for s in summary]
    print(" ".join(summary))
