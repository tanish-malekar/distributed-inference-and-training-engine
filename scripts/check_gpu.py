"""Read NVIDIA's GPU status without importing torch or allocating GPU memory."""

import argparse
import re
import subprocess
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    if args.index < 0:
        parser.error("GPU index cannot be negative")
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(args.index), "-q", "-x"], text=True
    )
    gpu = ET.fromstring(output).find("gpu")
    if gpu is None:
        raise SystemExit("Cannot read GPU status; refusing to launch.")
    utilization = gpu.findtext("utilization/gpu_util", "N/A")
    memory = gpu.findtext("fb_memory_usage/used", "N/A")
    processes = gpu.findall("processes/process_info")
    print(f"GPU {args.index}: {gpu.findtext('product_name')}")
    print(f"Utilization: {utilization} | Memory used: {memory}")
    print(f"Running GPU processes: {len(processes)}")
    for process in processes:
        print(f"  PID {process.findtext('pid')}: {process.findtext('process_name')}")
    util_match = re.fullmatch(r"\s*(\d+)\s*%\s*", utilization)
    mem_match = re.fullmatch(r"\s*(\d+)\s*MiB\s*", memory)
    # The Context3 A6000s showed 5-10 MiB of driver overhead when idle.
    # No process, 0% utilization, and at most that baseline are required.
    idle = (
        util_match is not None
        and int(util_match.group(1)) == 0
        and mem_match is not None
        and int(mem_match.group(1)) <= 10
        and not processes
    )
    if not idle:
        raise SystemExit("GPU is busy or its idle status is uncertain; refusing to launch.")
    print("GPU appears idle (allowing only 0-10 MiB of driver overhead).")


if __name__ == "__main__":
    main()
