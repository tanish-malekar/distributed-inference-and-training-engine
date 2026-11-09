"""Read NVIDIA's GPU status without importing torch or allocating GPU memory."""

import argparse
import re
import subprocess
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indices", help="One GPU index or a comma-separated list, e.g. 0,1")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+(,\d+)*", args.indices):
        parser.error("Give GPU indices as digits separated by commas, e.g. 0,1")
    indices = [int(value) for value in args.indices.split(",")]
    if len(set(indices)) != len(indices):
        parser.error("A GPU index appears more than once")
    # Every listed GPU must be idle; a single busy or unreadable one refuses the whole job.
    for index in indices:
        check(index)
    print(f"All {len(indices)} requested GPU(s) appear idle.")


def check(index: int) -> None:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "-q", "-x"], text=True
    )
    gpu = ET.fromstring(output).find("gpu")
    if gpu is None:
        raise SystemExit(f"Cannot read GPU {index} status; refusing to launch.")
    utilization = gpu.findtext("utilization/gpu_util", "N/A")
    memory = gpu.findtext("fb_memory_usage/used", "N/A")
    processes = gpu.findall("processes/process_info")
    print(f"GPU {index}: {gpu.findtext('product_name')}")
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
        raise SystemExit(f"GPU {index} is busy or its idle status is uncertain; refusing to launch.")
    print(f"GPU {index} appears idle (allowing only 0-10 MiB of driver overhead).")


if __name__ == "__main__":
    main()
