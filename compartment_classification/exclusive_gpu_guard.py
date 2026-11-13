"""Pause only our exporter if another compute process enters its GPU."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--process-record", type=Path, required=True)
    ap.add_argument("--complete", type=Path, required=True)
    args = ap.parse_args()
    record = json.loads(args.process_record.read_text())
    pid, gpu = record["pid"], record["cuda_visible_devices"]
    paused = False
    events = []
    destination = args.process_record.parent / "gpu_exclusive_guard.json"
    while True:
        try:
            process_file = Path(f"/proc/{pid}/cmdline")
            if not process_file.exists() or b"fusion50/refine/export.py" not in process_file.read_bytes():
                break
            if args.complete.exists():
                break
            raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]).decode()
            others = []
            for row in raw.splitlines():
                uuid, other = [x.strip() for x in row.split(",")]
                if uuid == gpu and int(other) != pid:
                    others.append(int(other))
            if others and not paused:
                os.kill(pid, signal.SIGSTOP)
                paused = True
                events.append(dict(action="pause_our_exporter", other_pids=others, time=time.time()))
                print(json.dumps(events[-1]), flush=True)
            elif not others and paused:
                os.kill(pid, signal.SIGCONT)
                paused = False
                events.append(dict(action="resume_our_exporter", time=time.time()))
                print(json.dumps(events[-1]), flush=True)
            destination.write_text(json.dumps(dict(exporter_pid=pid, gpu=gpu, paused=paused, events=events), indent=2) + "\n")
        except ProcessLookupError:
            break
        time.sleep(2)
    destination.write_text(json.dumps(dict(exporter_pid=pid, gpu=gpu, paused=paused, events=events,
                                          finished_unix=time.time(), export_complete=args.complete.exists()), indent=2) + "\n")


if __name__ == "__main__":
    main()
