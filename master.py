#!/usr/bin/env python3
import subprocess
import sys
import time

def run_script(path):
    """Run a Python script and abort if it fails."""
    print(f"➡️  Starting {path}...")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"❌  {path} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)
    print(f"✅  {path} finished successfully.\n")

def main():
    scripts = [
        "get_data2/live_update_pipeline2.py",
        "scripts/main_v3_batch3_copy.py",
        "wm.py",
    ]

    for script in scripts:
        run_script(script)

    print("🎉  All scripts completed. Sleeping for 24 hours now...")
    time.sleep(24 * 3600)
    print("⏰  Wake up! You can trigger the master script again if desired.")

if __name__ == "__main__":
    main()
