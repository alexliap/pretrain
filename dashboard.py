"""Launch the Trackio dashboard with write access (delete/rename enabled).

By default Trackio prints a read-only dashboard URL; the write-enabled URL
(which carries the per-launch ``write_token``) is only returned, not printed.
This helper launches the UI and prints that write URL so deleting/renaming
runs works straight from the browser.

Usage:
    uv run dashboard.py                  # all projects
    uv run dashboard.py test_run_lfm25   # open a specific project
"""

import sys

import trackio

if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else None

    # block_thread=True keeps the server alive until Ctrl+C.
    _, _, _, full_url = trackio.show(project=project, block_thread=False)

    print("\n" + "=" * 70)
    print("Write-enabled dashboard (delete/rename works here):")
    print(f"  {full_url}")
    print("=" * 70)
    print("\nPress Ctrl+C to stop the dashboard.\n")

    try:
        from trackio import utils

        utils.block_main_thread_until_keyboard_interrupt()
    except KeyboardInterrupt:
        pass
