import multiprocessing as mp
import os
import signal
import time
from collections.abc import Callable
from datetime import datetime
from multiprocessing.context import SpawnProcess
from pathlib import Path

from g7_openarm_config import general_config

LOG_ROOT = Path("logs")


def build_log_path(folder_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return LOG_ROOT / folder_name / f"{timestamp}.log"


def redirect_output_to_file(log_path: Path) -> None:
    if general_config.debugging:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        log_fd = os.open(
            str(log_path),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
        )

        os.dup2(log_fd, 1)  # stdout
        os.dup2(log_fd, 2)  # stderr

        os.close(log_fd)
    else:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        os.dup2(devnull_fd, 1)  # stdout
        os.dup2(devnull_fd, 2)  # stderr

        os.close(devnull_fd)


def run_silently(target: Callable[[], None], folder_name: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    log_path = build_log_path(folder_name)
    redirect_output_to_file(log_path)
    target()


def run_mujoco() -> None:
    from g7_openarm_mujoco.vr_sim_node import main

    run_silently(main, "mujoco")


def run_lowlevel() -> None:
    from g7_openarm_lowlevel.lowlevel_node import main

    run_silently(main, "lowlevel")


def run_state_estimator() -> None:
    from g7_openarm_state_estimator.odom_node import main

    run_silently(main, "state_estimator")


def run_wbc() -> None:
    from g7_openarm_wbc.wbc_node import main

    run_silently(main, "wbc")


def run_vr() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from g7_openarm_vr.vr_node import main

    main()


def stop_processes(
    processes: list[SpawnProcess],
    timeout: float = 3.0,
) -> None:
    # send SIGTERM
    for process in processes:
        if process.is_alive():
            process.terminate()

    deadline = time.monotonic() + timeout

    # wait for 3 seconds
    for process in processes:
        remaining = deadline - time.monotonic()

        if remaining > 0:
            process.join(timeout=remaining)

    # send SIGKILL
    for process in processes:
        if process.is_alive():
            print(f"Force killing {process.name}...")
            process.kill()

    for process in processes:
        process.join()


def main() -> None:
    ctx = mp.get_context("spawn")

    processes = [
        ctx.Process(target=run_mujoco, name="g7-mujoco"),
        ctx.Process(target=run_lowlevel, name="g7-lowlevel"),
        ctx.Process(target=run_state_estimator, name="g7-state-est"),
        ctx.Process(target=run_vr, name="g7-vr"),
        ctx.Process(target=run_wbc, name="g7-wbc"),
    ]

    for process in processes:
        process.start()

    try:
        while True:
            for process in processes:
                if not process.is_alive():
                    print(f"\n{process.name} exited with code {process.exitcode}")
                    return

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping all processes...")

    finally:
        stop_processes(processes, timeout=3.0)


if __name__ == "__main__":
    main()
