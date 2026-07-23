import multiprocessing as mp
import os
import signal
import time
from collections.abc import Callable
from multiprocessing.context import SpawnProcess


def suppress_output() -> None:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    os.dup2(devnull_fd, 1)  # stdout
    os.dup2(devnull_fd, 2)  # stderr

    os.close(devnull_fd)


def run_silently(target: Callable[[], None]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    suppress_output()
    target()


def run_mujoco() -> None:
    from g7_openarm_mujoco.real_cli import main

    run_silently(main)

def run_hardware() -> None:
    from g7_openarm_hardware.cli import main

    run_silently(main)

def run_imu() -> None:
    from g7_openarm_hardware.imu_cli import main

    run_silently(main)

def run_lowlevel() -> None:
    from g7_openarm_lowlevel.cli import main

    run_silently(main)


def run_state_estimator() -> None:
    from g7_openarm_state_estimator.cli import main

    run_silently(main)


def run_wbc() -> None:
    from g7_openarm_wbc.cli import main
    
    run_silently(main)


def run_monitor() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from g7_openarm_monitor.cli import main

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
        ctx.Process(target=run_mujoco, name="g7-mujoco-real"),
        ctx.Process(target=run_hardware, name="g7-hardware"),
        ctx.Process(target=run_imu, name="g7-imu"),
        ctx.Process(target=run_lowlevel, name="g7-lowlevel"),
        ctx.Process(target=run_state_estimator, name="g7-state-est"),
        ctx.Process(target=run_monitor, name="g7-monitor"),
        ctx.Process(target=run_wbc, name="g7-wbc")
    ]

    for process in processes:
        process.start()

    try:
        while True:
            for process in processes:
                if not process.is_alive():
                    print(
                        f"\n{process.name} exited "
                        f"with code {process.exitcode}"
                    )
                    return

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping all processes...")

    finally:
        stop_processes(processes, timeout=3.0)


if __name__ == "__main__":
    main()