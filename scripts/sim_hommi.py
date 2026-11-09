import argparse
import multiprocessing as mp
import os
import signal
import time

from collections.abc import Callable
from datetime import datetime
from g7_openarm_config import general_config
from g7_openarm_utils.layout import ControlMode
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported lazily: spawn re-imports this module in every child process.
    from g7_openarm_hommi.replay import ReplayOptions

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
    from g7_openarm_mujoco.mujoco_node import main

    run_silently(lambda: main(is_sim=True, is_vr=False, is_hommi=True), "mujoco")


def run_lowlevel() -> None:
    from g7_openarm_lowlevel.lowlevel_node import main

    run_silently(main, "lowlevel")


def run_state_estimator() -> None:
    from g7_openarm_state_estimator.odom_node import main

    run_silently(main, "state_estimator")


def run_wbc() -> None:
    from g7_openarm_wbc.wbc_node import main

    run_silently(main, "wbc")


def run_monitor() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from g7_openarm_monitor.monitor_node import main

    main()


def run_hommi(replay: "ReplayOptions | None" = None) -> None:
    from g7_openarm_hommi.hommi_node import main

    run_silently(lambda: main("simulation", replay), "hommi")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HoMMI ego3d policy in MuJoCo.")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="iPhUMI export used as the cameras (head pointmap + wrist RGB) instead "
        "of the black sim cameras; robot state still comes from the simulation",
    )
    parser.add_argument(
        "--episode",
        default="0",
        help="demonstration index (sorted by time) or a unique title substring",
    )
    parser.add_argument(
        "--mode",
        choices=("policy", "trajectory"),
        default="policy",
        help="policy: the recording replaces the cameras and the robot follows the "
        "policy; trajectory: no inference, the robot follows the demonstrated "
        "end-effector trajectory (the observation window still shows the recording)",
    )
    parser.add_argument("--loop", action="store_true", help="restart the recording at the end")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay = None
    if args.dataset is not None:
        from g7_openarm_hommi.replay import ReplayOptions

        if not args.dataset.is_dir():
            raise SystemExit(f"--dataset {args.dataset} is not a directory")
        replay = ReplayOptions(
            root=str(args.dataset.expanduser().resolve()),
            episode=args.episode,
            mode=args.mode,
            loop=args.loop,
        )
        print(f"Dataset replay: {replay.root} (episode {replay.episode}, mode {replay.mode})")

    if general_config.control_mode is ControlMode.BASE_ONLY:
        raise RuntimeError(
            "control_mode='base-only' requires a VR script; "
            "use scripts/sim_vr.py or scripts/real_vr.py"
        )

    ctx = mp.get_context("spawn")

    processes = [
        ctx.Process(target=run_mujoco, name="g7-mujoco"),
        ctx.Process(target=run_lowlevel, name="g7-lowlevel"),
        ctx.Process(target=run_state_estimator, name="g7-state-est"),
        ctx.Process(target=run_monitor, name="g7-monitor"),
        ctx.Process(target=run_wbc, name="g7-wbc"),
        ctx.Process(target=run_hommi, args=(replay,), name="g7-hommi"),
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
