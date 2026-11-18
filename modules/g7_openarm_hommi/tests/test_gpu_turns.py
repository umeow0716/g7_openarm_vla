import itertools
import threading
import time

from g7_openarm_hommi.gpu_turns import GpuTurns


def _run(turns, party, log, stop, hold_s):
    while not stop.is_set():
        with turns.turn(party):
            log.append(("in", party))
            time.sleep(hold_s)
            log.append(("out", party))


def test_turns_are_exclusive_and_alternate_when_both_wait():
    turns = GpuTurns()
    log: list[tuple[str, str]] = []
    stop = threading.Event()
    threads = [
        threading.Thread(target=_run, args=(turns, p, log, stop, 0.005))
        for p in ("policy", "stereo")
    ]
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join(1.0)

    owners = []
    for i in range(0, len(log) - 1, 2):
        assert log[i][0] == "in" and log[i + 1] == ("out", log[i][1]), "turns overlapped"
        owners.append(log[i][1])
    assert len(owners) > 10
    # Both loop back-to-back, so neither may take two turns in a row (after the first).
    assert all(a != b for a, b in itertools.pairwise(owners[1:]))


def test_a_lone_party_runs_back_to_back():
    turns = GpuTurns()
    for _ in range(3):
        with turns.turn("stereo"):
            pass
