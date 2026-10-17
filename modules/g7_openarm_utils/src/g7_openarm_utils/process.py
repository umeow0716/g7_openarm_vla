import os
import signal

def force_kill(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)