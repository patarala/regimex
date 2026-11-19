"""Lifetime of child processes supervised by the service."""

import ctypes
import os
import signal
import sys


def bind_to_parent(parent: int) -> bool:
    # Linux delivers the signal even while the child is blocked in native I/O.
    # Check again after prctl in case the coordinator died before registration.
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            return False
    return os.getppid() == parent
