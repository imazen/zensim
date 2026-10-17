"""Restore the caller's original signal mask before an owned command execs."""
import os
import signal
import sys
mask={int(value) for value in sys.argv[1].split(',') if value}
signal.pthread_sigmask(signal.SIG_SETMASK,mask)
os.execvp(sys.argv[2],sys.argv[2:])
