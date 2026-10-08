"""Private terminal source and authorized-destination binding (Linux)."""

import hashlib
import fcntl
import os
import stat
from pathlib import Path
import sys
import types
from contextlib import contextmanager
from contextvars import ContextVar

SOURCES = {}
_DESTINATIONS = ContextVar("terminal_destinations", default=None)
_PROTECTED_DEVICES = ContextVar("terminal_protected_devices", default=None)


@contextmanager
def device_scope():
    if _PROTECTED_DEVICES.get() is not None:
        yield
        return
    token = _PROTECTED_DEVICES.set(set())
    try:
        yield
    finally:
        _PROTECTED_DEVICES.reset(token)


def protect_device(device):
    devices = _PROTECTED_DEVICES.get()
    if devices is None:
        raise ValueError("protected device scope required")
    devices.add(device)


def protected_devices():
    return _PROTECTED_DEVICES.get() or ()


def label_identity(path):
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("regular original label file required")
    protect_device(info.st_dev)
    return info.st_dev, info.st_ino


def capture(path, executed):
    """Bootstrap must agree with the executing code; owners use load below."""
    path = Path(path).resolve()
    data = path.read_bytes()
    if compile(data, executed.co_filename, "exec", dont_inherit=True) != executed:
        raise ValueError("bootstrap source differs from executing code")
    SOURCES[path] = hashlib.sha256(data).hexdigest()


def load(path, name):
    """Hash and compile the same source buffer, bypassing bytecode caches."""
    path = Path(path).resolve()
    data = path.read_bytes()
    SOURCES[path] = hashlib.sha256(data).hexdigest()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@contextmanager
def destinations(stack):
    state = {"stack": stack}
    token = _DESTINATIONS.set(state)
    try:
        yield
    finally:
        _DESTINATIONS.reset(token)


def bind_destinations(auth, ledger, journal, output, admit, protected):
    # Resolve each name once at the authorization check. Later consumers use
    # the retained ledger fd and these canonical journal/output spellings.
    authorized = (admit(auth["ledger"]), admit(auth["journal"]))
    actual = (admit(ledger), admit(journal))
    if authorized != actual:
        raise ValueError("exposure destinations not authorized")
    state = _DESTINATIONS.get()
    if state is None:
        return  # read-only preflight caller; execute always binds destinations
    from _terminal_io import _open_admitted

    before = os.stat(actual[0], follow_symlinks=False)
    f = state["stack"].enter_context(
        _open_admitted(actual[0], "r+", protected=protected)
    )
    now = os.fstat(f.fileno())
    if (before.st_dev, before.st_ino) != (now.st_dev, now.st_ino):
        raise ValueError("authorized ledger identity changed")
    state.update(
        ledger=f, ledger_path=actual[0],
        paths={Path(journal): actual[1], Path(output): admit(output)}
    )


@contextmanager
def ledger_file(path, open_metadata):
    state = _DESTINATIONS.get()
    if state is not None:
        f = state["ledger"]
        with _locked_ledger(f, state["ledger_path"]):
            yield f
    else:
        with open_metadata(path, "r+") as f:
            with _locked_ledger(f, Path(path).resolve()):
                yield f


def _check_ledger(f, path):
    retained = os.fstat(f.fileno())
    canonical = os.stat(path, follow_symlinks=False)
    if (retained.st_nlink != 1 or canonical.st_nlink != 1
            or (retained.st_dev, retained.st_ino) != (canonical.st_dev, canonical.st_ino)
            or retained.st_dev in protected_devices()):
        raise ValueError("authorized ledger identity changed")


@contextmanager
def _locked_ledger(f, path):
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
        _check_ledger(f, path)
        f.seek(0)
        yield f
        _check_ledger(f, path)
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)


def append_ledger(f, path, text):
    state = _DESTINATIONS.get()
    path = state["ledger_path"] if state is not None else Path(path).resolve()
    # Recheck immediately before and after the durable append, including an
    # uncoordinated rename during the write. Never report success on an orphan.
    _check_ledger(f, path)
    f.seek(0, os.SEEK_END)
    f.write(text)
    f.flush()
    os.fsync(f.fileno())
    _check_ledger(f, path)


def destination(path, admit):
    state = _DESTINATIONS.get()
    paths = state.get("paths", {}) if state is not None else {}
    return paths[Path(path)] if Path(path) in paths else admit(path)


capture(__file__, sys._getframe().f_code)
