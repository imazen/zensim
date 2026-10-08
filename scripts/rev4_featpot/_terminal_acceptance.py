"""Private terminal source and authorized-destination binding (Linux)."""

import hashlib
import os
from pathlib import Path
import sys
import types
from contextlib import contextmanager
from contextvars import ContextVar

SOURCES = {}
_DESTINATIONS = ContextVar("terminal_destinations", default=None)


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
        ledger=f, paths={Path(journal): actual[1], Path(output): admit(output)}
    )


@contextmanager
def ledger_file(path, open_metadata):
    state = _DESTINATIONS.get()
    if state is not None:
        f = state["ledger"]
        f.seek(0)
        yield f
    else:
        with open_metadata(path, "r+") as f:
            yield f


def destination(path, admit):
    state = _DESTINATIONS.get()
    paths = state.get("paths", {}) if state is not None else {}
    return paths[Path(path)] if Path(path) in paths else admit(path)


capture(__file__, sys._getframe().f_code)
