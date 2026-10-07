"""Linux leaf binding for the terminal owner's admitted directory walk."""

import errno
import os
import stat


def _identity(info):
    return info.st_dev, info.st_ino


def _regular(info):
    if stat.S_ISLNK(info.st_mode):
        raise OSError(errno.ELOOP, "metadata leaf became a symlink")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("regular metadata file with one link required")


def _open_regular_at(parent, name, flags, exclusive=False):
    """Return a data fd for a checked inode; never reopen the caller's leaf."""
    leaf = data = None
    try:
        if exclusive:
            # O_EXCL never opens an existing leaf, including an inserted alias.
            data = os.open(name, flags, 0o600, dir_fd=parent)
        else:
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _regular(before)
            # O_PATH acquires identity without a data open/IN_OPEN event. A
            # hard link inserted after stat is checked before any data handle.
            leaf = os.open(name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
            info = os.fstat(leaf)
            _regular(info)
            if _identity(info) != _identity(before):
                raise ValueError("metadata leaf identity changed")
            # Only this kernel-held object can receive the data open. Follow
            # its /proc magic link deliberately, never a mutable caller name.
            data = os.open(f"/proc/self/fd/{leaf}", flags & ~os.O_NOFOLLOW)
            opened = os.fstat(data)
            _regular(opened)
            if _identity(opened) != _identity(info):
                raise ValueError("metadata handle identity changed")
        _regular(os.fstat(data))
        result, data = data, None
        return result
    finally:
        if data is not None:
            os.close(data)
        if leaf is not None:
            os.close(leaf)


def _open_admitted(resolved, mode="rb", **kwargs):
    """Walk the owner's admitted absolute spelling without following aliases."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    modes = {
        "rb": os.O_RDONLY,
        "r+": os.O_RDWR,
        "x": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        "w": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        "a": os.O_WRONLY | os.O_APPEND,
    }
    flags = modes[mode] | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    parent = os.open("/", directory_flags)
    fd = None
    try:
        for component in resolved.parts[1:-1]:
            next_parent = os.open(component, directory_flags, dir_fd=parent)
            os.close(parent)
            parent = next_parent
        fd = _open_regular_at(parent, resolved.name, flags, mode in ("x", "w"))
        bound = os.fdopen(fd, mode, **kwargs)
        fd = None  # bound owns it, including exceptional close paths
        return bound
    finally:
        os.close(parent)
        if fd is not None:
            os.close(fd)
