"""
The store interface and some implementations.

https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html

Some quotes from the spec, for easy reference:

* The store interface is intended to be simple to implement using a
  variety of different underlying storage technologies.
* It is assumed that the store holds (key, value) pairs, with only one
  such pair for any given key. I.e., a store is a mapping from keys to
  values.
* It is also assumed that keys are case sensitive, i.e., the keys “foo”
  and “FOO” are different.
* In the context of this interface, a key is a Unicode string, where
  the final character is not a '/' character.
* In the context of this interface, a prefix is a string containing
  only characters that are valid for use in keys and ending with a
  trailing '/' character.
* The store operations are grouped into three sets of capabilities:
  readable, writeable and listable. It is not necessary for a store
  implementation to support all of these capabilities.

"""

from pathlib import Path


def _check_key(ob: object, method: str, key: str, name: str = "key"):
    if not isinstance(key, str):
        raise TypeError(
            f"{ob.__class__.__name__}.{method}(): {name} must be a str, got {key!r}"
        )
    if key != "/" and key.startswith("/"):
        raise ValueError(
            f"{ob.__class__.__name__}.{method}(): {name} must not start with '/', got {key!r}"
        )
    if name == "key":
        if not key:
            raise ValueError(
                f"{ob.__class__.__name__}.{method}(): {name} must not be empty, got {key!r}"
            )
        if key.endswith("/"):
            raise ValueError(
                f"{ob.__class__.__name__}.{method}(): {name} must not end with '/', got {key!r}"
            )
    elif name == "prefix":  # a 'directory'
        if not key.endswith("/"):
            raise ValueError(
                f"{ob.__class__.__name__}.{method}(): {name} must end with '/', got {key!r}"
            )


class BaseStore:
    pass


class ReadableStore(BaseStore):
    def get(self, key: str) -> bytes:
        """Retrieve the value associated with a given key."""
        _check_key(self, "get", key)
        raise NotImplementedError()

    def get_partial_values(
        self, key_ranges: list[tuple[str, int, int | None]]
    ) -> list[bytes]:
        """Retrieve possibly partial values from given key_ranges.

        The ``key_ranges`` is an iterable of (key, range_start, range_length),
        where range_length may be None to indicate the full remaining length.
        """
        # Default implementation
        result = []
        for key, start, length in key_ranges:
            _check_key(self, "get_partial_values", key)
            i1 = int(start)
            i2 = None if length is None else i1 + length
            full_value = self.get(key)
            result.append(full_value[i1:i2])
        return result


class WritableStore(BaseStore):
    def set(self, key: str, value: bytes):
        """Store a (key, value) pair."""
        _check_key(self, "set", key)
        raise NotImplementedError()

    def set_partial_values(self, key_start_values: list[tuple[str, int, bytes]]):
        """Store values at a given key, starting at byte range_start."""
        # Default implementation, assumes ReadableStore with .get()
        for key, start, value in key_start_values:
            _check_key(self, "set_partial_values", key)
            i1 = int(start)
            i2 = i1 + len(value)
            full_value = self.get(key)
            full_value = full_value[:i1] + value + full_value[i2:]
            self.set(key, full_value)

    def erase(self, key: str):
        """Erase the given key/value pair from the store."""
        _check_key(self, "erase", key)
        raise NotImplementedError()

    def erase_values(self, keys: list[str]):
        """Erase the given key/value pairs from the store."""
        # Default implementation
        for key in keys:
            _check_key(self, "erase_values", key)
            self.erase(key)

    def erase_prefix(self, prefix: str):
        """Erase all keys with the given prefix from the store.

        The prefix represents a 'directory'; it must end with a '/'.
        """
        # Default implementation, assumes ListableStore with .list_prefix()

        _check_key(self, "erase_values", prefix, "prefix")
        for key in self.list_prefix(prefix):
            self.erase(key)


class ListableStore(BaseStore):
    def list(self) -> list[str]:
        """Retrieve all keys in the store."""
        raise NotImplementedError()

    def list_prefix(self, prefix: str) -> list[str]:
        """Retrieve all keys with a given prefix.

        The prefix represents a 'directory'; it must end with a '/'. This method
        lists the full (recursive) list of items in that directory.

        For example, if a store contains the keys “a/b”, “a/c/d” and “e/f/g”,
        then ``list_prefix("a/")`` would return “a/b” and “a/c/d”.
        """
        # Default implementation
        _check_key(self, "list_prefix", prefix, "prefix")
        return [key for key in self.list() if key.startswith(prefix)]

    def list_dir(self, prefix: str) -> list[str]:
        """Retrieve all keys within a given directory.

        The prefix represents a 'directory'; it must end with a '/'. This method
        lists only the keys in that directory and not in that of any
        subdirectories. But it does return prefixes (i.e. directories) within
        the given directory.

        For example, if a store contains the keys “a/b”, “a/c”, “a/d/e”,
        “a/f/g”, then ``list_dir("a/")`` would return keys “a/b” and “a/c” and
        prefixes “a/d/” and “a/f/”. ``list_dir("b/")`` would return the empty
        set.
        """
        # Default implementation
        _check_key(self, "list_dir", prefix, "prefix")
        n = len(prefix)
        keys = set()
        for key in self.list():
            if key.startswith(prefix):
                key, dash, _rest = key[n:].partition("/")
                keys.add(prefix + key + dash)
        return sorted(keys)


# %%%%% Implementations


class LocalStore(ReadableStore, WritableStore, ListableStore):
    def __init__(self, path: str | Path):
        self._path = Path(path)

    def __repr__(self):
        return f"<LocalStore '{self._path}' at {hex(id(self))}>"

    def get(self, key: str) -> bytes:
        _check_key(self, "get", key)
        p = self._path.joinpath(*key.split("/"))
        if p.is_file():
            return p.read_bytes()
        else:
            raise IOError(f"get(): key {key!r} does not exist.")

    def get_partial_values(
        self, key_ranges: list[str, tuple[int, int | None]]
    ) -> list[bytes]:
        result = []
        for key, start, length in key_ranges:
            _check_key(self, "get_partial_values", key)
            i1 = int(start)
            length = None if length is None else int(length)
            p = self._path.joinpath(*key.split("/"))
            if not p.is_file():
                raise IOError(f"Key {key!r} does not exist.")
            with p.open("rb") as f:
                f.seek(i1)
                result.append(f.read(length))
        return result

    def set(self, key: str, value: bytes):
        _check_key(self, "set", key)
        p = self._path.joinpath(*key.split("/"))
        p.parent.mkdir(parent=False, exist_ok=True)
        p.write_bytes(value)

    def set_partial_values(self, key_start_values: list[tuple[str, int, bytes]]):
        for key, start, value in key_start_values:
            _check_key(self, "set_partial_values", key)
            i1 = int(start)
            p = self._path.joinpath(*key.split("/"))
            p.parent.mkdir(parent=False, exist_ok=True)
            with p.open("ab") as f:
                f.seek(i1)
                f.write(value)

    def erase(self, key: str):
        _check_key(self, "erase", key)
        p = self._path.joinpath(*key.split("/"))
        if p.is_file():
            p.unlink()
        else:
            raise IOError(f"erase(): key {key!r} does not exist.")

    def erase_values(self, keys: list[str]):
        return super().erase_values(keys)  # use default implementation

    def erase_prefix(self, prefix: str):
        return super().erase_prefix(keys)  # use default implementation
        # could use shutil.rmtree if we want to optimize this

    def list(self) -> list[str]:
        return sorted([str(p.relative_to(self._path)) for p in self._path.rglob("*")])

    def list_prefix(self, prefix: str) -> list[str]:
        _check_key(self, "list_prefix", prefix, "prefix")
        d = self._path.joinpath(*prefix.split("/"))
        return sorted([str(p.relative_to(self._path)) for p in d.rglob("*")])

    def list_dir(self, prefix: str) -> list[str]:
        _check_key(self, "list_dir", prefix, "prefix")
        d = self._path.joinpath(*prefix.split("/"))
        keys = set()
        for p in d.iterdir():
            key = str(p.relative_to(self._path))
            dash = "/" if p.is_dir() else ""
            keys.add(key + dash)
        return sorted(keys)
