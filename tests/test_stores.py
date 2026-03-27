from pathlib import Path
import shutil

from simplezarr.stores import ReadableStore, WritableStore, ListableStore, check_key
from simplezarr.stores import MemoryStore, LocalStore, WrapperStore

import pytest

List = list

data = {
    "foo": b"hi-foo",
    "bar": b"hi-bar",
    "dir1/foo": b"hi-foo",
    "dir1/bar": b"hi-bar",
    "dir2/foo": b"hi-foo",
    "dir2/bar": b"hi-bar",
}

test_dir1 = Path(__file__).absolute().parent / "test-data"


def setup_module():
    shutil.rmtree(test_dir1, ignore_errors=True)
    for k, v in data.items():
        f = test_dir1.joinpath(*k.split("/"))
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(v)


def teardown_module():
    shutil.rmtree(test_dir1)


# Define classes that minimally extend the base classes so their default implementations (like partial read and writes) can be tested.


class StoreThatFillsTheGaps:
    def __init__(self):
        self._store = data.copy()

    def get(self, key: str) -> bytes:
        check_key(self, "get", key)
        try:
            return self._store[key]
        except KeyError:
            raise IOError(f"get(): key {key!r} does not exist.") from None

    def set(self, key: str, value: bytes):
        check_key(self, "set", key)
        self._store[key] = value

    def erase(self, key: str):
        check_key(self, "erase", key)
        try:
            self._store.pop(key)
        except KeyError:
            raise IOError(f"erase(): key {key!r} does not exist.") from None

    def list(self) -> list[str]:
        return sorted(self._store.keys())


class ReadableStorePlus(StoreThatFillsTheGaps, ReadableStore):
    pass


class WritableStorePlus(StoreThatFillsTheGaps, WritableStore):
    def list_prefix(self, prefix: str) -> List[str]:
        check_key(self, "list_prefix", prefix, "prefix")
        return [key for key in self.list() if key.startswith(prefix)]


class ListableStorePlus(StoreThatFillsTheGaps, ListableStore):
    pass


class MemoryStorePlus(MemoryStore):
    def __init__(self):
        super().__init__(data)


class LocalStorePlus(LocalStore):
    def __init__(self):
        super().__init__(test_dir1)


class WrapperStorePlus(WrapperStore):
    def __init__(self):
        super().__init__(MemoryStore(data))


store_classes = [
    ReadableStorePlus,
    WritableStorePlus,
    ListableStorePlus,
    MemoryStorePlus,
    LocalStorePlus,
]


@pytest.mark.parametrize("cls", store_classes)
def test_read(cls):
    # Test the default logic in the ReadableStore base class

    assert isinstance(cls, type)
    if not issubclass(cls, ReadableStore):
        pytest.skip()

    store = cls()


@pytest.mark.parametrize("cls", store_classes)
def test_list(cls):
    assert isinstance(cls, type)
    if not issubclass(cls, ListableStore):
        pytest.skip()

    store = cls()


@pytest.mark.parametrize("cls", store_classes)
def test_write(cls):
    assert isinstance(cls, type)
    if not issubclass(cls, WritableStore):
        pytest.skip()

    store = cls()


if __name__ == "__main__":
    setup_module()
    for func in [test_read, test_list, test_write]:
        for cls in store_classes:
            print(f"{func.__name__}[{cls.__name__}] ... ", end="")
            try:
                func(cls)
            except pytest.skip.Exception:
                print("skip")
            else:
                print("done")
    print("all done")
    teardown_module()
