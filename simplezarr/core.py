""" """

from __future__ import annotations  # Using class names for types without Ruff F821
import json

import numpy as np

from .stores import ReadableStore, WritableStore, ListableStore


def load_zarr(store: ReadableStore) -> ZarrNode:
    return ZarrNode._from_path(store, "")


def join(*path_parts):
    return "/".join(path_parts).lstrip("/")


class ZarrNode:
    def __init__(
        self,
        store: ReadableStore | ListableStore | WritableStore,
        path: str,
        _metadata: dict | None = None,
    ):
        # Check path
        if not isinstance(path, str):
            raise TypeError(f"{self.__class__.__name__} path must be str, got {path!r}")
        path = path.lstrip("/")
        if path.endswith("/"):
            raise ValueError(
                f"{self.__class__.__name__} path must not end with '/' unless root, got {path!r}"
            )

        self._store = store
        self._path = path
        self._name = self._path.rsplit("/", 1)[-1]

        # Get metadata as a dict
        if _metadata is not None:
            metadata = _metadata
        else:
            json_text = self._store.get(join(self._path, "zarr.json")).decode()
            metadata = json.loads(json_text)
        assert isinstance(metadata, dict)
        self._metadata = metadata

        self._parse_metadata()
        self._init_node()

    @classmethod
    def _from_path(cls, store, path):
        json_text = store.get(join(path, "zarr.json")).decode()
        metadata = json.loads(json_text)

        if metadata["zarr_format"] != 3:
            raise RuntimeError("Assuming Zarr version 3")

        node_type = metadata["node_type"]
        if node_type == "group":
            return ZarrGroup(store, path, _metadata=metadata)
        elif node_type == "array":
            return ZarrArray(store, path, _metadata=metadata)
        else:
            raise RuntimeError(f"Unexpected node type {node_type!r}")

    @property
    def store(self):
        return self._store

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def metadata(self):
        return self._metadata

    def pprint_metadata(self):
        print(json.dumps(self._metadata, indent=4))

    def _one_line_repr(self):
        return f"<{self.__class__.__name__} '{self._path}' at {hex(id(self))}>"

    def _parse_metadata(self):
        raise NotImplementedError()

    def _init_node(self):
        raise NotImplementedError()


class ZarrGroup(ZarrNode):
    def __repr__(self):
        return self.get_structure(max_depth=1)

    def _one_line_repr(self):
        return f"<{self.__class__.__name__} '{self._path}' with {len(self._children)} children at {hex(id(self))}>"

    @property
    def children(self) -> tuple[ZarrNode]:
        return tuple(self._children.values())

    @property
    def attributes(self):
        return self._attributes

    def show_structure(self, max_depth=999):
        print(self.get_structure(max_depth=max_depth))

    def get_structure(self, max_depth=999, indent=0):
        indent_str = " " * indent
        r = indent_str + self._one_line_repr()
        if self._children and max_depth > 0:
            for child in self.children:
                r += "\n"
                if isinstance(child, ZarrGroup):
                    r += child.get_structure(indent + 4, max_depth - 1)
                else:
                    r += " " * (indent + 4) + child._one_line_repr()
        return r

    def __getitem__(self, path):
        if not isinstance(path, str):
            raise TypeError("ZarrGroup indexing must be done with a str path.")

        name, _, remaining_path = path.rstrip("/").partition("/")

        try:
            ob = self._children[name]
        except KeyError:
            raise KeyError(
                f"ZarrGroup '{self._path}' does not have a child named {name!r}."
            ) from None

        if remaining_path:
            return ob[remaining_path]
        else:
            return ob

    def _parse_metadata(self):
        meta = self._metadata

        # Parse mandatory fields
        assert meta["node_type"] == "group"

        # Parse optional fields
        self._attributes = meta.get("attributes", None)

    def _init_node(self):
        # Assume ListableStore

        # todo: use consolidated metadata
        # todo: use list_dir only lazily

        n = len(self._path)
        items = self._store.list_dir(self._path + "/")
        dir_names = [item[n:].strip("/") for item in items if item.endswith("/")]

        self._children = {}
        for name in dir_names:
            try:
                node = ZarrNode._from_path(self._store, f"{self._path}/{name}")
            except IOError:
                continue
            else:
                self._children[name] = node


class ZarrArray(ZarrNode):
    def __repr__(self):
        return self._one_line_repr()
        # todo: could add info on chunks

    def _one_line_repr(self):
        shape_str = "x".join(str(i) for i in self.shape)
        return f"<{self.__class__.__name__} '{self._path}' {shape_str} {self.dtype} at {hex(id(self))}>"

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        return self._chunk_shape

    @property
    def chunk_size(self) -> int:
        return int(np.prod(self._chunk_shape))

    def get_chunk(self, *index):
        pass

    @property
    def codec(self):
        pass

    def _parse_metadata(self):
        meta = self._metadata

        # Parse mandatory fields

        assert meta["node_type"] == "array"

        self._shape = tuple(int(i) for i in meta["shape"])
        self._dtype = meta["data_type"]

        self._chunk_grid = meta["chunk_grid"]
        assert self._chunk_grid["name"] == "regular"
        self._chunk_shape = self._chunk_grid["configuration"]["chunk_shape"]

        self._chunk_key_encoding = meta["chunk_key_encoding"]
        assert self._chunk_key_encoding["name"] == "default"
        self._chunk_separator = self._chunk_key_encoding["configuration"]["separator"]

        self._fill_value = meta["fill_value"]
        self._codecs = meta["codecs"]
        assert len(self._codecs) >= 1
        assert self._codecs[0]["name"] == "bytes"

        # Parse optional fields

        self._attributes = meta.get("attributes", None)
        self._storage_transformers = meta.get("storage_transformers", None)
        self._dimension_names = meta.get("dimension_names", None)

    def _init_node(self):
        pass


class ZarrChunk:
    pass
