"""
https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-encoding
https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html

* In encoding, you start with an array, and end with bytes. There is exactly one array-> bytes codec step.
* In decoding, you start with bytes and end with an array. There is exactly one bytes-> array codec step.
* This specification defines a set of codecs (“core codecs”) which all Zarr implementations SHOULD implement.

Third party code can use can subclass ``BaseCodec`` and use ``register_codec()`` to implement custom codecs as extensions.
"""

from __future__ import annotations

import sys

import numcodecs
import numpy as np


CODEC_CLASS_BY_NAME = {}


def register_codec(cls: type):
    assert isinstance(cls, type) and issubclass(cls, BaseCodec)
    assert cls.name
    CODEC_CLASS_BY_NAME[cls.name] = cls


ndarray = np.ndarray


def create_ndarray_type(shape: tuple[int, ...], dtype: str):
    assert isinstance(dtype, str)
    shape_str = "x".join(str(i) for i in shape)
    return type(
        f"ndarray_{shape_str}_{dtype}",
        (ndarray,),
        {"shape": shape, "dtype": dtype, "__module__": ""},
    )


def encode_bytes(array: ndarray, codec_dicts: list[dict]) -> bytes:
    # Get codecs, with their order validated
    array_type = create_ndarray_type(array.shape, array.dtype)
    codecs, decoded_representation_types = resolve_codecs_from_dicts(
        codec_dicts, array_type
    )

    # Encode
    value = array
    assert isinstance(value, decoded_representation_types[0])
    for i in range(len(codecs)):
        value = codecs[i].encode(value)
        assert isinstance(value, decoded_representation_types[i + 1])

    return value


def decode_bytes(
    encoded_bytes: bytes, codec_dicts: list[dict], array_type: type
) -> ndarray:
    # Get codecs, with their order validated
    codecs, decoded_representation_types = resolve_codecs_from_dicts(
        codec_dicts, array_type
    )

    # Reverse; this is a *decoder*
    codecs.reverse()
    decoded_representation_types.reverse()

    # Decode
    value = encoded_bytes
    assert isinstance(value, decoded_representation_types[0])
    for i in range(len(codecs)):
        value = codecs[i].decode(value, decoded_representation_types[i + 1])

    return value


def resolve_codecs_from_dicts(
    codec_dicts: list[dict], array_type: type
) -> tuple[list[BaseCodec], list[type]]:
    # Create codecs
    codecs = []
    for codec_dict in codec_dicts:
        name = codec_dict["name"]
        configuration = codec_dict["configuration"]
        try:
            cls = CODEC_CLASS_BY_NAME[name]
        except KeyError:
            raise RuntimeError(f"Unknown Zarr codec {name}") from None
        codecs.append(cls(**configuration))

    # Resolve types
    # See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#determination-of-encoded-representations
    decoded_representation_types = [array_type]
    for i in range(len(codecs)):
        t = codecs[i].compute_encoded_representation_type(
            decoded_representation_types[i]
        )
        decoded_representation_types.append(t)

    assert decoded_representation_types[-1] is bytes

    return codecs, decoded_representation_types


class BaseCodec:
    name = ""

    def __init__(self, **configuration):
        self._configuration = configuration

    def compute_encoded_representation_type(self, decoded_representation_type: type):
        assert isinstance(decoded_representation_type, type)
        raise NotImplementedError()

    def encode(self, decoded: bytes | ndarray) -> bytes | ndarray:
        raise NotImplementedError()

    def decode(
        self, encoded: bytes | ndarray, decoded_representation_type: type
    ) -> bytes | ndarray:
        raise NotImplementedError()


@register_codec
class BytesCodec(BaseCodec):
    """
    Implements an ``array -> bytes`` codec that encodes arrays of fixed-size numeric
    data types as a sequence of bytes in lexicographical order. For multi-byte
    data types, it encodes the array either in little endian or big endian.

    https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
    """
    name = "bytes"

    def compute_encoded_representation_type(self, decoded_representation_type: type):
        assert isinstance(decoded_representation_type, type)
        if issubclass(decoded_representation_type, ndarray):
            return bytes
        else:
            raise RuntimeError("BytesCodec only encodes arrays.")

    def encode(self, decoded: bytes | ndarray) -> bytes | ndarray:
        raise NotImplementedError()

    def decode(
        self, encoded: bytes | ndarray, decoded_representation_type: type
    ) -> ndarray:
        if not issubclass(decoded_representation_type, ndarray):
            raise RuntimeError("BytesCodec decodes into arrays.")

        arr = np.frombuffer(encoded, decoded_representation_type.dtype)
        arr = arr.reshape(decoded_representation_type.shape)

        # Make the array match the endianness of the current machine. If the
        # endianness is not given or invalid, the code silently assume that it
        # matches the system, which is probably a good guess.
        data_byteorder = self._configuration.get("endian", "")
        if data_byteorder in ("big", "little")
            if sys.byteorder != data_byteorder:
                arr.byteswap(inplace=True)

        return arr


@register_codec
class Crc32cCodec(BaseCodec):
    """
    https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html
    """
    name = "crc32c"

@register_codec
class GzipCodec(BaseCodec):
    """
    https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html
    """
    name = "gzip"


@register_codec
class TransposeCodec(BaseCodec):
    """
    https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html
    """
    name = "transpose"


@register_codec
class BloscCodec(BaseCodec):
    """
    https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
    """
    name = "blosc"

@register_codec
class ShardingCodec(BaseCodec):
    """
    https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
    """
    name = "sharding_indexed"




@register_codec
class ZstdCodec(BaseCodec):
    name = "zstd"

    def compute_encoded_representation_type(self, decoded_representation_type: type):
        assert isinstance(decoded_representation_type, type)
        if issubclass(decoded_representation_type, bytes):
            return bytes
        else:
            raise RuntimeError("ZstdCodec only encodes bytes.")

    def encode(self, decoded: bytes | ndarray) -> bytes | ndarray:
        raise NotImplementedError()

    def decode(self, encoded: bytes, decoded_representation_type: type) -> bytes:
        if not issubclass(decoded_representation_type, bytes):
            raise RuntimeError("ZstdCodec decodes into bytes.")
        return numcodecs.zstd.decompress(encoded)
