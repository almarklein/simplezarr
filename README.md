# simplezarr
A simple, elegant, and efficient Zarr implementation

The `simplezarr` library provides a simple way to use Zarr files. The code
adheres close to the Zarr 3.0 standard. Extra functionality is provided in the form of functions in `simplezarr.utils`.

Compared to zarr-py, simplezarr keeps the code simple, providing easy access to data and metadata. This makes it easy
to adopt in various use-cases. Although it may feel a bit more low-level, it is more flexible this way.

SimpleZarr makes no attempt to parallelize reads, and therefore is (perhaps surprisingly) faster than zarr-py.
You can still parellize reads by using e.g. a `ThreadPoolExecutor` in your code.
