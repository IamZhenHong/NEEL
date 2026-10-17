"""Read Inspect .eval logs (ZIP with zstd / method 93). Reading only; run with python3 -I.
Pass eval paths as args from a dir OUTSIDE the downloads (brief rule 2)."""
import zipfile, struct, io, json
import zstandard as zstd


def _read_member_raw(z, name):
    """Return decompressed bytes for a zip member, supporting zstd (method 93)."""
    zi = z.getinfo(name)
    if zi.compress_type != 93:
        return z.read(name)
    # Manually locate the local file header to find the compressed data start.
    fh = z.fp
    fh.seek(zi.header_offset)
    sig, ver, flags, method, mtime, mdate, crc, csize, usize, nlen, elen = \
        struct.unpack("<IHHHHHIIIHH", fh.read(30))
    fh.seek(zi.header_offset + 30 + nlen + elen)
    comp = fh.read(zi.compress_size)
    return zstd.ZstdDecompressor().decompress(comp, max_output_size=max(zi.file_size, 1) * 2 or 1 << 28)


def load_json(z, name):
    return json.loads(_read_member_raw(z, name))


def open_eval(path):
    return zipfile.ZipFile(path)


def sample_names(z):
    return sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json"))


def header(z):
    return load_json(z, "header.json")
