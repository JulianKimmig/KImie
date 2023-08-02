import unicodedata
import re


def string_to_valid_filename(fn: str):
    fn = unicodedata.normalize("NFKD", fn).encode("ascii", "replace").decode("ascii")
    fn = fn.replace("\\", "_")
    fn = fn.replace("/", "_")
    fn = re.sub(r"[^\w\s.()-]", ".", fn)
    return fn
