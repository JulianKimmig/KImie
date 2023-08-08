import unicodedata
import re
from urllib.parse import unquote
from KImie import KIMIE_LOGGER


def string_to_valid_filename(fn: str):
    fn = unicodedata.normalize("NFKD", fn).encode("ascii", "replace").decode("ascii")
    fn = fn.replace("\\", "_")
    fn = fn.replace("/", "_")
    fn = re.sub(r"[^\w\s.()-]", ".", fn)
    return fn


def response_to_filename(response):
    # Check if "Content-Disposition" is in the response headers
    if "Content-Disposition" in response.headers:
        # Extract filename from the "Content-Disposition" header
        cd = response.headers["Content-Disposition"]
        KIMIE_LOGGER.debug(
            "use filename from Content-Disposition header: %s",
            cd,
        )
        fname = re.findall(r"filename\*?=([^;]+)", cd)
        # If a filename is found in the header, use it
        if len(fname) > 0:
            # The filename may be URL-encoded. If so, decode it.
            # It might also be in the format "UTF-8''filename", so remove the leading charset and ''.
            fname = unquote(fname[0]).split("'")[-1]
        else:
            raise ValueError("Content-Disposition header could not be parsed")
    else:
        # If "Content-Disposition" is not in the headers, extract the filename from the URL
        fname = response.url.split("/")[-1]

    fname = string_to_valid_filename(fname)  # Sanitize the filename

    return fname
