"""URL identity for local snapshot lookup, not redirect or authority inference."""
from urllib.parse import urlsplit, urlunsplit


def page_url(value):
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in {'http', 'https'} or not parts.hostname or parts.username or parts.password:
            return None
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                           parts.path.rstrip('/') or '/', parts.query, ''))
    except ValueError:
        return None
