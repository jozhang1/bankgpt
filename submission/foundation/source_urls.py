"""URL identity for local snapshot lookup, not redirect or authority inference."""
from datetime import date
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


def latest_snapshot_by_url(snapshots, as_of=None):
    """Select the newest captured version of each URL up to the query date.

    A URL missing from a later snapshot is not treated as a deletion. A newer
    capture of that same URL replaces its earlier version.
    """
    cutoff = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    cutoff = cutoff or date.max
    latest = {}
    for snapshot in snapshots:
        taken_on = date.fromisoformat(snapshot.taken_on)
        if taken_on > cutoff:
            continue
        for page in snapshot.pages:
            key = page_url(page.url)
            if key is None:
                continue
            candidate = (taken_on, snapshot.snapshot_id)
            if key not in latest or candidate > latest[key][0]:
                latest[key] = (candidate, snapshot.snapshot_id)
    return {url: snapshot_id for url, (_, snapshot_id) in latest.items()}
