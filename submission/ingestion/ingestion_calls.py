"""Count actual model-client invocations after cache lookup, without quotas."""
from collections import Counter


class IngestionModelMeter:
    def __init__(self, caller, counts):
        self.caller, self.counts = caller, counts
        self.model = getattr(caller, 'model', None)

    def __call__(self, request, timeout):
        stage = ('validation' if 'required_review_fields' in request.get('input', {}) else 'extraction')
        self.counts[stage] += 1
        self.counts['total'] += 1
        try:
            return self.caller(request, timeout)
        except Exception:
            self.counts['errors'] += 1
            raise


def new_call_counts():
    return Counter(extraction=0, validation=0, total=0, errors=0)
