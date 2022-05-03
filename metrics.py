import numpy as np


def dcg_score(gains):
    return sum([g / np.log2(i + 2) for i, g in enumerate(gains)])


def ndcg(gains, at=5):
    assert len(gains) >= at, f"Trying to calculate NDSG@{at} while having {len(gains)} objects"
    dcg = dcg_score(gains[:at])
    idcg = dcg_score(sorted(gains, reverse=True)[:at])
    if idcg == 0.:
        return 0
    else:
        return dcg / idcg
