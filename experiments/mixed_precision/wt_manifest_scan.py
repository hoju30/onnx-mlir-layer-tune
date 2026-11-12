#!/usr/bin/env python3
"""Verified scan of earlier text manifests (runs/**/*segments*.json) for article-level isolation.

Replaces run_olmoe_selective_sensitivity.used_wt103_all for new rounds. That function decided the source of a manifest only from
its 'source' / 'split' fields; runs/olmoe_act_outlier_diag_20261023/segments.json has neither, so its 64 WikiText-103 articles
were not excluded and the token-spike round drew one of them again (Brushstrokes series). The old function is left unchanged so
that the code hashes recorded by earlier rounds stay valid.

Rules (no silent skip):
- entries with content_sha256: excluded iff the hash is a WikiText-103 train article (strict boundaries); the declared label is
  only cross-checked, and a manifest whose hashes are neither all WikiText-103 nor all WikiText-2 is an error;
- entries without content_sha256: must be shown to be WikiText-2 text (title of a WikiText-2 article, or WikiText-2 character
  offsets together with a run plan naming WikiText-2); WikiText-2 articles are excluded from every WikiText-103 draw anyway
  (normalized title and content hash); anything else is an error.
"""
import glob
import json
from pathlib import Path

import run_olmoe_route_mass_restore as mr

HERE = Path(__file__).resolve().parent


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def corpora():
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    return w103, wt2


def used_wt103_verified(out, w103=None, wt2=None):
    """content hashes of every WikiText-103 article in an earlier manifest, with a per-manifest evidence report; manifests inside
    out are skipped (path containment)."""
    if w103 is None or wt2 is None:
        w103, wt2 = corpora()
    H103, H2 = {a['ch'] for a in w103}, {a['ch'] for a in wt2}
    T2 = {a['title'] for a in wt2}
    used, rows = set(), []
    for f in sorted(glob.glob(str(HERE / 'runs/**/*segments*.json'), recursive=True)):
        if out is not None and Path(f).resolve().is_relative_to(Path(out).resolve()):
            continue
        s = json.load(open(f))
        rel = str(Path(f).relative_to(HERE))
        if not isinstance(s, list) or not s or not all(isinstance(x, dict) for x in s):
            raise SystemExit(f'manifest {rel}: unexpected format')
        declared = sorted({str(x.get('source', x.get('split', '?'))) for x in s})
        hashes = [x.get('content_sha256') for x in s]
        row = {'path': rel, 'sha256': _sha(f), 'articles': len(s), 'declared': declared}
        if all(h is not None for h in hashes):
            n103, n2 = sum(h in H103 for h in hashes), sum(h in H2 for h in hashes)
            if n103 == len(s):
                row['resolved'], row['evidence'] = 'wikitext-103', 'all content hashes are WikiText-103 train articles'
                used |= set(hashes)
            elif n2 == len(s):
                row['resolved'], row['evidence'] = 'wikitext-2', 'all content hashes are WikiText-2 articles'
            else:
                raise SystemExit(f'manifest {rel}: {n103} WT103 / {n2} WT2 of {len(s)} hashes; unresolved')
            claims103 = any('wikitext-103' in d.lower() for d in declared)
            if claims103 and row['resolved'] != 'wikitext-103':
                raise SystemExit(f'manifest {rel}: declared WikiText-103 but hashes are not')
            row['declared_missing'] = declared == ['?']
        elif all(h is None for h in hashes):
            plan = Path(f).parent / 'plan.json'
            ptxt = plan.read_text() if plan.exists() else ''
            names_wt2 = 'wikitext-2' in ptxt.lower()
            ok = [x['title'] in T2 or (names_wt2 and 'article_chars' in x) for x in s]
            if not all(ok):
                raise SystemExit(f'manifest {rel}: {len(s) - sum(ok)} entries without hash not resolvable to WikiText-2')
            row['resolved'] = 'wikitext-2'
            row['evidence'] = (f'{sum(x["title"] in T2 for x in s)} titles are WikiText-2 articles; '
                               f'{sum(x["title"] not in T2 for x in s)} by WikiText-2 character offsets with the run plan naming WikiText-2'
                               f' ({"plan names WikiText-2" if names_wt2 else "plan does not name the corpus"})')
        else:
            raise SystemExit(f'manifest {rel}: mixed entries with and without content hashes')
        rows.append(row)
    return used, rows
