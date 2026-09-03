"""candidates.py -- split the classifier's one label into two views.

The vendored classifier (``app/core/classifier.py``, a verbatim copy of
scoring-functions' ``src/classifier.py``) returns ONE label per mode: a slot
name for a mode that won an external slot and passed the purity gates
("Tx"), that name starred if it failed them ("Tx*"), or a Step-4 stretch/bend
label ("S"/"B"). A mode that took a slot therefore never showed its
stretch/bend character at all.

That conflates two questions which are only sometimes the same one:

  1. Is this mode a stretch or a bend?  -- answerable for EVERY mode, from
     s[V_S] alone, and always meaningful.
  2. Which mode best matches each rigid-body slot? -- the Step-2 assignment.
     Meaningful when real T/R modes are present (a 3N set), and NOT when they
     are absent: an EMIT set has no rigid-body modes, but the assignment is
     one-to-one over whatever pool it is given, so six modes are still named
     Tx..Rz. There they describe T/R *character*, not rigid-body motion.

So both are computed and kept apart. Every mode carries a stretch/bend label;
the assignment's picks are additionally offered as CANDIDATES, each with the
slot score that won it as a confidence, for the caller to accept or ignore.
Nothing here re-runs the classifier or second-guesses it -- the assignment and
the purity gates are exactly as they were, only presented separately.

This module deliberately holds no thresholds of its own: it reads labels the
classifier already produced and calls its ``vib_label`` for the S/B split, so
a recalibration there flows through with no change here.
"""

from __future__ import annotations

from .classifier import (external_axis, is_clean_external, is_external_label,
                         vib_label)


def sb_label(mode, thresholds, scheme="binary"):
    """The stretch/bend label for ANY mode, from its own s[V_S].

    Step 4 in the classifier only ever ran on modes left over after the
    external assignment; this applies the same cut to every mode, including
    the six the assignment took. Same function, same threshold -- the only
    difference is that nothing is excluded from it.
    """
    return vib_label(mode["V"], thresholds, scheme=scheme)


def candidate_of(mode):
    """The external-slot candidacy of one classified mode, or None.

    Returns the slot it was assigned ("Tx"), the score that won it (signed, so
    a mode translating along -x is not silently reported as +), the confidence
    (that score's magnitude -- what the Hungarian assignment maximised), and
    whether it passed both purity gates. `label` is the classifier's own
    verdict, kept verbatim so accepting a candidate reproduces exactly what
    the algorithm used to output for it, star and all.
    """
    label = mode["classification"]
    if not is_external_label(label):
        return None
    slot = external_axis(label)
    axis_type, axis = slot[0], slot[1].lower()
    value = mode["T" if axis_type == "T" else "R"][axis]
    return {
        "slot": slot,
        "kind": axis_type,           # "T" or "R", for grouping in the UI
        "axis": axis,
        "score": float(value),
        "confidence": abs(float(value)),
        "clean": is_clean_external(label),
        "label": label,              # "Tx" or "Tx*"
        "annotation": mode["annotation"],
    }


def split_views(scored, thresholds, scheme="binary"):
    """Annotate `scored` (classify_all_modes output) in place with both views.

    Adds to every mode:
      sb_label   -- "S"/"B" (or "SB" under scheme="threeway"), always present.
      candidate  -- the dict above, or None.

    Returns the candidate list in slot order (Tx, Ty, Tz, Rx, Ry, Rz), each
    entry carrying the index of the mode it belongs to, so a caller can render
    it as its own table without re-scanning. Slot order rather than confidence
    order: the six slots are a fixed frame, and sorting by confidence would
    reshuffle the table between molecules and hide a weak axis among strong
    ones.
    """
    out = []
    for i, m in enumerate(scored):
        m["sb_label"] = sb_label(m, thresholds, scheme)
        cand = candidate_of(m)
        m["candidate"] = cand
        if cand is not None:
            out.append(dict(cand, mode_index=i, name=m["name"]))
    out.sort(key=lambda c: _SLOT_ORDER.get(c["slot"], 99))
    return out


_SLOT_ORDER = {s: i for i, s in enumerate(("Tx", "Ty", "Tz", "Rx", "Ry", "Rz"))}
