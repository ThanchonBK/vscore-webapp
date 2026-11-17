"""pipeline.py -- parse -> align -> score -> classify, in one call.

Mirrors ``main.build_scorer_and_final`` + ``classify_all_modes`` from the
reference implementation so this app and the CLI agree by construction. The
whole thing runs in ~11 ms for naphthalene (18 atoms, 54 modes), which is why
there is no job queue, no polling and no server-side run state anywhere in
this app.

Everything the browser needs -- table rows, the 3Dmol payload, provenance --
comes out of :func:`analyse` as one JSON-safe dict.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import frame as frame_mod
from .candidates import split_views
from .classifier import (Thresholds, classify_all_modes, external_slots, is_linear,
                         predicted_category, round_score)
from .parsers import ParseError, displacement_precision
from .scoring import ModeScorer

SCORE_KEYS = ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "V_S")

# The s[V_S] range inside which an S/B label is flagged as not interpretable
# on its own (JCC revision, Reviewer 1 item 11). This is the paper's own range:
# the 95 % interval of the optimal tau_SB over 10^5 bootstrap replicates,
# [0.315, 0.525], rounded half up to 0.01 -- scoring-functions
# src/figures.py AMBIGUITY_RANGE (commit 04787a6). It is used whenever no
# delta is given; a user delta replaces it with the symmetric tau_SB +/- delta.
AMBIGUITY_RANGE = (0.32, 0.53)
# What the delta field starts at on the page when a user wants a symmetric band.
DEFAULT_DELTA = 0.10

# Frame of reference for the T/R scores (scoring-functions src/frame.py, the
# R1.5 rule). "symmetry" is upstream's default; "eigh" is the submitted
# version's inertia-eigenvector frame (MIT), kept for comparison.
FRAME_RULES = frame_mod.FRAME_RULES          # ("symmetry", "eigh")
TIE_BREAKS = frame_mod.TIE_BREAKS            # ("tmax", "gap")
LINEAR_TURNS = frame_mod.LINEAR_RULES        # ("vector", "scan")

# Two modes are taken as degenerate when their frequencies (or EMIT
# eigenvalues) differ by no more than this, in the file's own units. Symmetry-
# degenerate pairs in the reference set agree EXACTLY (benzene, adamantane,
# CO2, the benzene EMIT set); the closest accidental pair is naphthalene's at
# 0.25 cm-1, which is not a degeneracy and must not be flagged as one.
DEFAULT_DEG_TOL = 0.1

_EXTERNAL_SLOTS = {"Tx", "Ty", "Tz", "Rx", "Ry", "Rz"}

# Labels produced by Step 4 of the classifier, with the plain-language reading
# the UI shows. External slots (Tx..Rz) are described in place.
LABEL_TEXT = {
    "S": "stretching",
    "B": "bending",
    "SB": "mixed stretch/bend",
}


def _rotation_between(a, b):
    """Angle in degrees of the rigid rotation taking geometry `a` onto `b`.

    Both are COM-centred, so the optimal rotation is the Kabsch one; the angle
    comes from its trace. Used only to report how far the principal-axis
    alignment turned the molecule.
    """
    if len(a) < 2:
        return 0.0
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    cos = (np.trace(r) - 1.0) / 2.0
    return round(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), 1)


def _detect_mode_set(n_modes, n_atoms, n_ext, linear):
    """Work out from the counts alone what kind of mode set this is.

    "3n"         every mode, translations and rotations included.
    "3n-6"       vibrations only (3N-5 when linear) -- a complete set.
    "arbitrary"  any other count: a hand-picked selection of modes. Treated as
                 vibrations, with the ideal T/R references still constructed so
                 the external slots are filled by them and none of the supplied
                 modes can be claimed for one.

    The first two are exact and unambiguous; anything else falls to the third
    rather than being refused, because scoring a few modes of interest is a
    legitimate thing to want.
    """
    full = 3 * n_atoms
    if n_modes == full:
        return "3n"
    if n_modes == full - n_ext:
        return "3n-6"
    return "arbitrary"


def analyse(atoms, coords, bonds, modes, title="", source="", warnings=None,
            mode_set="auto", tau_sb=None, delta=None,
            deg_tol=DEFAULT_DEG_TOL, reject=(), frame_rule="symmetry",
            keep_input_frame=False, tie_break="tmax", linear_turn="vector",
            sym_tol=None, max_scan_work=None):
    """Score and classify one molecule. Returns a JSON-safe payload.

    ``coords``/``modes`` are taken in whatever frame they arrived in; both are
    turned together into the principal-axis frame the T/R scores are defined
    in, which is also the frame the viewer shows. Which principal-axis frame is
    the job of ``frame_rule`` (see below).

    ``mode_set`` says what the supplied modes are, which decides whether the
    ideal T/R references are added to the pool:

    "3n-6"  vibrations only (3N-6, or 3N-5 linear). The six external slots have
            no genuine candidates among the supplied modes, so the constructed
            ideal T/R modes are prepended to fill them. Without that, Step 2's
            assignment claims real vibrations instead -- on H2O it labelled a
            0.997 O-H stretch "Tx*".

    "3n"    the complete 3N set, T/R included. The external slots are filled
            from the supplied modes themselves and nothing is constructed, so
            the six T/R modes are identified by the algorithm rather than
            supplied alongside it.

    ``tau_sb`` overrides the calibrated S/B cut point (None keeps it).
    ``delta`` sets the interpretability band to tau_SB +/- delta; None uses the
    paper's ambiguity range (AMBIGUITY_RANGE). A mode inside it is flagged as
    borderline. ``deg_tol`` groups
    modes whose frequency/eigenvalue agree to within it as one degenerate set;
    every mode must carry one for that check to run. ``reject`` lists 1-based
    numbers of the user's modes that may NOT take a T/R slot -- the assignment
    is re-solved without them, so each slot falls to its next-best mode.

    Frame (scoring-functions src/frame.py; for a symmetric or spherical top or
    a linear molecule the principal axes are not unique, so the T/R scores
    need a rule to be reproducible):
    ``frame_rule``  "symmetry" (default, the R1.5 rule) or "eigh" (the
                    submitted version's MIT frame).
    ``keep_input_frame``  keep the input orientation when the rule allows it.
    ``tie_break``   step 2 of the symmetry rule: "tmax" (default) or "gap".
    ``linear_turn`` turn of a linear molecule about its axis: "vector"
                    (default) or "scan".
    ``sym_tol``     symmetry tolerance in Angstrom (None = frame.SYM_TOL).
    ``max_scan_work`` refuse the linear "scan" turn when modes x atoms exceeds
                    this (None = no limit). The scan rescores every mode at 900
                    turns: a 30-atom chain took 10 s, past a serverless time
                    limit, so the deployment sets a cap; localhost does not.
    Every option is passed per call; frame.py's module-level setters are never
    used, so one request cannot change another's frame.
    """
    warnings = list(warnings or [])
    if delta is not None:
        delta = _check_range("δ", delta, 0.0, 1.0)
    for what, val, allowed in (("frame rule", frame_rule, FRAME_RULES),
                               ("tie-break", tie_break, TIE_BREAKS),
                               ("linear turn", linear_turn, LINEAR_TURNS)):
        if val not in allowed:
            raise ParseError(f"The {what} must be one of {', '.join(allowed)}; got {val!r}.")
    if sym_tol is not None:
        sym_tol = _check_range("The symmetry tolerance", sym_tol, 0.0, 1.0)
        if sym_tol == 0:
            raise ParseError("The symmetry tolerance must be greater than 0.")
    deg_tol = _check_range("the degeneracy tolerance", deg_tol, 0.0, None)
    notes = []                      # informational; not a problem with the input
    if mode_set not in ("auto", "3n", "3n-6", "arbitrary"):
        raise ValueError("mode_set must be 'auto', '3n', '3n-6' or "
                         f"'arbitrary', got {mode_set!r}")

    if not bonds:
        # Belt and braces: every reader already refuses this, because an empty
        # bond list makes Vscore() return exactly 0.0 for every mode, and
        # 0.0 <= tau_B, so the whole table would come back labelled "B".
        raise ParseError(
            "No bond connectivity available. The V-score is a sum over bonds; "
            "without them every mode would score 0.000 and be labelled bending.")

    thresholds = Thresholds.calibrated()
    tau_sb_calibrated = thresholds.tau_SB
    if tau_sb is not None:
        thresholds = replace(thresholds,
                             tau_SB=_check_range("τ_SB", tau_sb, 0.0, 1.0))
    scorer = ModeScorer(atoms, coords, bonds)
    _as_given = np.array(scorer.coords, dtype=float)   # COM-shifted, not yet aligned

    # EMIT eigenvectors are stored in MIT's frame of the input geometry, NOT the
    # input frame: rotating them as if they were applies the transform twice
    # (done from a raw Gaussian orientation, that mislabels 9 of benzene's 36
    # EMIT modes). Gaussian normal modes are printed in the geometry's own
    # frame. `eigen=` in a .vsc is what sets is_emit. Mirrors scoring-functions
    # main.build_scorer_and_final, both branches.
    is_emit = any(m.get("is_emit") for m in modes)
    if (frame_rule == "symmetry" and linear_turn == "scan" and max_scan_work
            and len(modes) * len(atoms) > max_scan_work and is_linear(scorer)):
        raise ParseError(
            f"The 'scan' linear turn rescores all {len(modes)} modes at 900 angles, "
            "which is too slow for this server at this size. Use the default "
            "'vector' turn, or run the app locally for 'scan'.")
    if frame_rule == "eigh":
        rotated = scorer.MIT([dict(m) for m in modes], rotate_modes=not is_emit)
        frame_info = dict(kind="", n=0, evaluations=0, values=[],
                          step="eigh (submitted version)")
    else:
        if is_emit:
            # back to the input frame first: d_in = d_MIT R^T
            R = scorer.mit_rotation_matrix()
            vecs_in = [np.asarray(m["vector"], dtype=float) @ R.T for m in modes]
        else:
            vecs_in = [np.asarray(m["vector"], dtype=float) for m in modes]
        try:
            F, frame_info = frame_mod.select_frame_for_scorer(
                scorer, vecs_in, keep_input_frame=bool(keep_input_frame),
                tol=sym_tol, tie_break=tie_break, linear_rule=linear_turn)
        except Exception as exc:             # a frame-rule failure, said plainly
            raise ParseError(
                f"The symmetry frame rule could not pick a frame ({exc}). Try "
                "the submitted (eigh) frame, or a larger symmetry tolerance if "
                "the coordinates carry fewer than 5 decimals.")
        scorer.apply_frame(F)
        rotated = []
        for m, v in zip(modes, vecs_in):
            nm = dict(m)
            nm["vector"] = v @ F.T
            rotated.append(nm)
    frame_rotation = _rotation_between(_as_given, np.array(scorer.coords, dtype=float))
    # Self-assign the name rather than trusting one to survive parsing:
    # write_vsc/parse_vsc do not round-trip 'label', so every mode would
    # otherwise fall back to classify_all_modes()'s generic "Mode i".
    for i, m in enumerate(rotated):
        m["label"] = f"EMIT {i + 1}" if is_emit else f"Vib {i + 1}"

    linear = is_linear(scorer)
    ideal_R = scorer.construct_R()
    if linear:
        # Both frame rules place the linear axis on x, so construct_R()'s "Rx" is an
        # all-zero placeholder rather than a real external mode. Left in, it
        # falls through to Step 4 and is mislabelled BENDING (V=0 <= tau_B).
        ideal_R = [m for m in ideal_R if m["label"] != "Rx"]

    n_ext = 3 + len(ideal_R)                 # external slots: 3 T + 3 R (2 linear)
    detected = _detect_mode_set(len(modes), len(atoms), n_ext, linear)
    if mode_set == "auto":
        mode_set = detected
    elif mode_set == "arbitrary":
        # A declaration, not a detection: the caller is asserting these are
        # vibrations. Honoured whatever the count, so a selection that happens
        # to number 3N or 3N-6 is still given the constructed T/R references
        # rather than being read as a complete set.
        pass
    elif mode_set != detected:
        raise ParseError(
            f"This file holds {len(modes)} modes, i.e. {detected}, "
            f"but {mode_set} was requested.")

    if mode_set == "3n":
        final = rotated                      # nothing constructed; T/R must be in here
        n_ref = 0
    else:
        final = scorer.construct_T() + ideal_R + rotated
        n_ref = len(scorer.construct_T()) + len(ideal_R)

    scored = classify_all_modes(scorer, final, thresholds)

    # Rejected T/R winners. The vendored classifier takes no exclusions, so the
    # one-to-one assignment is re-solved here over the remaining modes with the
    # same cost (|slot score|) and the same solver, and its verdict replaces
    # the classifier's. A rejected mode then reads as its own S/B label.
    _, _, slots = external_slots(scorer)
    if reject and n_ref:
        raise ParseError(
            "Nothing to reject: with a vibrations-only set the constructed T/R "
            "references fill every slot, so none of your modes holds one.")
    rejected = _rejected_indices(reject, len(rotated), n_ref, len(slots), len(final))
    if rejected:
        assign = _assign(scored, slots, rejected)
        for i, m in enumerate(scored):
            slot = assign.get(i)
            m["tr_label"] = slot
            m["tr_score"] = _slot_score(m, slot) if slot else None

    # The classifier reports its two questions separately -- vib_label (S/B,
    # every mode) and tr_label (the Hungarian slot a mode won, or None). The
    # views below read the combined verdict, which is the classifier's own
    # rule: the slot if the mode won one, else its vib_label. There is no
    # purity gate and so nothing to annotate.
    for m in scored:
        m["classification"] = predicted_category(m["tr_label"], m["vib_label"])
        m["annotation"] = ""
    # Split the classifier's single verdict into the two views the UI offers:
    # a stretch/bend label on EVERY mode, and the external assignment's picks
    # held aside as candidates. Neither re-runs the classifier -- see
    # candidates.py for why the two are worth separating.
    candidates = split_views(scored, thresholds)
    # classify_all_modes returns one entry per mode in `final`, in the same
    # order, but drops the displacement vector -- the viewer needs it, so pair
    # them back up here rather than re-deriving.
    # With constructed references, POSITION identifies them (they win their own
    # slot every time -- 0/463 counterexamples). Under 3N nothing is
    # constructed, so the CLASSIFICATION is what says a mode is external.
    rows = [_row(m, vec["vector"], i,
                 constructed=bool(n_ref) and i < n_ref,
                 is_reference=(i < n_ref if n_ref
                               else m["classification"] in _EXTERNAL_SLOTS),
                 # the SET-level flag, not the per-mode one: is_emit decides
                 # rotation for the whole pool, and a per-row mix would name the
                 # CSV column "Eigenvalue" on some rows and "Freq" on others,
                 # which DictWriter rejects outright.
                 is_emit=is_emit)
            for i, (m, vec) in enumerate(zip(scored, final))]
    references, vibrations = rows[:n_ref], rows[n_ref:]

    # ---- interpretability band around tau_SB ------------------------------
    # The S/B label is a hard cut, so a mode a hair either side of tau_SB gets
    # a confident-looking letter it has not earned. Flag those rather than
    # move them: the label still follows the cut, the flag says not to lean
    # on it. Constructed references are rigid-body motions, never S/B.
    tau = thresholds.tau_SB
    tau_overridden = thresholds.tau_SB != tau_sb_calibrated
    if delta is None and tau_overridden:
        # The paper's range is the bootstrap spread of the CALIBRATED tau_SB;
        # around any other cut it would flag modes nowhere near the cut. Fall
        # back to a symmetric band around the user's tau and say so.
        delta = DEFAULT_DELTA
        notes.append(
            f"τ_SB was set to {tau:.2f}, so the paper's ambiguity range "
            f"({AMBIGUITY_RANGE[0]:.2f}–{AMBIGUITY_RANGE[1]:.2f}, which belongs to "
            f"the calibrated {tau_sb_calibrated:.2f}) does not apply; borderline "
            f"modes are flagged within δ = {DEFAULT_DELTA:.2f} of {tau:.2f} instead.")
    if delta is None:
        amb_lo, amb_hi = AMBIGUITY_RANGE
        band = (f"the ambiguity range {amb_lo:.2f}–{amb_hi:.2f} (95 % bootstrap "
                "interval of τ_SB)")
    else:
        amb_lo, amb_hi = tau - delta, tau + delta
        band = f"δ = {delta:.2f} of τ_SB = {tau:.2f}"
    for i, r in enumerate(rows):
        v = float(scored[i]["V"])
        # The label compares s[V_S] ROUNDED half up to 0.01 with tau_SB
        # (classifier.round_score, upstream a140090), so the margin is taken on
        # that same rounded score -- on the raw one, adamantane Vib 22
        # (0.4256 -> 0.43, labelled S) showed a negative margin.
        r["sb_margin"] = round(float(round_score(v)) - tau, 4)
        # a small guard so 0.53 typed as the bound includes a score printed 0.53
        r["sb_ambiguous"] = bool(not r["constructed"]
                                 and amb_lo - 1e-9 <= v <= amb_hi + 1e-9)
    ambiguous = [r for r in vibrations if r["sb_ambiguous"]]
    if ambiguous:
        warnings.append(
            f"{len(ambiguous)} mode{'s' if len(ambiguous) != 1 else ''} "
            f"ha{'ve' if len(ambiguous) != 1 else 's'} s[V_S] within {band} "
            f"({_name_list(ambiguous)}). Their S/B label is set by a cut the "
            "score barely clears, so it is not interpretable on its own — read "
            "these as borderline stretch/bend, marked ? in the table.")

    # ---- degeneracy -------------------------------------------------------
    # Degenerate modes can only be recognised from equal eigenvalues, so the
    # check needs one on every mode it covers. A 3N file may leave its six
    # rigid-body modes without a frequency -- they are not vibrations, and a
    # degeneracy among them says nothing about a stretch or a bend -- so only
    # a mode that is otherwise read as a vibration counts as missing.
    degenerate_sets = []
    for r in rows:
        r["degenerate_set"] = None
    valued = [r for r in vibrations if r["frequency"] is not None]
    missing = [r for r in vibrations if r["frequency"] is None
               and not (mode_set == "3n" and r["ext_label"] in _EXTERNAL_SLOTS)]
    if missing:
        what = "eigenvalue" if is_emit else "frequency"
        warnings.append(
            f"Degeneracy was not checked: {len(missing)} of {len(vibrations)} "
            f"modes carry no {what} ({_name_list(missing)}). Degenerate modes "
            f"can only be recognised from equal {what}s, so add "
            f"{'eigen=' if is_emit else 'freq='} to every mode to have them "
            "flagged. Within a degenerate set the individual modes are not "
            "unique, and neither are their scores.")
    else:
        degenerate_sets = _degenerate_sets(valued, deg_tol)
        for k, ds in enumerate(degenerate_sets, 1):
            ds["id"] = k
            for idx in ds["members"]:
                rows[idx]["degenerate_set"] = k
        if degenerate_sets:
            unit = "" if is_emit else " cm⁻¹"
            notes.append(
                f"{len(degenerate_sets)} degenerate set"
                f"{'s' if len(degenerate_sets) != 1 else ''} "
                f"({'eigenvalues' if is_emit else 'frequencies'} within "
                f"{deg_tol:g}{unit}), tagged d1, d2, … in the table. Any "
                "orthogonal mix of the modes in a set is an equally valid set of "
                "normal modes, so the scores of the individual members depend on "
                "an arbitrary choice the program made — compare sets as a whole.")
            # One warning for all affected sets, not one box per set: benzene
            # EMIT produced five boxes that pushed the table off the screen.
            flagged = []
            for ds in degenerate_sets:
                labs = sorted({rows[i]["label"] for i in ds["members"]})
                amb = [rows[i] for i in ds["members"] if rows[i]["sb_ambiguous"]]
                if len(labs) > 1 or amb:
                    why = (f"members labelled {' and '.join(labs)}"
                           if len(labs) > 1 else
                           f"{_name_list(amb)} inside the ambiguity band")
                    ds["concern"] = why
                    flagged.append(
                        f"d{ds['id']} ({ds['names']}, {ds['frequency']:.2f}{unit}): {why}")
            if flagged:
                degeneracy_warning = (
                    f"{len(flagged)} degenerate set{'s' if len(flagged) != 1 else ''} "
                    "could be labelled differently by an equally valid choice of "
                    "modes within the set: " + "; ".join(flagged) + ".")
                warnings.append(degeneracy_warning)

    # ---- T/R candidates: who takes the slot if the winner is rejected ------
    user = {i for i in range(n_ref, len(final)) if i not in rejected}
    for c in candidates:
        if c["mode_index"] < n_ref:
            continue
        c["number"] = c["mode_index"] - n_ref + 1
        alt = _assign(scored, slots, rejected | {c["mode_index"]})
        nxt = next((i for i, s in alt.items() if s == c["slot"]), None)
        c["runner_up"] = None if nxt is None else {
            "mode_index": nxt, "number": nxt - n_ref + 1 if nxt >= n_ref else None,
            "name": scored[nxt]["name"],
            "score": round(float(_slot_score(scored[nxt], c["slot"])), 4),
            "related_score": round(abs(float(_slot_score(scored[nxt], c["slot"]))), 4),
            "constructed": nxt < n_ref,
        }
        # Rejecting one winner can reshuffle other slots too (the assignment is
        # one-to-one), so say which -- the user is about to accept all of it.
        base = {i: s for i, s in enumerate(m["tr_label"] for m in scored) if s}
        c["also_moves"] = [
            {"slot": s, "from": scored[i]["name"],
             "to": next((scored[j]["name"] for j, t in alt.items() if t == s), None)}
            for i, s in sorted(base.items(), key=lambda kv: _SLOT_RANK[kv[1]])
            if s != c["slot"] and alt.get(i) != s]
        # The slot's own column, ranked: what "second best" means before the
        # one-to-one constraint is applied.
        ranked = sorted(user, key=lambda i: -abs(_slot_score(scored[i], c["slot"])))[:3]
        c["ranking"] = [{"name": scored[i]["name"],
                         "related_score": round(abs(float(_slot_score(scored[i], c["slot"]))), 4)}
                        for i in ranked]
    rejected_rows = [{"number": i - n_ref + 1, "name": scored[i]["name"],
                      "label": rows[i]["label"]} for i in sorted(rejected)]

    # An arbitrary selection is declared to be vibrations, so the constructed
    # T/R modes are pure machinery here: they exist to occupy the six external
    # slots and stop a supplied mode being claimed for one. They are not
    # results, so they are not reported and the page offers no toggle for them.
    # They stay in the pool above -- dropping them from the CLASSIFICATION is
    # what would break it (on H2O that labelled a 0.997 O-H stretch "Tx*").
    if mode_set == "arbitrary":
        references = []

    if mode_set == "arbitrary":
        shape = "3N-5" if linear else "3N-6"
        how = ("taken as vibrations" if detected == "arbitrary"
               else f"treated as vibrations even though the count is {detected}")
        notes.append(
            f"{len(vibrations)} mode{'s' if len(vibrations) != 1 else ''} "
            f"{how} (3N = {3 * len(atoms)}, {shape} = {3 * len(atoms) - n_ext}). "
            "The ideal T/R references were constructed and fill every external "
            "slot, so none of your modes is offered as a T/R candidate — each "
            "is classified on its own s[V_S].")

    if mode_set == "3n":
        # Read the CLASSIFIER's verdict, not the displayed label: the default
        # view now calls every mode a stretch or a bend, so counting "TR" in
        # `label` would report zero assigned slots on a perfectly good 3N set.
        # No purity gate here (see classifier.py's Step 3 docstring), so
        # every Step-2 winner is a bare slot name -- there is no
        # starred/mixed-external case left to report.
        assigned = [r for r in vibrations if r["ext_label"][0] in "TR"]
        if len(assigned) < n_ext:
            warnings.append(
                f"Only {len(assigned)} of the {n_ext} external slots were "
                "assigned, so the file may not hold a complete 3N mode set.")

    dp = displacement_precision(modes)
    if dp <= 2:
        warnings.append(
            f"Displacements carry only {dp} decimal places, which is standard "
            "Gaussian output rather than freq=hpmodes. The V-score and the "
            "S/B/SB labels are robust to this, but individual T-scores can shift "
            "by up to ~0.3 because Tscore() unit-normalises each atom's "
            "displacement. Re-run with freq=hpmodes for publication numbers.")

    return {
        "title": title or "molecule",
        "source": source,
        "n_atoms": len(atoms),
        "n_modes": len(vibrations),
        "linear": bool(linear),
        "atoms": list(atoms),
        "bonds": [[int(a), int(b)] for a, b in bonds],
        "bond_labels": [f"{atoms[a]}{a+1}-{atoms[b]}{b+1}" for a, b in bonds],
        "geometry": np.asarray(scorer.coords).round(6).tolist(),
        # Which frame the T/R scores are in, and how it was chosen.
        "frame": {
            "rule": frame_rule,
            "keep_input_frame": bool(keep_input_frame),
            "tie_break": tie_break,
            "linear_turn": linear_turn,
            "sym_tol": sym_tol if sym_tol is not None else frame_mod.SYM_TOL,
            "kind": frame_info.get("kind") or "",
            "step": frame_info.get("step") or "",
            "candidates": len(frame_info.get("values") or []),
        },
        # How far the frame turned the molecule from the frame it arrived in. The
        # scores -- and the axes the viewer draws -- are defined in the rotated
        # frame, so a large angle here is why the picture looks turned relative
        # to the input deck. It is not a disagreement between the two.
        "frame_rotation_deg": frame_rotation,
        "vibrations": vibrations,
        "references": references,
        # The external assignment's picks, held aside. Constructed T/R
        # references are dropped: they exist precisely to win these slots, so
        # listing them as "candidates" would be circular -- the question is
        # only ever which of the USER's modes look external.
        "candidates": [c for c in candidates if c["mode_index"] >= n_ref],
        "thresholds": {
            "tau_TR": thresholds.tau_TR,
            "tau_S": thresholds.tau_S,
            "tau_B": thresholds.tau_B,
            # The operative cut point under the binary scheme the classifier
            # runs: tau_SB splits S from B. tau_S/tau_B are kept for the
            # three-way scheme; tau_TR gates nothing (diagnostic only).
            "tau_SB": thresholds.tau_SB,
            "tau_SB_calibrated": tau_sb_calibrated,
            "tau_SB_overridden": thresholds.tau_SB != tau_sb_calibrated,
            "delta": delta,
            "ambiguity_range": [round(amb_lo, 4), round(amb_hi, 4)],
            "ambiguity_from": "paper" if delta is None else "delta",
            "deg_tol": deg_tol,
            "v_weighting": thresholds.v_weighting,
        },
        "degenerate_sets": degenerate_sets,
        "degeneracy_checked": not missing,
        "rejected": rejected_rows,
        "precision_dp": dp,
        "mode_set": mode_set,
        "is_emit": bool(is_emit),
        "warnings": warnings,
        "notes": notes,
        "freq_range": _freq_range(vibrations),
    }


def _row(m, vector, index, is_reference=False, is_emit=False, constructed=False):
    """One table row: the seven scores, the label, and which score dominates."""
    scores = {
        "Tx": m["T"]["x"], "Ty": m["T"]["y"], "Tz": m["T"]["z"],
        "Rx": m["R"]["x"], "Ry": m["R"]["y"], "Rz": m["R"]["z"],
        "V_S": m["V"],
    }
    # Largest |value|, not largest value: s[V_S] alone cannot be negative, so a
    # raw comparison would structurally favour it and would pick +0.087 over
    # -0.283. A mode translating along -x is as translational as one along +x.
    dominant = max(SCORE_KEYS, key=lambda k: abs(scores[k]))

    # Two views of the same mode (see candidates.py):
    #   label     -- the stretch/bend one, true of every mode, shown by default
    #   ext_label -- the classifier's own verdict, shown when the user accepts
    #                this mode's T/R candidacy. Identical to `label` for a mode
    #                that never took a slot.
    # A CONSTRUCTED ideal T/R reference is a rigid-body motion by definition --
    # it exists only to occupy an external slot -- so calling it a stretch or a
    # bend would be false. It keeps the classifier's label in both views; the
    # split applies to the user's own modes, which are what the question is
    # about.
    label = m["classification"] if constructed else m["sb_label"]
    ext_label = m["classification"]

    # What the UI highlights follows the LABEL, not the raw abs-max, because
    # the label is what the algorithm actually concluded. Computed for BOTH
    # views: the default one labels every mode from s[V_S] and so always marks
    # it, and the accepted-candidate view marks by the rules below.
    #
    #   external (Tx..Rz)  -> that slot's score. No purity gate (see
    #                         classifier.py's Step 3 docstring), so this is
    #                         every Step-2 winner, not just a "clean" subset.
    #   internal (S/B/SB)  -> s[V_S], the score the label is derived from.
    #                         The T/R columns are diagnostics here, and
    #                         highlighting an abs-max that lands there (43%
    #                         of real modes) would assert external character
    #                         the algorithm never assigned.
    def _highlights_for(lab):
        if lab[0] in "TR":
            return [lab]
        return ["V_S"]

    # The default view labels every mode from s[V_S], so that is what it marks.
    highlights = _highlights_for(label)
    highlight = highlights[0]
    ext_highlights = _highlights_for(ext_label)
    return {
        "index": index,
        "name": m["name"],
        "is_reference": bool(is_reference),
        "constructed": bool(constructed),
        "is_emit": bool(is_emit),
        "highlight": highlight,
        "highlights": highlights,
        "frequency": _num(m["frequency"]),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "dominant": dominant,
        "dominant_value": round(float(scores[dominant]), 4),
        "label": label,
        "label_text": _label_text(label),
        "annotation": "",
        "ext_label": ext_label,
        "ext_label_text": _label_text(ext_label),
        # Always "" -- classifier.py's Step 3 has no purity gate to annotate
        # a failure of. Kept for shape parity with the JS/CSV consumers.
        "ext_annotation": m["annotation"],
        "ext_highlight": ext_highlights[0],
        "ext_highlights": ext_highlights,
        "candidate": m["candidate"],
        "irrep": m["irrep"],
        "reduced_mass": _num(m["reduced_mass"]),
        "force_constant": _num(m["force_constant"]),
        "bonds": [{"pair": f"{b['i_label']}-{b['j_label']}",
                   "s_AB": round(float(b["s_AB"]), 4),
                   "rel_db": round(float(b["rel_db"]), 5)} for b in m["bonds"]],
        "vector": np.asarray(vector).round(5).tolist(),
    }


_SLOT_RANK = {s: i for i, s in enumerate(("Tx", "Ty", "Tz", "Rx", "Ry", "Rz"))}


def _check_range(what, value, lo, hi):
    """A user-supplied number, or a ParseError saying what was wrong with it."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ParseError(f"{what} must be a number, got {value!r}.")
    if not np.isfinite(v) or v < lo or (hi is not None and v > hi):
        rng = f"between {lo:g} and {hi:g}" if hi is not None else f"at least {lo:g}"
        raise ParseError(f"{what} must be {rng}, got {v:g}.")
    return v


def _rejected_indices(reject, n_user, n_ref, n_slots, n_pool):
    """1-based user mode numbers -> indices into the classifier's pool."""
    out = set()
    for k in reject or ():
        try:
            k = int(k)
        except (TypeError, ValueError):
            raise ParseError(f"Rejected mode {k!r} is not a mode number.")
        if not 1 <= k <= n_user:
            raise ParseError(f"Rejected mode {k} does not exist (modes 1–{n_user}).")
        out.add(n_ref + k - 1)
    if n_pool - len(out) < n_slots:
        raise ParseError(
            f"Rejecting {len(out)} modes leaves {n_pool - len(out)}, too few to "
            f"fill the {n_slots} T/R slots.")
    return out


def _slot_score(m, slot):
    return m["T" if slot[0] == "T" else "R"][slot[1].lower()]


def _assign(scored, slots, excluded=()):
    """The classifier's Step-3 assignment, re-solved without `excluded` modes.

    Same cost (|slot score|) and same solver as classify_all_modes, so with
    nothing excluded it reproduces the classifier's own verdict exactly.
    Returns {pool index: slot}.
    """
    cols = [i for i in range(len(scored)) if i not in excluded]
    cost = np.array([[abs(_slot_score(scored[i], s)) for i in cols] for s in slots])
    rows, picks = linear_sum_assignment(cost, maximize=True)
    return {cols[c]: slots[r] for r, c in zip(rows, picks)}


def _degenerate_sets(vibrations, tol):
    """Group modes whose frequency/eigenvalue agree to within `tol`.

    Chained on the sorted values, so a set is a run of neighbours each within
    tol of the next. Returns only sets of two or more, in frequency order.
    """
    order = sorted(vibrations, key=lambda r: r["frequency"])
    groups, cur = [], order[:1]
    for prev, r in zip(order, order[1:]):
        if abs(r["frequency"] - prev["frequency"]) <= tol:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        if len(g) < 2:
            continue
        g = sorted(g, key=lambda r: r["index"])
        out.append({"members": [r["index"] for r in g],
                    "names": ", ".join(r["name"] for r in g),
                    "n": len(g),
                    "frequency": float(np.mean([r["frequency"] for r in g]))})
    return out


def _name_list(rows, most=6):
    names = [r["name"] for r in rows]
    return ", ".join(names[:most]) + (f" and {len(names) - most} more"
                                      if len(names) > most else "")


def _label_text(label):
    if label in LABEL_TEXT:
        return LABEL_TEXT[label]
    return f"external: {label}"


def _num(v):
    if v is None:
        return None
    f = float(v)
    return f if np.isfinite(f) else None


def _freq_range(rows):
    freqs = [r["frequency"] for r in rows if r["frequency"] is not None]
    if not freqs:
        return [0.0, 0.0]
    return [float(min(freqs)), float(max(freqs))]


def to_csv_rows(payload, include_references=False, incorporate=False):
    """Flatten to CSV rows in the same shape as the reference implementation's
    ``classify_to_rows`` output, so a download drops straight into the existing
    analysis (one ``s_AB[X#-Y#]`` column per bond)."""
    rows = []
    source = (payload["references"] + payload["vibrations"]
              if include_references else payload["vibrations"])
    for r in source:
        row = {
            "Mode": r["name"],
            ("Eigenvalue" if r["is_emit"] else "Freq"): r["frequency"],
            "Tx": r["scores"]["Tx"], "Ty": r["scores"]["Ty"], "Tz": r["scores"]["Tz"],
            "Rx": r["scores"]["Rx"], "Ry": r["scores"]["Ry"], "Rz": r["scores"]["Rz"],
            "V_Stretch": r["scores"]["V_S"],
            "Mu": r["reduced_mass"], "K": r["force_constant"], "Irrep": r["irrep"],
            "label": r["ext_label"] if incorporate else r["label"],
            "annotation": r["ext_annotation"] if incorporate else r["annotation"],
            # Which view produced this file, recorded per row so a saved CSV
            # cannot later be mistaken for the other one. With this False, a
            # T/R candidate is reported in the two columns below and NOT in
            # the label; with it True the label is the classifier's verdict.
            "tr_candidates_incorporated": bool(incorporate),
            "tr_candidate_slot": (r["candidate"] or {}).get("slot"),
            "tr_candidate_related_score": (
                round((r["candidate"] or {}).get("related_score"), 4)
                if r["candidate"] else None),
            # highlighted_score is what the table marks: the dominant score on a
            # constructed T/R reference, and always V_S on a real vibration.
            # dominant_score is the raw abs-max across all seven, kept because it
            # is a real diagnostic -- but it is NOT a claim of external character
            # (no real vibration is ever assigned a T/R slot).
            "highlighted_score": "+".join(
                r["ext_highlights"] if incorporate else r["highlights"]),
            "dominant_score": r["dominant"],
            "is_reference_mode": r["is_reference"],
            # Interpretability: how far s[V_S] sits from the cut, whether that is
            # inside the +/- delta band, and which degenerate set (if any) the
            # mode belongs to. The thresholds ride along on every row so a
            # saved file says what it was cut with.
            "sb_margin": r["sb_margin"],
            "sb_ambiguous": r["sb_ambiguous"],
            "degenerate_set": r["degenerate_set"],
            "tau_SB": payload["thresholds"]["tau_SB"],
            "ambiguity_lo": payload["thresholds"]["ambiguity_range"][0],
            "ambiguity_hi": payload["thresholds"]["ambiguity_range"][1],
            "frame_rule": payload["frame"]["rule"],
        }
        for b in r["bonds"]:
            row[f"s_AB[{b['pair']}]"] = b["s_AB"]
        rows.append(row)
    return rows
