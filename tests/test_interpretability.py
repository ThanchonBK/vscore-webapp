"""Interpretability aids added for the JCC revision (Reviewer 1, items 4/11).

  * degeneracy flagging from the eigenvalues (frequencies / EMIT eigenvalues),
    which therefore have to be present on every vibration for it to run;
  * user-set tau_SB and delta, with a borderline flag on any mode whose
    s[V_S] lies within delta of tau_SB;
  * rejecting a T/R winner so its slot falls to the next mode in line.

All driven from the shipped examples/ so the suite is self-contained.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.classifier import external_slots  # noqa: E402
from app.core.parsers import ParseError, parse_vsc  # noqa: E402
from app.core.pipeline import _assign, analyse, to_csv_rows  # noqa: E402
from app.core.scoring import ModeScorer  # noqa: E402
from app.main import _build, _options, _vsc_text  # noqa: E402

EX = ROOT / "examples"


def _vsc(stem):
    return (EX / f"{stem}.vsc").read_text()


def _run(stem, **kw):
    v = parse_vsc(_vsc(stem))
    return analyse(v["atoms"], v["coords"], v["bonds"], v["modes"], title=stem, **kw)


def _labels(p):
    return {r["name"]: r["label"] for r in p["vibrations"]}


# ---------------------------------------------------------------- degeneracy
def test_benzene_degenerate_pairs_are_found():
    p = _run("C6H6")
    assert p["degeneracy_checked"]
    assert len(p["degenerate_sets"]) == 10              # D6h: ten E pairs
    assert all(d["n"] == 2 for d in p["degenerate_sets"])
    tagged = [r["name"] for r in p["vibrations"] if r["degenerate_set"]]
    assert len(tagged) == 20


def test_accidental_near_degeneracy_is_not_flagged():
    """Naphthalene (D2h) has no degenerate modes; its closest pair is 0.25 cm-1
    apart, which the default tolerance must not call a degeneracy."""
    assert _run("C10H8")["degenerate_sets"] == []


def test_a_wider_tolerance_does_pick_it_up():
    assert len(_run("C10H8", deg_tol=0.3)["degenerate_sets"]) == 1


def test_missing_eigenvalues_disable_the_check_and_say_so():
    text = re.sub(r"\s+freq=\S+", "", _vsc("C6H6"))
    v = parse_vsc(text)
    p = analyse(v["atoms"], v["coords"], v["bonds"], v["modes"])
    assert not p["degeneracy_checked"]
    assert p["degenerate_sets"] == []
    assert any("Degeneracy was not checked" in w for w in p["warnings"])


def test_emit_degeneracy_uses_eigenvalues():
    p = _run("C6H6_EMIT")
    assert p["is_emit"] and p["degeneracy_checked"]
    assert max(d["n"] for d in p["degenerate_sets"]) >= 9


# ------------------------------------------------- tau / delta and borderline
def test_benzene_borderline_modes_match_the_reviewer():
    """Reviewer 1 item 11: nu10 (0.41) and nu14 (0.45/0.51) are ambiguous."""
    p = _run("C6H6")
    amb = {r["name"]: r["scores"]["V_S"] for r in p["vibrations"] if r["sb_ambiguous"]}
    assert set(amb) == {"Vib 13", "Vib 14", "Vib 16"}
    # no delta given -> the default delta 0.10 around tau_SB 0.43
    assert p["thresholds"]["delta"] == 0.10
    assert p["thresholds"]["ambiguity_range"] == [0.33, 0.53]
    assert any("δ = 0.10 of τ_SB = 0.43 (0.33–0.53)" in w for w in p["warnings"])
    # the pair is degenerate AND borderline -> its own warning
    deg = [w for w in p["warnings"] if "degenerate set" in w]
    assert len(deg) == 1, "one combined warning, not one box per set"
    assert "d5 (Vib 13, Vib 14" in deg[0]


def test_band_is_tau_plus_minus_delta():
    for kw, (lo, hi) in (({}, (0.33, 0.53)), ({"delta": 0.05}, (0.38, 0.48)),
                         ({"tau_sb": 0.5, "delta": 0.2}, (0.3, 0.7))):
        p = _run("C6H6", **kw)
        assert p["thresholds"]["ambiguity_range"] == [lo, hi], kw
        for r in p["vibrations"]:
            assert r["sb_ambiguous"] == (lo - 1e-9 <= r["scores"]["V_S"] <= hi + 1e-9) \
                or min(abs(r["scores"]["V_S"] - lo), abs(r["scores"]["V_S"] - hi)) < 1e-4


def test_default_band_differs_from_the_paper_range_on_purpose():
    """Default delta 0.10 gives 0.33-0.53; the paper's range is 0.32-0.53.
    Adamantane Vib 42 (0.3275) is the example case: borderline in the paper's
    range, not in the default band. Pinned so a change of mind is deliberate."""
    r = next(r for r in _run("C10H16")["vibrations"] if r["name"] == "Vib 42")
    assert 0.32 <= r["scores"]["V_S"] < 0.33 and not r["sb_ambiguous"]


def test_delta_zero_flags_nothing():
    p = _run("C6H6", delta=0)
    assert not any(r["sb_ambiguous"] for r in p["vibrations"])


def test_tau_override_moves_the_cut():
    base, moved = _run("C6H6"), _run("C6H6", tau_sb=0.5)
    assert moved["thresholds"]["tau_SB"] == 0.5
    assert moved["thresholds"]["tau_SB_overridden"]
    assert not base["thresholds"]["tau_SB_overridden"]
    flipped = [n for n in _labels(base) if _labels(base)[n] != _labels(moved)[n]]
    assert flipped == ["Vib 13"]                   # 0.451: S at 0.42, B at 0.50
    for r in moved["vibrations"]:
        assert r["label"] == ("S" if r["scores"]["V_S"] >= 0.5 else "B")


def test_constructed_references_are_never_borderline():
    p = _run("C6H6", delta=1.0)
    assert not any(r["sb_ambiguous"] for r in p["references"])
    assert all(r["sb_ambiguous"] for r in p["vibrations"])


@pytest.mark.parametrize("kw", [{"tau_sb": 1.5}, {"tau_sb": -0.1},
                                {"delta": -1}, {"deg_tol": -1}, {"delta": "x"}])
def test_bad_settings_are_refused(kw):
    with pytest.raises(ParseError):
        _run("C6H6", **kw)


def test_csv_carries_the_new_columns():
    p = _run("C6H6", tau_sb=0.45, delta=0.05)
    rows = {r["Mode"]: r for r in to_csv_rows(p)}
    assert rows["Vib 16"]["tau_SB"] == 0.45
    assert (rows["Vib 16"]["ambiguity_lo"], rows["Vib 16"]["ambiguity_hi"]) == (0.4, 0.5)
    assert rows["Vib 16"]["frame_rule"] == "symmetry"
    assert rows["Vib 13"]["sb_ambiguous"] is True        # 0.451 vs 0.45
    assert rows["Vib 13"]["degenerate_set"] == 5
    assert rows["Vib 5"]["degenerate_set"] is None


# ---------------------------------------------------------- T/R rejection
@pytest.mark.parametrize("stem", ["HCOOH_3N", "C6H6_EMIT", "C6H6", "CO2", "C10H8"])
def test_reassignment_reproduces_the_classifier(stem):
    """_assign() with nothing excluded must be the classifier's own verdict,
    or every 'next in line' it reports is about a different assignment."""
    v = parse_vsc(_vsc(stem))
    p = analyse(v["atoms"], v["coords"], v["bonds"], v["modes"])
    got = {r["index"]: r["ext_label"] for r in p["references"] + p["vibrations"]
           if r["ext_label"] in ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz")}
    # rebuild the classifier's input the same way analyse() does
    from app.core.classifier import classify_all_modes, Thresholds
    from app.core import frame as frame_mod
    import numpy as np
    scorer = ModeScorer(v["atoms"], v["coords"], v["bonds"])
    emit = any(m.get("is_emit") for m in v["modes"])
    vecs = [np.asarray(m["vector"], float) for m in v["modes"]]
    if emit:
        R = scorer.mit_rotation_matrix()
        vecs = [d @ R.T for d in vecs]
    F, _ = frame_mod.select_frame_for_scorer(scorer, vecs, keep_input_frame=False,
                                             tie_break="tmax", linear_rule="vector")
    scorer.apply_frame(F)
    rot = [dict(m, vector=d @ F.T) for m, d in zip(v["modes"], vecs)]
    if p["mode_set"] == "3n":
        final = rot
    else:
        R = scorer.construct_R()
        if p["linear"]:
            R = [m for m in R if m["label"] != "Rx"]
        final = scorer.construct_T() + R + rot
    scored = classify_all_modes(scorer, final, Thresholds.calibrated())
    _, _, slots = external_slots(scorer)
    assert _assign(scored, slots) == got


@pytest.mark.parametrize("stem", ["HCOOH_3N", "C6H6_EMIT"])
def test_rejecting_a_winner_hands_its_slot_to_the_runner_up(stem):
    base = _run(stem)
    for c in base["candidates"]:
        p = _run(stem, reject=[c["number"]])
        now = {x["slot"]: x["name"] for x in p["candidates"]}
        assert now[c["slot"]] == c["runner_up"]["name"]
        for mv in c["also_moves"]:
            assert now[mv["slot"]] == mv["to"]
        r = next(r for r in p["vibrations"] if r["name"] == c["name"])
        assert r["ext_label"] in ("S", "B")       # back to its own S/B label
        assert r["candidate"] is None
        assert p["rejected"] == [{"number": c["number"], "name": c["name"],
                                  "label": r["label"]}]


def test_rejections_stack():
    base = _run("HCOOH_3N")
    tx = next(c for c in base["candidates"] if c["slot"] == "Tx")
    once = _run("HCOOH_3N", reject=[tx["number"]])
    tx2 = next(c for c in once["candidates"] if c["slot"] == "Tx")
    twice = _run("HCOOH_3N", reject=[tx["number"], tx2["number"]])
    tx3 = next(c for c in twice["candidates"] if c["slot"] == "Tx")
    assert tx3["name"] == tx2["runner_up"]["name"]
    assert len(twice["rejected"]) == 2


@pytest.mark.parametrize("bad", [[0], [99], ["Vib 1"]])
def test_bad_rejections_are_refused(bad):
    with pytest.raises(ParseError):
        _run("HCOOH_3N", reject=bad)


def test_ranking_lists_the_slot_column_best_first():
    for c in _run("C6H6_EMIT")["candidates"]:
        scores = [k["related_score"] for k in c["ranking"]]
        assert scores == sorted(scores, reverse=True)
        assert c["ranking"][0]["name"] == c["name"]


# ------------------------------------------------- form plumbing / round trip
def test_options_parse_blank_as_default_and_split_rejects():
    assert _options("", "", "", []) == {}
    assert _options("0.5", " 0.05 ", "1", ["3", "4,5"]) == {
        "tau_sb": 0.5, "delta": 0.05, "deg_tol": 1.0, "reject": ["3", "4", "5"]}
    with pytest.raises(ParseError):
        _options("abc", "", "", [])


@pytest.mark.parametrize("stem", ["C6H6", "HCOOH_3N", "C6H6_EMIT"])
def test_rescore_from_the_pages_own_vsc_is_identical(stem):
    """The settings form posts the page's .vsc back. With unchanged settings
    that must reproduce the page exactly, or re-scoring would silently move
    numbers that the user did not touch."""
    first, raw = _build("vsc", None, None, _vsc(stem), None, None, f"{stem}.vsc")
    again, _ = _build("vsc", None, None, _vsc_text(first, raw), None, None, None,
                      opts=_options(str(first["thresholds"]["tau_SB"]),
                                    "" if first["thresholds"]["delta"] is None
                                    else str(first["thresholds"]["delta"]),
                                    str(first["thresholds"]["deg_tol"]), []),
                      source=first["source"])
    assert again == first


def test_rescore_of_a_selection_stays_a_selection():
    text = _vsc("HOCl")
    first, raw = _build("paste", None, None, None, None, None, None, pasted=text)
    again, _ = _build("paste", None, None, None, None, None, None,
                      pasted=_vsc_text(first, raw), opts={"delta": 0.2},
                      source=first["source"])
    assert again["mode_set"] == "arbitrary"
    assert again["thresholds"]["delta"] == 0.2
    assert again["source"] == first["source"]


# ---------------------------------------------------------- frame of reference
def test_frame_defaults_to_the_symmetry_rule():
    p = _run("C6H6")
    assert p["frame"]["rule"] == "symmetry"
    assert p["frame"]["kind"] == "symmetric"         # D6h: a symmetric top


@pytest.mark.parametrize("stem", ["C6H6", "C10H8", "CO2", "C6H6_EMIT", "HCOOH_3N"])
def test_frame_choice_never_moves_v_s_or_the_s_b_label(stem):
    """The frame is a T/R matter: s[V_S] is rotation-invariant, so no frame
    choice may change it or the stretch/bend label."""
    a, b = _run(stem), _run(stem, frame_rule="eigh")
    assert b["frame"]["rule"] == "eigh"
    for x, y in zip(a["vibrations"], b["vibrations"]):
        assert x["scores"]["V_S"] == pytest.approx(y["scores"]["V_S"], abs=1e-4)
        assert x["label"] == y["label"]


def _tr_change_on_rotation(stem, rule):
    import numpy as np
    from app.core.frame import rotation
    v = parse_vsc(_vsc(stem))
    R = rotation(np.array([0.3, -0.5, 0.8]), 0.7)
    turned = [dict(m, vector=np.asarray(m["vector"]) @ R.T) for m in v["modes"]]
    a = analyse(v["atoms"], v["coords"], v["bonds"], v["modes"], frame_rule=rule)
    b = analyse(v["atoms"], np.asarray(v["coords"]) @ R.T, v["bonds"], turned,
                frame_rule=rule)
    return max(abs(x["scores"][k] - y["scores"][k])
               for x, y in zip(a["vibrations"], b["vibrations"])
               for k in ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz"))


@pytest.mark.parametrize("stem", ["C10H16", "CO2", "C6H6"])
def test_symmetry_rule_does_not_depend_on_input_orientation(stem):
    """The point of the rule: rotate the input, get the same T/R scores."""
    assert _tr_change_on_rotation(stem, "symmetry") < 2e-3


@pytest.mark.parametrize("stem", ["C10H16", "CO2"])
def test_the_submitted_frame_does_depend_on_it(stem):
    """...which the submitted MIT frame does not guarantee: adamantane
    (spherical top) and CO2 (linear) move by ~0.5-0.65 under the same turn.
    Guards the test above against passing vacuously."""
    assert _tr_change_on_rotation(stem, "eigh") > 0.3


@pytest.mark.parametrize("kw", [{"frame_rule": "nope"}, {"tie_break": "x"},
                                {"linear_turn": "x"}, {"sym_tol": 0}])
def test_bad_frame_options_are_refused(kw):
    with pytest.raises(ParseError):
        _run("C6H6", **kw)


def test_frame_options_survive_the_form():
    o = _options("", "", "", [], "eigh", "1", "gap", "scan", "1e-4")
    assert o == {"frame_rule": "eigh", "keep_input_frame": True, "tie_break": "gap",
                 "linear_turn": "scan", "sym_tol": 1e-4}
    p = _run("C6H6", **o)
    assert p["frame"]["rule"] == "eigh" and p["frame"]["tie_break"] == "gap"


# ------------------------------------------------- bugs found in the 2026-11-16 sweep
def test_margin_sign_agrees_with_the_label_everywhere():
    """The label cuts on s[V_S] rounded half up to 0.01, so the margin must be
    taken on that too. Adamantane Vib 22 (0.4256 -> 0.43, S) read -0.0044."""
    for stem in ["C10H16", "C10H8", "C6H6", "C6H6_EMIT", "XeF2Cl2", "HOCl", "CO2"]:
        for r in _run(stem)["vibrations"]:
            assert (r["label"] == "S") == (r["sb_margin"] >= 0), (stem, r["name"])
    r = next(r for r in _run("C10H16")["vibrations"] if r["name"] == "Vib 22")
    assert r["label"] == "S" and r["sb_margin"] == 0.0


def test_band_follows_a_users_tau():
    p = _run("C6H6", tau_sb=0.8)
    assert p["thresholds"]["ambiguity_range"] == [0.7, 0.9]


def test_emit_degeneracy_note_says_eigenvalues():
    notes = " ".join(_run("C6H6_EMIT")["notes"])
    assert "eigenvalues within" in notes and "frequencies within" not in notes


def test_rejecting_on_a_vibrations_only_set_is_refused():
    with pytest.raises(ParseError, match="Nothing to reject"):
        _run("C6H6", reject=[1])


def test_text_fields_accept_as_much_as_an_upload():
    """The re-score form posts the page's .vsc as a TEXT field; Starlette's
    default 1 MB cap on those broke re-scoring anything between 1 and 4 MB."""
    from starlette.formparsers import MultiPartParser
    from app.config import MAX_UPLOAD_BYTES
    import app.main  # noqa: F401  (sets the cap on import)
    assert MultiPartParser.max_part_size >= MAX_UPLOAD_BYTES


# ------------------------------------------------- deployment limits (2026-11-17)
def _chain(n):
    import numpy as np
    X = [[0.0, 0.0, 1.3 * k] for k in range(n)]
    at = ["H"] + ["C"] * (n - 2) + ["H"]
    B = [(k, k + 1) for k in range(n - 1)]
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3 * n, 3 * n - 5)))
    return at, X, B, [{"frequency": 100.0 + i, "vector": Q[:, i].reshape(n, 3)}
                      for i in range(3 * n - 5)]


def test_scan_is_refused_past_the_cap_and_only_then():
    at, X, B, M = _chain(12)                        # 31 modes x 12 atoms = 372
    with pytest.raises(ParseError, match="too slow"):
        analyse(at, X, B, M, linear_turn="scan", max_scan_work=300)
    assert analyse(at, X, B, M, linear_turn="scan", max_scan_work=400)["frame"]["step"]
    assert analyse(at, X, B, M, linear_turn="scan")["frame"]["step"]   # no cap locally
    # non-linear molecules never scan, so the cap never applies to them
    assert _run("C10H16", linear_turn="scan", max_scan_work=1)["frame"]["kind"] == "spherical"


def test_the_cap_is_set_on_the_deployment_only(monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "IS_LOCAL", False)
    assert m._options("", "", "", [])["max_scan_work"] == m.SCAN_WORK_CAP
    monkeypatch.setattr(m, "IS_LOCAL", True)
    assert "max_scan_work" not in m._options("", "", "", [])


def test_pages_carry_the_request_cap_only_when_there_is_one():
    from app.main import templates
    env = templates.env
    old = env.globals["request_cap"]
    try:
        env.globals["request_cap"] = 4_400_000
        assert 'data-cap="4400000"' in env.get_template("index.html").render(request=None)
        env.globals["request_cap"] = None
        assert "data-cap" not in env.get_template("index.html").render(request=None)
    finally:
        env.globals["request_cap"] = old
