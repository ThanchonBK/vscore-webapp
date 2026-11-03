"""ORCA .hess and Molden input: the same molecule must score the same.

Two kinds of check. Real files (tests/data/cclib/, divinylbenzene at
B3LYP/STO-3G from Gaussian 16, ORCA 6 and a Molden writer) show the readers
agree with the Gaussian path across programs. Synthetic files built here from
the shipped .vsc examples pin the format details -- column blocks, zero modes,
bohr, comment lines, mass weighting -- to exact equality with the .vsc path.

Run:  python -m pytest tests/test_qc_parsers.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.parsers import (BOHR_TO_ANGSTROM, ParseError,  # noqa: E402
                              parse_bond_file, parse_frequency_file,
                              parse_gaussian_log, parse_molden, parse_orca_hess,
                              parse_vsc)
from app.core.pipeline import analyse  # noqa: E402
from app.core.utils import atomicMass  # noqa: E402
from app.main import _build  # noqa: E402

CCLIB = ROOT / "tests" / "data" / "cclib"
EXAMPLES = ROOT / "examples"


def _read(name):
    return (CCLIB / name).read_text()


def _score(g, bonds):
    p = analyse(g["atoms"], g["coords"], bonds, g["modes"], title="t", source="t")
    return sorted(p["vibrations"], key=lambda r: r["frequency"])


# ----------------------------------------------------------------------
# Real files: one molecule, three programs
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def gaussian_dvb():
    g = parse_gaussian_log(_read("dvb_ir_g16.out"))
    return _score(g, parse_bond_file(_read("dvb_g16.bonds"), len(g["atoms"])))


@pytest.mark.parametrize("data, bonds, max_dv", [
    # Same functional and basis as the Gaussian run: V_S agrees to <0.01.
    ("dvb_ir_orca6.hess", "dvb_orca6.bonds", 0.01),
    # A separate calculation (frequencies differ by up to 7 cm-1), so looser.
    ("dvb_ir.molden", "dvb_molden.bonds", 0.03),
])
def test_real_file_matches_gaussian(gaussian_dvb, data, bonds, max_dv):
    g = parse_frequency_file(_read(data))
    rows = _score(g, parse_bond_file(_read(bonds), len(g["atoms"])))
    assert len(rows) == len(gaussian_dvb) == 54
    assert [r["label"] for r in rows] == [r["label"] for r in gaussian_dvb]
    dv = max(abs(a["scores"]["V_S"] - b["scores"]["V_S"])
             for a, b in zip(rows, gaussian_dvb))
    assert dv < max_dv
    assert max(abs(a["frequency"] - b["frequency"])
               for a, b in zip(rows, gaussian_dvb)) < 10.0


def test_real_orca_geometry_is_converted_from_bohr():
    g = parse_orca_hess(_read("dvb_ir_orca6.hess"))
    d = [np.linalg.norm(g["coords"][i] - g["coords"][j])
         for i, j in parse_bond_file(_read("dvb_orca6.bonds"), 20)]
    assert 1.05 < min(d) and max(d) < 1.50          # C-H .. C-C, in Angstrom


def test_real_orca_symmetry_adapted_zero_modes_are_dropped():
    # This ORCA 6 run used 'usesym': its six 0 cm-1 modes carry real T/R
    # vectors rather than zeros, and must still be removed.
    g = parse_orca_hess(_read("dvb_ir_orca6.hess"))
    assert len(g["modes"]) == 54
    assert min(m["frequency"] for m in g["modes"]) > 40.0


# ----------------------------------------------------------------------
# Synthetic files from the shipped examples
# ----------------------------------------------------------------------
def _example(stem):
    return parse_vsc((EXAMPLES / f"{stem}.vsc").read_text())


def _masses(atoms):
    return np.array([atomicMass[a.lower()] for a in atoms])


def _vectors(v, mass_weighted):
    out = []
    for m in v["modes"]:
        x = np.asarray(m["vector"], float)
        if mass_weighted:
            x = x * np.sqrt(_masses(v["atoms"]))[:, None]
        out.append(x / np.linalg.norm(x))
    return out


def _hess_text(v, with_zero_modes=True, ncols=5, fmt="{:11.6f}", n_zero=None):
    """An ORCA-style .hess: comment lines, 3N columns in blocks, bohr."""
    n = len(v["atoms"])
    nc = 3 * n
    vecs = _vectors(v, mass_weighted=False)
    freqs = [m["frequency"] for m in v["modes"]]
    if with_zero_modes:
        nz = (nc - len(vecs)) if n_zero is None else n_zero
        vecs = [np.zeros((n, 3))] * nz + vecs
        freqs = [0.0] * nz + freqs
    mat = np.array([x.ravel() for x in vecs]).T            # one mode per column
    out = ["", "$orca_hessian_file", "", "$vibrational_frequencies", str(len(freqs))]
    out += [f"{k:5d}   {f:14.6f}" for k, f in enumerate(freqs)]
    out += ["", "$normal_modes", f"{nc} {mat.shape[1]}"]
    for c0 in range(0, mat.shape[1], ncols):
        cols = range(c0, min(c0 + ncols, mat.shape[1]))
        out.append("        " + "".join(f"{c:11d}" for c in cols))
        for r in range(nc):
            out.append(f"{r:6d}  " + "".join(fmt.format(mat[r, c]) for c in cols))
    out += ["", "#", "# The atoms: label  mass x y z (in bohrs)", "#", "$atoms", str(n)]
    for a, (x, y, z) in zip(v["atoms"], v["coords"]):
        b = [c / BOHR_TO_ANGSTROM for c in (x, y, z)]
        out.append(f" {a:2s} {atomicMass[a.lower()]:10.4f} {b[0]:13.6f} {b[1]:13.6f} {b[2]:13.6f}")
    out += ["", "$end", ""]
    return "\n".join(out)


def _molden_text(v, mass_weighted=False):
    out = ["[Molden Format]", "[Title]", "synthetic", "[FREQ]"]
    out += [f"{m['frequency']:12.4f}" for m in v["modes"]]
    out += ["[FR-COORD]"]
    for a, xyz in zip(v["atoms"], v["coords"]):
        out.append(f"{a} " + " ".join(f"{c / BOHR_TO_ANGSTROM:14.8f}" for c in xyz))
    out += ["[FR-NORM-COORD]"]
    for k, x in enumerate(_vectors(v, mass_weighted), 1):
        out.append(f"vibration {k}")
        out += [" ".join(f"{c:14.8f}" for c in row) for row in x]
    return "\n".join(out) + "\n"


def _assert_same_scores(v, g):
    a = _score(v, v["bonds"])
    b = _score(g, v["bonds"])
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra["label"] == rb["label"]
        for k, val in ra["scores"].items():
            assert rb["scores"][k] == pytest.approx(val, abs=2e-4), (k, ra["frequency"])


@pytest.mark.parametrize("stem", ["HOCl", "C10H8", "XeF2Cl2"])
def test_synthetic_hess_scores_like_vsc(stem):
    v = _example(stem)
    g = parse_orca_hess(_hess_text(v))
    assert g["atoms"] == v["atoms"]
    assert np.allclose(g["coords"], v["coords"], atol=1e-5)
    _assert_same_scores(v, g)


def test_synthetic_hess_linear_molecule_drops_five():
    v = _example("CO2")
    vib = [m for m in v["modes"] if m.get("frequency")]
    if len(vib) != 4:
        pytest.skip("CO2 example is not the 3N-5 vibration set")
    v = dict(v, modes=vib)
    assert len(parse_orca_hess(_hess_text(v))["modes"]) == 4


def test_hess_reads_scientific_notation_and_six_columns():
    # ORCA 3 printed 6 columns; ORCA 4+ prints 5, sometimes in E notation.
    v = _example("HOCl")
    g = parse_orca_hess(_hess_text(v, ncols=6, fmt="{:17.9E}"))
    _assert_same_scores(v, g)


@pytest.mark.parametrize("mass_weighted", [False, True])
def test_synthetic_molden_scores_like_vsc(mass_weighted):
    v = _example("C10H8")
    g = parse_molden(_molden_text(v, mass_weighted))
    _assert_same_scores(v, g)


def test_unconverted_mass_weighting_would_change_the_answer():
    # Why the convention check exists: feed mass-weighted vectors through
    # without conversion and V_S moves enough to matter.
    v = _example("HOCl")
    mw = dict(v, modes=[dict(m, vector=x) for m, x in
                        zip(v["modes"], _vectors(v, mass_weighted=True))])
    dv = max(abs(a["scores"]["V_S"] - b["scores"]["V_S"])
             for a, b in zip(_score(v, v["bonds"]), _score(mw, v["bonds"])))
    assert dv > 0.05


def test_non_orthogonal_vectors_are_refused():
    v = _example("HOCl")
    text = _molden_text(v).split("[FR-NORM-COORD]")[0] + "[FR-NORM-COORD]\n"
    for k in range(3):
        text += f"vibration {k + 1}\n" + "1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n"
    with pytest.raises(ParseError, match="not mutually orthogonal"):
        parse_molden(text)


def test_wrong_number_of_zero_modes_is_refused():
    v = _example("HOCl")
    with pytest.raises(ParseError):
        parse_orca_hess(_hess_text(v, n_zero=4))


def test_truncated_normal_modes_is_refused():
    text = _hess_text(_example("HOCl"))
    head, tail = text.split("$normal_modes")
    block, rest = tail.split("\n\n", 1)
    text = head + "$normal_modes" + "\n".join(block.splitlines()[:-3]) + "\n\n" + rest
    with pytest.raises(ParseError, match="does not match|incomplete"):
        parse_orca_hess(text)


@pytest.mark.parametrize("text, match", [
    ("$orca_hessian_file\n$atoms\n1\n H 1.0 0 0 0\n$end\n", r"\$vibrational_frequencies"),
    ("[Molden Format]\n[Atoms] Angs\nH 1 1 0 0 0\n", r"\[FREQ\]"),
    ("just some text\n", "Not recognised"),
])
def test_missing_sections_are_refused(text, match):
    with pytest.raises(ParseError, match=match):
        parse_frequency_file(text)


# ----------------------------------------------------------------------
# Bond files
# ----------------------------------------------------------------------
def test_bond_list_formats_agree():
    pairs = "# HOCl\n1 2\n1 3   # O-Cl\n"
    rows = "[CONNECTIVITY]\n1 2 3\n2\n3\n"
    assert parse_bond_file(pairs, 3) == parse_bond_file(rows, 3) == [(0, 1), (0, 2)]


def test_bond_file_out_of_range_is_refused():
    with pytest.raises(ParseError, match="references atom 4"):
        parse_bond_file("1 4\n", 3)


def test_bond_file_empty_is_refused():
    with pytest.raises(ParseError, match="empty"):
        parse_bond_file("# nothing\n", 3)


def test_com_deck_is_accepted_as_bond_file():
    deck = ("%chk=x\n# freq geom=connectivity\n\nt\n\n0 1\n"
            " O 0 0 0\n H 1 0 0\n Cl 0 1 0\n\n 1 2 1.0 3 1.0\n 2\n 3\n\n")
    assert parse_bond_file(deck, 3, "HOCl.com") == [(0, 1), (0, 2)]


# ----------------------------------------------------------------------
# The upload route
# ----------------------------------------------------------------------
def test_upload_route_scores_orca():
    payload, raw = _build("qc", None, None, None, None, None, None,
                          qc_text=_read("dvb_ir_orca6.hess"), qc_name="dvb.hess",
                          bond_text=_read("dvb_orca6.bonds"), bond_name="dvb.bonds")
    assert payload["n_modes"] == 54 and payload["title"] == "dvb"
    assert len(raw["bonds"]) == 20


def test_upload_route_requires_bond_file():
    with pytest.raises(ParseError, match="No bond file"):
        _build("qc", None, None, None, None, None, None,
               qc_text=_read("dvb_ir_orca6.hess"), qc_name="dvb.hess")


# ----------------------------------------------------------------------
# Regressions from the stress run (one per bug found)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["nan", "inf", "abc", "1,5"])
@pytest.mark.parametrize("where", ["freq", "coord", "vector"])
def test_non_finite_or_junk_numbers_are_parse_errors(bad, where):
    # Each of these crashed with a raw ValueError / LinAlgError (a 500) before.
    v = _example("HOCl")
    hess = _hess_text(v)
    if where == "freq":
        hess = hess.replace("667.640600", bad)
    elif where == "coord":
        hess = hess.replace("2.262989", bad)
    else:
        hess = hess.replace("0.850149", bad)
    with pytest.raises(ParseError):
        parse_orca_hess(hess)
    mol = _molden_text(v).replace("667.6406", bad) if where == "freq" else \
        _molden_text(v).replace("2.26298861", bad) if where == "coord" else \
        _molden_text(v).replace("0.85014918", bad)
    with pytest.raises(ParseError):
        parse_molden(mol)


def test_short_molden_row_is_a_parse_error():
    text = _molden_text(_example("HOCl")).replace("    0.00088000     0.85014918    -0.00000000",
                                                  "    0.00088000     0.85014918")
    with pytest.raises(ParseError, match="three numbers"):
        parse_molden(text)


def test_out_of_range_column_header_is_a_parse_error():
    text = _hess_text(_example("HOCl")).replace("          5          6          7          8",
                                                "          5          6          7     999999", 1)
    with pytest.raises(ParseError, match="column header"):
        parse_orca_hess(text)


def test_genuine_zero_frequency_vibration_after_externals_is_kept():
    # ORCA lists the externals first; a real mode printing 0.00 may follow.
    text = _hess_text(_example("HOCl")).replace("667.640600", "0.000000")
    assert len(parse_orca_hess(text)["modes"]) == 3


def test_cartesian_input_is_never_converted_when_ambiguous():
    # Linear: symmetry makes the modes orthogonal under BOTH metrics, so
    # orthogonality alone cannot tell; Cartesian must be left alone.
    v = _example("CO2")
    vib = [m for m in v["modes"] if m.get("frequency")]
    if len(vib) != 4:
        pytest.skip("CO2 example is not the 3N-5 vibration set")
    g = parse_molden(_molden_text(dict(v, modes=vib), mass_weighted=False))
    for got, m in zip(g["modes"], vib):
        x = np.asarray(m["vector"], float)
        assert np.allclose(got["vector"], x / np.linalg.norm(x), atol=1e-7)


def test_precision_warning_survives_conversion():
    v = _example("HOCl")
    text = "\n".join(" ".join(f"{float(x):.2f}" for x in line.split())
                     if line and line[0] in " -0123456789" else line
                     for line in _molden_text(v, mass_weighted=True).splitlines()) + "\n"
    payload, _ = _build("qc", None, None, None, None, None, None, qc_text=text,
                        qc_name="x.molden", bond_text="1 2\n1 3\n", bond_name="b.txt")
    assert payload["precision_dp"] == 2
    assert any("decimal places" in w for w in payload["warnings"])


@pytest.mark.parametrize("data, bonds", [("dvb_ir_orca6.hess", "dvb_orca6.bonds"),
                                         ("dvb_ir.molden", "dvb_molden.bonds")])
def test_vsc_download_roundtrips_exactly(data, bonds):
    from app.main import _vsc_text
    p1, raw = _build("qc", None, None, None, None, None, None, qc_text=_read(data),
                     qc_name=data, bond_text=_read(bonds), bond_name=bonds)
    p2, _ = _build("vsc", None, None, _vsc_text(p1, raw), None, None, "x.vsc")
    for a, b in zip(p1["vibrations"], p2["vibrations"]):
        assert a["scores"] == b["scores"] and a["label"] == b["label"]


def test_gaussian_vsc_output_is_unchanged():
    from app.core.parsers import write_vsc
    for f in sorted(EXAMPLES.glob("*.vsc")):
        v = parse_vsc(f.read_text())
        body = [ln for ln in write_vsc(v["atoms"], v["coords"], v["bonds"],
                                       v["modes"]).splitlines() if not ln.startswith("#")]
        assert body == [ln for ln in f.read_text().splitlines() if not ln.startswith("#")], f.name


def test_byte_order_mark_is_stripped_on_upload():
    import asyncio
    from app.main import _read as read_upload

    class Upload:
        filename = "x.molden"

        def __init__(self, data):
            self.data = data

        async def read(self):
            return self.data

    mol = _molden_text(_example("HOCl"))
    text = asyncio.run(read_upload(Upload(b"\xef\xbb\xbf" + mol.encode()), "f"))
    assert parse_frequency_file(text)["atoms"] == ["O", "H", "Cl"]
    bonds = asyncio.run(read_upload(Upload(b"\xef\xbb\xbf1 2\n1 3\n"), "f"))
    assert parse_bond_file(bonds, 3) == [(0, 1), (0, 2)]
