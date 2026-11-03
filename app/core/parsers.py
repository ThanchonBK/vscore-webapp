"""parsers.py -- readers for the three input paths, all producing one shape.

Every path converges on the same four values, which are exactly what
``ModeScorer`` needs and nothing more (verified against the reference
implementation in scoring-functions/):

    atoms   list[str]        element symbol per atom      -> atomic mass
    coords  (N,3) float      equilibrium geometry         -> COM, principal axes, bonds
    bonds   list[(i,j)]      0-based index pairs          -> s[V_S] ONLY
    modes   list[dict]       {'frequency','vector',...}   -> the mode itself

Frequency, reduced mass, force constant and irrep ride along as row metadata;
the scorer never reads them. The Cartesian Hessian -- the largest block in a
VEDA .fmt -- is not needed at all and is not parsed.

Readers:
  * ``parse_gaussian_log``  geometry + modes from a .log (NO connectivity --
    Gaussian does not print it in the output; it lives in the input deck)
  * ``parse_connectivity``  bonds from a .com/.gjf deck
  * ``parse_orca_hess``     geometry + modes from an ORCA .hess
  * ``parse_molden``        geometry + modes from a Molden file (Molpro and others)
  * ``parse_bond_file``     bonds for those two, from a .com/.gjf or a bond list
  * ``parse_vsc``           all four from one .vsc file

The ORCA and Molden readers also bring their vectors to the Gaussian
convention (unit-norm Cartesian displacements, 3N-6 or 3N-5 of them), so
everything downstream sees one shape whichever program wrote the file.

and one writer, ``write_vsc``, so a run can be replayed from a single small
file. The writer stores the ORIGINAL geometry, not the principal-axis-aligned
one, so a .vsc is a faithful record of what was uploaded.
"""

from __future__ import annotations

import re

import numpy as np

from .utils import get_symbol, get_atomic_number, atomicMass

VSC_VERSION = "1.0"

_DASH_RE = re.compile(r"^-+$")


class ParseError(ValueError):
    """Raised with a message intended to be shown directly to the user."""


# ======================================================================
# Shared Gaussian frequency-block helpers (ported from scoring-functions
# src/parser.py so the two agree line-for-line on block layout).
# ======================================================================
def _values_after_dash_marker(line):
    parts = line.split()
    for i, tok in enumerate(parts):
        if _DASH_RE.match(tok):
            return [float(x) for x in parts[i + 1:]]
    raise ValueError(f"no dash-marker token found in line: {line!r}")


def _parse_freq_block_header(lines, line_idx):
    freqs = _values_after_dash_marker(lines[line_idx])
    n = len(freqs)

    irreps = lines[line_idx - 1].split() if line_idx - 1 >= 0 else []
    if len(irreps) != n:
        irreps = [None] * n

    def _optional(offset):
        try:
            vals = _values_after_dash_marker(lines[line_idx + offset])
        except (ValueError, IndexError):
            return [None] * n
        return vals if len(vals) == n else [None] * n

    return freqs, irreps, _optional(1), _optional(2)


def _parse_hp_mode_vectors(lines, data_start, natoms, num_modes):
    temp = np.zeros((num_modes, natoms, 3))
    cur, count = data_start, 0
    while count < 3 * natoms and cur < len(lines):
        parts = lines[cur].split()
        if len(parts) < 3:
            break
        try:
            c_idx = int(parts[0]) - 1
            a_idx = int(parts[1]) - 1
            if a_idx >= natoms:
                cur += 1
                continue
            vals = [float(v) for v in parts[3:]]
            for m in range(min(len(vals), num_modes)):
                temp[m, a_idx, c_idx] = vals[m]
            count += 1
            cur += 1
        except ValueError:
            break
    return temp, cur


# ======================================================================
# Path 1a: Gaussian .log  ->  atoms, coords, modes
# ======================================================================
def parse_gaussian_log(text):
    """Geometry and normal modes from Gaussian output text.

    Returns ``{'atoms', 'coords', 'modes'}``. Deliberately no 'bonds' key --
    a Gaussian .log carries no connectivity, and inventing one here is exactly
    the silent-corruption failure the reference implementation refuses to make.
    """
    lines = text.splitlines()

    starts = [i for i, l in enumerate(lines) if "Standard orientation" in l]
    if not starts:
        starts = [i for i, l in enumerate(lines) if "Input orientation" in l]
    if not starts:
        raise ParseError(
            "No 'Standard orientation' or 'Input orientation' block found. "
            "This does not look like a Gaussian output file.")

    geom_line = starts[-1] + 5
    coords, symbols = [], []
    i = geom_line
    while i < len(lines) and "-----" not in lines[i]:
        parts = lines[i].split()
        if len(parts) < 6:
            break
        try:
            symbols.append(get_symbol(int(parts[1])))
            coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
        except ValueError:
            pass
        i += 1

    natoms = len(coords)
    if natoms == 0:
        raise ParseError("Found an orientation block but could not read any atoms from it.")

    modes = _parse_log_modes(lines, geom_line, natoms)
    if not modes:
        raise ParseError(
            "No vibrational modes found. This looks like a Gaussian job without a "
            "frequency calculation -- the scorer needs displacement vectors, so run "
            "the job with 'freq' (ideally 'freq=hpmodes') and upload that output.")

    expected = {3 * natoms - 6, 3 * natoms - 5}
    if len(modes) not in expected:
        raise ParseError(
            f"Parsed {len(modes)} vibrational modes for {natoms} atoms, but expected "
            f"3N-6={3*natoms-6} (nonlinear) or 3N-5={3*natoms-5} (linear). "
            "The frequency block was mis-parsed or the file is truncated.")

    return {"atoms": symbols, "coords": np.array(coords), "modes": modes}


def _parse_log_modes(lines, geom_line, natoms):
    has_hp = any("Coord Atom Element:" in l for l in lines)
    freq_lines = [i for i, l in enumerate(lines)
                  if "Frequencies --" in l and i > geom_line]
    if not freq_lines:
        freq_lines = [i for i, l in enumerate(lines) if "Frequencies --" in l]

    modes = []
    for start in freq_lines:
        _parse_log_block(lines, start, natoms, modes, force_hp_only=has_hp)
    return modes


def _parse_log_block(lines, line_idx, natoms, modes, force_hp_only):
    try:
        freqs, irreps, mus, ks = _parse_freq_block_header(lines, line_idx)
    except (ValueError, IndexError):
        return

    n_block = len(freqs)
    data_start, is_hp = -1, False
    for i in range(line_idx + 1, min(len(lines), line_idx + 30)):
        line = lines[i].strip()
        if "Coord Atom Element:" in line:
            data_start, is_hp = i + 1, True
            break
        if "Atom" in line and "AN" in line:
            data_start, is_hp = i + 1, False
            break
        parts = line.split()
        if len(parts) > 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            try:
                float(parts[3])
                data_start, is_hp = i, True
                break
            except ValueError:
                pass

    if data_start == -1:
        return
    # When HP blocks exist, the Standard block repeats the same frequencies at
    # lower precision -- taking both would double-count every mode.
    if force_hp_only and not is_hp:
        return

    if is_hp:
        vecs, _ = _parse_hp_mode_vectors(lines, data_start, natoms, n_block)
        for m in range(n_block):
            modes.append({"frequency": freqs[m], "vector": vecs[m], "is_emit": False,
                          "reduced_mass": mus[m], "force_constant": ks[m],
                          "irrep": irreps[m]})
        return

    block = [[] for _ in range(n_block)]
    cur, count = data_start, 0
    while count < natoms and cur < len(lines):
        parts = lines[cur].split()
        try:
            raw = [float(v) for v in parts[2:]]
            for m in range(n_block):
                c = m * 3
                if c + 3 <= len(raw):
                    block[m].append(raw[c:c + 3])
            count += 1
            cur += 1
        except (ValueError, IndexError):
            break

    for m in range(n_block):
        if len(block[m]) == natoms:
            modes.append({"frequency": freqs[m], "vector": np.array(block[m]),
                          "is_emit": False, "reduced_mass": mus[m],
                          "force_constant": ks[m], "irrep": irreps[m]})


# ======================================================================
# Path 1b: Gaussian .com / .gjf  ->  bonds
# ======================================================================
def parse_connectivity(text, natoms=None):
    """Bond list from a Gaussian input deck's geom=connectivity section.

    Format is ``center  partner [order]  partner [order] ...``; bond orders are
    tolerated and ignored (they play no part in the V-score). Returns 0-based,
    de-duplicated, sorted index pairs.
    """
    lines = [l.rstrip() for l in text.splitlines()]

    # The charge/multiplicity line ("0 1") marks the start of the molecule
    # specification; connectivity follows the blank line after the geometry.
    idx, found = 0, False
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2 and _is_int(parts[0]) and _is_int(parts[1]):
            idx, found = i + 1, True
            break
    if not found:
        raise ParseError(
            "No charge/multiplicity line (e.g. '0 1') found in the input deck, so the "
            "connectivity section could not be located. Is this really a .com/.gjf file?")

    while idx < len(lines) and lines[idx].strip():
        idx += 1
    idx += 1

    bonds = []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            break
        parts = line.split()
        if parts and parts[0].isdigit():
            center = int(parts[0]) - 1
            for k in range(1, len(parts), 2):
                try:
                    neighbour = int(parts[k]) - 1
                except ValueError:
                    continue
                if neighbour == center:
                    continue
                pair = tuple(sorted((center, neighbour)))
                if pair not in bonds:
                    bonds.append(pair)
        idx += 1

    if not bonds:
        raise ParseError(
            "The input deck has no connectivity section. Re-run Gaussian with "
            "'geom=connectivity' in the route line, or upload a .vsc file with an "
            "explicit [CONNECTIVITY] block. Without bonds the V-score is undefined -- "
            "it would silently evaluate to 0.000 and label every mode as bending.")

    if natoms is not None:
        bad = [p for p in bonds if p[0] >= natoms or p[1] >= natoms or p[0] < 0]
        if bad:
            raise ParseError(
                f"Connectivity references atom index {max(max(p) for p in bad) + 1}, but "
                f"the geometry has only {natoms} atoms. The .com and .log do not describe "
                "the same molecule.")

    return sorted(bonds)


def _is_int(s):
    return s.lstrip("+-").isdigit()


# ======================================================================
# Path 3: ORCA .hess / Molden  ->  atoms, coords, modes
# ======================================================================
# Neither format carries connectivity, so -- exactly as for a Gaussian .log --
# the bonds come from a separate file (``parse_bond_file``) and are never
# inferred from distances.
BOHR_TO_ANGSTROM = 0.529177210903

# A non-orthogonality above this, under the best-fitting metric, means the
# vectors are not normal modes of one Hessian (or were mis-parsed). Gaussian's
# 5-dp hpmodes already reach 7.5e-3 on heavy-atom molecules (PCl3, SeBr2), and
# 4-dp writers more; unrelated vectors overlap at O(0.1-1).
_ORTHO_TOL = 5e-2
# How much better the mass-weighted reading must fit before vectors are
# converted; anything less is ambiguous and stays Cartesian.
_DECISIVE = 3.0


def parse_orca_hess(text):
    """Geometry and normal modes from an ORCA ``.hess`` file.

    Reads ``$atoms`` (symbol, mass, x y z in bohr), ``$vibrational_frequencies``
    and ``$normal_modes`` (a 3N x 3N matrix, one mode per column, printed in
    column blocks). ORCA documents these columns as Cartesian displacements,
    normalised -- the same convention as Gaussian -- which
    ``_cartesian_modes`` confirms from the vectors themselves rather than
    trusting. The Hessian block is not needed and is not parsed.

    Returns ``{'atoms', 'coords', 'modes'}``; no 'bonds', see above.
    """
    sections = _dollar_sections(text)
    for req in ("atoms", "vibrational_frequencies", "normal_modes"):
        if req not in sections:
            raise ParseError(
                f"No ${req} block found. This does not look like an ORCA .hess file "
                "from a frequency calculation.")

    rows = sections["atoms"]
    try:
        natoms = int(rows[0].split()[0])
    except (IndexError, ValueError):
        raise ParseError("The $atoms block does not start with an atom count.")
    if natoms < 1:
        raise ParseError("The $atoms block declares no atoms.")
    atoms, coords, masses = [], [], []
    for n, line in enumerate(rows[1:natoms + 1], 1):
        parts = line.split()
        if len(parts) < 5:
            raise ParseError(f"$atoms line {n} should read 'symbol mass x y z': {line!r}")
        atoms.append(_element(parts[0], f"$atoms line {n}"))
        mass, *xyz = _numbers(parts[1:5], f"$atoms line {n}")
        if mass <= 0:
            raise ParseError(f"$atoms line {n}: mass {mass} is not positive.")
        masses.append(mass)
        coords.append([v * BOHR_TO_ANGSTROM for v in xyz])
    if len(atoms) != natoms:
        raise ParseError(f"$atoms declares {natoms} atoms but lists {len(atoms)}.")

    freqs = []
    for line in sections["vibrational_frequencies"][1:]:
        parts = line.split()
        if len(parts) >= 2:
            freqs.append(_numbers(parts[1:2], "$vibrational_frequencies")[0])
    ncoord = 3 * natoms
    if len(freqs) != ncoord:
        raise ParseError(
            f"$vibrational_frequencies lists {len(freqs)} values, expected 3N = {ncoord}.")

    matrix = _orca_column_blocks(sections["normal_modes"], ncoord)
    modes = [{"frequency": freqs[k], "vector": matrix[:, k].reshape(natoms, 3),
              "is_emit": False, "reduced_mass": None, "force_constant": None,
              "irrep": None} for k in range(ncoord)]
    modes = _drop_projected_modes(modes, natoms, np.array(coords))
    _cartesian_modes(modes, np.array(masses), "ORCA .hess")
    return {"atoms": atoms, "coords": np.array(coords), "modes": modes}


def _dollar_sections(text):
    """'$name' blocks of an ORCA .hess -> {name: [non-blank lines]}.

    ORCA puts '#' comment lines between and inside blocks (e.g. '# The atoms:
    label  mass x y z'); they are dropped.
    """
    sections, current = {}, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        if s.startswith("$"):
            current = s[1:].split()[0].lower() if len(s) > 1 else None
            if current == "end":
                break
            sections[current] = []
        elif current and s:
            sections[current].append(s)
    return sections


def _orca_column_blocks(lines, ncoord):
    """The '$normal_modes' matrix: a 'rows cols' line, then blocks of columns,
    each a header of column indices followed by 'row v v v ...' lines."""
    try:
        nrow, ncol = (int(v) for v in lines[0].split()[:2])
    except (IndexError, ValueError):
        raise ParseError("$normal_modes does not start with its 'rows cols' dimensions.")
    if nrow != ncoord or ncol != ncoord:
        raise ParseError(
            f"$normal_modes is {nrow} x {ncol}, but the geometry needs {ncoord} x {ncoord}.")

    matrix = np.full((ncoord, ncoord), np.nan)
    cols = []
    for line in lines[1:]:
        parts = line.split()
        if all(_is_int(p) for p in parts):          # a column-index header
            cols = [int(p) for p in parts]
            if len(set(cols)) != len(cols) or not all(0 <= c < ncoord for c in cols):
                raise ParseError(f"$normal_modes has an invalid column header: {line!r}")
            continue
        if not _is_int(parts[0]):
            raise ParseError(f"$normal_modes has an unreadable line: {line!r}")
        row = int(parts[0])
        vals = _numbers(parts[1:], "$normal_modes")
        if len(vals) != len(cols) or not 0 <= row < ncoord:
            raise ParseError(f"$normal_modes line does not match its block header: {line!r}")
        matrix[row, cols] = vals
    if np.isnan(matrix).any():
        raise ParseError("$normal_modes is incomplete -- the file looks truncated.")
    return matrix


_MOLDEN_SECTION_RE = re.compile(r"^\s*\[([A-Za-z0-9_\- ]+)\]\s*(.*)$")


def parse_molden(text):
    """Geometry and normal modes from the vibrational part of a Molden file.

    Reads ``[FREQ]``, ``[FR-COORD]`` (symbol x y z, bohr) and
    ``[FR-NORM-COORD]`` (one 'vibration n' header per mode, then N rows of
    dx dy dz). This covers Molpro (``put,molden``) and any other program that
    writes the format. Writers disagree on whether the vectors are mass-weighted,
    and the format does not say, so the convention is detected from the vectors
    (``_cartesian_modes``) and converted to Cartesian if needed. The [Atoms],
    [GTO] and [MO] sections are not read: [FR-COORD] is the geometry the modes
    belong to.

    Returns ``{'atoms', 'coords', 'modes'}``; no 'bonds'.
    """
    sections, current = {}, None
    for line in text.splitlines():
        m = _MOLDEN_SECTION_RE.match(line)
        if m:
            current = m.group(1).strip().upper()
            sections[current] = []
            continue
        if current and line.strip():
            sections[current].append(line.strip())

    if not text.lstrip().upper().startswith("[MOLDEN FORMAT]"):
        raise ParseError("This does not look like a Molden file (no '[Molden Format]' header).")
    for req in ("FREQ", "FR-COORD", "FR-NORM-COORD"):
        if req not in sections:
            raise ParseError(
                f"No [{req}] section. The Molden file has no vibrational data -- write "
                "it after a frequency calculation (Molpro: 'put,molden' after 'frequencies').")

    atoms, coords = [], []
    for n, line in enumerate(sections["FR-COORD"], 1):
        parts = line.split()
        if len(parts) < 4:
            raise ParseError(f"[FR-COORD] line {n} should read 'symbol x y z': {line!r}")
        atoms.append(_element(parts[0], f"[FR-COORD] line {n}"))
        coords.append([v * BOHR_TO_ANGSTROM
                       for v in _numbers(parts[1:4], f"[FR-COORD] line {n}")])
    natoms = len(atoms)
    if natoms == 0:
        raise ParseError("[FR-COORD] is empty.")

    freqs = [_numbers(line.split()[:1], "[FREQ]")[0] for line in sections["FREQ"]]

    vectors, rows = [], None
    for line in sections["FR-NORM-COORD"]:
        if line.lower().startswith("vibration"):
            if rows is not None:
                vectors.append(rows)
            rows = []
            continue
        if rows is None:
            raise ParseError(f"[FR-NORM-COORD] row before any 'vibration' header: {line!r}")
        parts = line.split()
        if len(parts) < 3:
            raise ParseError(f"[FR-NORM-COORD] row is not three numbers: {line!r}")
        rows.append(_numbers(parts[:3], "[FR-NORM-COORD]"))
    if rows is not None:
        vectors.append(rows)

    if len(vectors) != len(freqs):
        raise ParseError(
            f"[FREQ] lists {len(freqs)} frequencies but [FR-NORM-COORD] holds "
            f"{len(vectors)} vibrations.")
    bad = [k + 1 for k, v in enumerate(vectors) if len(v) != natoms]
    if bad:
        raise ParseError(
            f"Vibration {bad[0]} has the wrong number of rows for {natoms} atoms in "
            "[FR-COORD].")

    modes = [{"frequency": f, "vector": np.array(v), "is_emit": False,
              "reduced_mass": None, "force_constant": None, "irrep": None}
             for f, v in zip(freqs, vectors)]
    modes = _drop_projected_modes(modes, natoms, np.array(coords))
    masses = np.array([atomicMass[a.lower()] for a in atoms])
    _cartesian_modes(modes, masses, "Molden file")
    return {"atoms": atoms, "coords": np.array(coords), "modes": modes}


def _numbers(tokens, where):
    """Finite floats, or a ParseError naming the offending token.

    NaN and inf are refused, not just non-numbers: a NaN coordinate makes the
    linearity SVD fail to converge, and a NaN frequency cannot be serialised to
    the JSON the result page is built from -- both would surface as a 500.
    """
    out = []
    for tok in tokens:
        try:
            v = float(tok)
        except ValueError:
            raise ParseError(f"{where}: {tok!r} is not a number.")
        if not np.isfinite(v):
            raise ParseError(f"{where}: {tok!r} is not a finite number.")
        out.append(v)
    return out


def _element(tok, where):
    """Element symbol from a token like 'C', 'cl', 'C1' or '6'."""
    if tok.isdigit():
        sym = get_symbol(int(tok))
        if sym != "X":
            return sym
    else:
        sym = re.sub(r"[^A-Za-z]", "", tok).capitalize()
        if sym.lower() in atomicMass:
            return sym
    raise ParseError(
        f"{where}: unrecognised element {tok!r}; no atomic mass on file. Element "
        "identity affects all seven scores, so this cannot be defaulted.")


def _is_linear(coords, tol=1e-3):
    centred = coords - coords.mean(axis=0)
    if len(coords) < 3:
        return True
    return np.linalg.svd(centred, compute_uv=False)[1] < tol


def _drop_projected_modes(modes, natoms, coords):
    """Remove the projected-out translations/rotations a 3N listing carries.

    ORCA (and some Molden writers) list all 3N modes, with the 5-6 external
    ones set to exactly 0 cm-1 -- as all-zero vectors in a .hess, as real T/R
    vectors in symmetry-adapted runs. Those are dropped so the result has the
    same 3N-6 / 3N-5 vibrations a Gaussian log does. A set with no zeros is
    passed through untouched (a 3N set with unprojected T/R, or vibrations
    only), and the pipeline's own mode-set detection decides what it is.
    """
    zero = [k for k, m in enumerate(modes)
            if abs(m["frequency"]) < 1e-3 or not np.any(m["vector"])]
    if not zero:
        return modes
    n_ext = 5 if _is_linear(coords) else 6
    if len(zero) > n_ext and zero[:n_ext] == list(range(n_ext)):
        # ORCA always lists the external modes first, then any imaginary ones,
        # then the rest; a genuine vibration that prints as 0.00 can follow.
        # Drop by position and keep it -- unless its vector is empty too.
        zero = zero[:n_ext]
        if any(not np.any(m["vector"]) for k, m in enumerate(modes) if k not in zero):
            raise ParseError(
                "A mode after the translations and rotations has an all-zero "
                "displacement vector, so it cannot be scored.")
    if len(zero) != n_ext:
        raise ParseError(
            f"Found {len(zero)} zero-frequency modes; a {'linear' if n_ext == 5 else 'nonlinear'} "
            f"molecule has {n_ext} translations and rotations. Either the Hessian was not "
            "projected cleanly or the file is mis-parsed, so these are not dropped by guesswork.")
    keep = [m for k, m in enumerate(modes) if k not in zero]
    if len(keep) != 3 * natoms - n_ext:
        raise ParseError(
            f"After removing {n_ext} zero modes, {len(keep)} remain; expected "
            f"{3 * natoms - n_ext}.")
    return keep


def _cartesian_modes(modes, masses, what):
    """Make every vector a unit-norm Cartesian displacement, in place.

    s[V_S] is invariant to a mode's overall scale but not to the RELATIVE
    displacement of its atoms, so a mass-weighted vector (q = M^1/2 x) scores
    differently from the Cartesian one Gaussian prints. Which one a file holds
    is read off the vectors with two independent tests:

      * orthogonality -- normal modes are orthogonal as mass-weighted vectors
        (Q^T Q diagonal), and as Cartesian displacements only under the mass
        metric (X^T M X diagonal);
      * zero net momentum (Eckart) -- a vibration has sum_i m_i x_i = 0 as a
        Cartesian displacement, sum_i sqrt(m_i) q_i = 0 when mass-weighted.

    Either test alone can be blind: in a linear molecule or a set with one mode
    per irrep, symmetry makes the vectors orthogonal under both metrics, and
    with near-equal masses both tests barely differ. So vectors are converted
    only when the mass-weighted reading fits DECISIVELY better on the two tests
    combined; otherwise they are taken as Cartesian, the convention both ORCA
    and the Molden format document. Vectors orthogonal under neither metric are
    not normal modes of one Hessian and are refused.

    The precision the file was written at is measured before anything is
    rescaled and kept on each mode as ``source_dp``, so the low-precision
    warning still reflects the file rather than the normalised floats.
    """
    X = np.array([np.asarray(m["vector"], float).ravel() for m in modes]).T
    if not np.isfinite(X).all():
        raise ParseError(f"The {what} holds non-numeric (NaN/inf) displacements.")
    dp = displacement_precision(modes)
    convert = False
    if len(modes) >= 2 and np.ptp(masses) > 0:
        m3 = np.repeat(masses, 3)
        # numpy 2.x on macOS Accelerate raises spurious FP warnings from matmul
        # on finite input; the inputs were just checked finite, so silence them.
        with np.errstate(all="ignore"):
            off_cart = _offdiag(X.T @ (m3[:, None] * X))
            off_mw = _offdiag(X.T @ X)
        if min(off_cart, off_mw) > _ORTHO_TOL:
            raise ParseError(
                f"The {what}'s mode vectors are not mutually orthogonal under either the "
                f"Cartesian or the mass-weighted convention (worst overlaps {off_cart:.3f} / "
                f"{off_mw:.3f}). They are not normal modes of one Hessian, or the file was "
                "mis-parsed.")
        mw_fit = off_mw + _net_momentum(X, np.sqrt(masses))
        cart_fit = off_cart + _net_momentum(X, masses)
        convert = mw_fit * _DECISIVE < cart_fit
    if convert:
        for m in modes:
            m["vector"] = np.asarray(m["vector"], float) / np.sqrt(masses)[:, None]
    _normalise(modes)
    for m in modes:
        m["source_dp"] = dp


def _net_momentum(X, weights):
    """Median over modes of |sum_i w_i x_i| / sum_i w_i |x_i| -- ~0 when the
    weighting matches the vectors' convention. The median, not the max, so the
    rigid translations a 3N set may carry do not swamp the vibrations."""
    n = len(weights)
    per_mode = []
    for k in range(X.shape[1]):
        x = X[:, k].reshape(n, 3) * weights[:, None]
        scale = np.abs(x).sum()
        per_mode.append(np.linalg.norm(x.sum(axis=0)) / scale if scale > 0 else 0.0)
    return float(np.median(per_mode))


def _offdiag(G):
    d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    C = G / np.outer(d, d)
    np.fill_diagonal(C, 0.0)
    return float(np.abs(C).max())


def _normalise(modes):
    for m in modes:
        v = np.asarray(m["vector"], float)
        n = np.linalg.norm(v)
        m["vector"] = v / n if n > 0 else v


def parse_frequency_file(text):
    """ORCA .hess or Molden, told apart by content rather than file extension."""
    if text.lstrip().upper().startswith("[MOLDEN FORMAT]"):
        return parse_molden(text)
    if "$normal_modes" in text or "$orca_hessian_file" in text:
        return parse_orca_hess(text)
    raise ParseError(
        "Not recognised as an ORCA .hess file (no $normal_modes block) or a Molden "
        "file (no '[Molden Format]' header). For a Gaussian .log use the Gaussian tab.")


def parse_bond_file(text, natoms, name=""):
    """Bonds for the ORCA/Molden path, from either kind of file.

    A Gaussian ``.com``/``.gjf`` deck is read by ``parse_connectivity``.
    Anything else is a plain bond list in the .vsc ``[CONNECTIVITY]`` row
    format -- ``center partner partner ...``, 1-based, so ``i j`` per line
    works too; '#' starts a comment and an optional ``[CONNECTIVITY]`` header
    line is skipped.
    """
    if name.lower().endswith((".com", ".gjf")):
        return parse_connectivity(text, natoms)
    lines = [_strip(line) for line in text.splitlines()]
    lines = [line for line in lines if line and not _SECTION_RE.match(line)]
    if not lines:
        raise ParseError("The bond file is empty -- at least one bond is required.")
    return _parse_vsc_connectivity(lines, natoms)


# ======================================================================
# Path 2: the all-in-one .vsc file
# ======================================================================
_SECTION_RE = re.compile(r"^\s*\[([A-Za-z_]+)\]\s*(.*)$")


def _strip(line):
    """Drop a trailing '#' comment and surrounding whitespace."""
    return line.split("#", 1)[0].strip()


def parse_vsc(text):
    """Read a .vsc file. Returns ``{'atoms','coords','bonds','modes','title'}``."""
    raw = text.splitlines()
    title = ""
    sections, current = {}, None

    for line in raw:
        if line.lstrip().startswith("#"):
            head = line.lstrip("# \t")
            if head.upper().startswith("TITLE"):
                title = head[5:].strip()
            continue
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1).upper()
            sections[current] = {"arg": m.group(2).strip(), "lines": []}
            continue
        if current:
            body = _strip(line)
            if body:
                sections[current]["lines"].append(body)

    for req in ("GEOMETRY", "MODES"):
        if req not in sections:
            raise ParseError(
                f"Missing required [{req}] section. A .vsc file needs [GEOMETRY], "
                "[CONNECTIVITY] and [MODES].")

    atoms, coords = _parse_vsc_geometry(sections["GEOMETRY"]["lines"])
    natoms = len(atoms)

    if "CONNECTIVITY" not in sections:
        raise ParseError(
            "Missing [CONNECTIVITY] section. Without bonds the V-score is undefined "
            "(it would evaluate to 0.000 and label every mode as bending), so this "
            "file is rejected rather than scored.")
    bonds = _parse_vsc_connectivity(sections["CONNECTIVITY"]["lines"], natoms)

    modes = _parse_vsc_modes(sections["MODES"]["lines"], natoms,
                             sections["MODES"]["arg"])

    return {"atoms": atoms, "coords": coords, "bonds": bonds,
            "modes": modes, "title": title}


def _parse_vsc_geometry(lines):
    atoms, coords = [], []
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) < 4:
            raise ParseError(
                f"[GEOMETRY] line {n} has {len(parts)} fields, expected at least 4 "
                f"('index element x y z' or 'element x y z'): {line!r}")
        # Tolerate a leading serial index.
        if len(parts) >= 5 and parts[0].isdigit():
            parts = parts[1:]
        tok = parts[0]
        if tok.isdigit():
            sym = get_symbol(int(tok))
            if sym == "X":
                raise ParseError(f"[GEOMETRY] line {n}: unknown atomic number {tok}.")
        else:
            sym = tok.capitalize()
            if sym.lower() not in atomicMass:
                raise ParseError(
                    f"[GEOMETRY] line {n}: unrecognised element {tok!r}; no atomic "
                    "mass on file. Element identity affects all seven scores, not "
                    "just the V-score, so this cannot be defaulted.")
        try:
            xyz = [float(v) for v in parts[1:4]]
        except ValueError:
            raise ParseError(f"[GEOMETRY] line {n}: coordinates are not numbers: {line!r}")
        atoms.append(sym)
        coords.append(xyz)
    if not atoms:
        raise ParseError("[GEOMETRY] section is empty.")
    return atoms, np.array(coords)


def _parse_vsc_connectivity(lines, natoms):
    bonds = []
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        try:
            center = int(parts[0]) - 1
        except ValueError:
            raise ParseError(f"[CONNECTIVITY] line {n}: first field must be an atom index.")
        for k in range(1, len(parts)):
            tok = parts[k]
            if "." in tok:          # a bond order -- ignored
                continue
            try:
                neighbour = int(tok) - 1
            except ValueError:
                raise ParseError(
                    f"[CONNECTIVITY] line {n}: {tok!r} is neither an atom index nor a "
                    "bond order.")
            if neighbour == center:
                continue
            for a in (center, neighbour):
                if not 0 <= a < natoms:
                    raise ParseError(
                        f"[CONNECTIVITY] line {n} references atom {a + 1}, but the "
                        f"geometry has {natoms} atoms.")
            pair = tuple(sorted((center, neighbour)))
            if pair not in bonds:
                bonds.append(pair)
    if not bonds:
        raise ParseError("[CONNECTIVITY] section is empty -- at least one bond is required.")
    return sorted(bonds)


_MODE_HEAD_RE = re.compile(r"^mode\s+(\d+)\s*(.*)$", re.IGNORECASE)


def _parse_vsc_modes(lines, natoms, declared):
    modes, cur, rows = [], None, []

    def flush():
        if cur is None:
            return
        if len(rows) != natoms:
            raise ParseError(
                f"[MODES] mode {cur['n']} has {len(rows)} displacement rows but the "
                f"geometry has {natoms} atoms.")
        modes.append({"frequency": cur["freq"], "vector": np.array(rows),
                      "is_emit": cur["is_emit"], "reduced_mass": cur["mu"],
                      "force_constant": cur["k"], "irrep": cur["irrep"]})

    for line in lines:
        head = _MODE_HEAD_RE.match(line)
        if head:
            flush()
            rows = []
            meta = _parse_kv(head.group(2))
            # Every one of freq/eigen/mu/k/irrep is optional -- the scorer reads
            # none of them, they are row metadata only. Absent freq stays None
            # rather than defaulting to 0.0, so the UI shows a dash instead of an
            # invented 0.00 cm-1. 'eigen' marks an EMIT mode (already living in
            # the principal-axis frame, so MIT() must not re-rotate its vector --
            # see pipeline.analyse()); it is mutually exclusive with 'freq' in
            # every file this writer produces, but if both are present eigen wins
            # since it is the more specific tag.
            eigen = meta.get("eigen")
            cur = {"n": int(head.group(1)),
                   "freq": eigen if eigen is not None else meta.get("freq"),
                   "mu": meta.get("mu"), "k": meta.get("k"),
                   "irrep": meta.get("irrep"), "is_emit": eigen is not None}
            continue
        if cur is None:
            raise ParseError(
                f"[MODES] displacement row before any 'mode N' header: {line!r}")
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            parts = parts[1:]
        if len(parts) < 3:
            raise ParseError(f"[MODES] mode {cur['n']}: expected 'dx dy dz', got {line!r}")
        try:
            rows.append([float(v) for v in parts[:3]])
        except ValueError:
            raise ParseError(
                f"[MODES] mode {cur['n']}: displacements are not numbers: {line!r}")
    flush()

    if not modes:
        raise ParseError("[MODES] section contains no modes.")
    if declared:
        try:
            want = int(declared.split()[0])
        except (ValueError, IndexError):
            want = None
        if want is not None and want != len(modes):
            raise ParseError(
                f"[MODES] header declares {want} modes but {len(modes)} were found.")
    return modes


def _parse_kv(s):
    """'freq=667.64 irrep=A'' -> {'freq': 667.64, 'irrep': "A'"}"""
    out = {}
    for tok in s.split():
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        key = key.lower()
        if key in ("freq", "eigen", "mu", "k"):
            try:
                out[key] = float(val)
            except ValueError:
                pass
        elif key == "irrep":
            out["irrep"] = val
    return out


# ======================================================================
# The .vsc writer
# ======================================================================
def write_vsc(atoms, coords, bonds, modes, title="", source=""):
    """Serialise a parsed molecule to .vsc text.

    Callers must pass the ORIGINAL frame, not a principal-axis-aligned one.
    MIT() is NOT idempotent: its sign-fix heuristic negates the rotation when
    the heaviest atom's projected coordinates sum negative, so re-aligning an
    already-aligned geometry can flip axes and invert the T/R scores. Passing
    the frame the data arrived in makes a .vsc round trip exact.
    """
    out = [f"#VSCORE {VSC_VERSION}"]
    if title:
        out.append(f"#TITLE   {title}")
    if source:
        out.append(f"#SOURCE  {source}")
    out.append("")

    # Gaussian data is exact at 6 dp (geometry) and 5 dp (hpmodes) and is
    # written exactly as before. Data from ORCA/Molden carries more digits,
    # and truncating them would not round-trip: the T/R scores can move on a
    # 1e-6 A change in geometry (the principal-axis frame is ill-conditioned
    # for symmetric molecules), so as many digits are kept as the data needs.
    cdp = _needed_dp(coords, 6)
    vdp = _needed_dp([np.asarray(m["vector"]) for m in modes], 5)
    cw, vw = cdp + 8, vdp + 7

    out.append("[GEOMETRY] Angstrom")
    for i, (sym, xyz) in enumerate(zip(atoms, coords), 1):
        out.append(f"{i:5d}  {sym:<2s} {xyz[0]:{cw}.{cdp}f} {xyz[1]:{cw}.{cdp}f} "
                   f"{xyz[2]:{cw}.{cdp}f}")
    out.append("")

    out.append("[CONNECTIVITY]")
    partners = {i: [] for i in range(len(atoms))}
    for a, b in bonds:
        partners[a].append(b)
    for i in range(len(atoms)):
        row = "  ".join(str(j + 1) for j in sorted(partners[i]))
        out.append(f"{i + 1:5d}  {row}".rstrip())
    out.append("")

    out.append(f"[MODES] {len(modes)}")
    for n, m in enumerate(modes, 1):
        # Emit only the metadata that exists. All of it is optional on the way
        # in, so writing "freq=0.0000" for a mode that never had a frequency
        # would fabricate one on the way out.
        meta = []
        if m.get("frequency") is not None:
            key = "eigen" if m.get("is_emit") else "freq"
            meta.append(f"{key}={m['frequency']:.4f}")
        if m.get("reduced_mass") is not None:
            meta.append(f"mu={m['reduced_mass']:.4f}")
        if m.get("force_constant") is not None:
            meta.append(f"k={m['force_constant']:.4f}")
        if m.get("irrep"):
            meta.append(f"irrep={m['irrep']}")
        out.append(f"  mode {n}" + ("   " + "   ".join(meta) if meta else ""))
        for i, d in enumerate(np.asarray(m["vector"]), 1):
            out.append(f"{i:5d} {d[0]:{vw}.{vdp}f} {d[1]:{vw}.{vdp}f} {d[2]:{vw}.{vdp}f}")
    out.append("")
    return "\n".join(out)


def _needed_dp(arrays, least, most=15):
    """Fewest decimals in [least, most] that reproduce every value exactly
    (15 is lossless for any coordinate or displacement this pipeline sees)."""
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays]) if len(arrays) else np.zeros(0)
    for dp in range(least, most):
        # exact: the printed value must read back as the identical float
        if all(float(f"{v:.{dp}f}") == v for v in vals):
            return dp
    return most


# ======================================================================
# Precision guard
# ======================================================================
def displacement_precision(modes, sample=4000):
    """Smallest number of decimal places that reproduces every displacement.

    Standard Gaussian output prints 2 dp; ``freq=hpmodes`` prints 5. This
    matters because ``Tscore()`` unit-normalises EACH atom's displacement
    before summing, so an atom printed as '0.00 0.00 0.00' is either masked to
    zero or promoted to a full unit vector depending on rounding noise.
    Measured on real molecules, 2 dp input moves individual T-scores by up to
    0.31 on a [-1,1] scale (V and the S/B/SB labels survive, since the
    stretch/bend gap is 0.73 wide).
    """
    recorded = [m["source_dp"] for m in modes if m.get("source_dp") is not None]
    if recorded:                     # measured before an ORCA/Molden rescale
        return min(recorded)
    vals = []
    for m in modes:
        vals.extend(np.asarray(m["vector"]).ravel().tolist())
        if len(vals) > sample:
            break
    vals = np.array(vals[:sample])
    for dp in range(1, 7):
        if np.allclose(vals, np.round(vals, dp), atol=0, rtol=0):
            return dp
    return 6
