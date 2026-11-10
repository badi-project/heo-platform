"""Tests of the measured-pattern indexing and lattice refinement (src/xrdfit.py).
Agreement with the JavaScript twin is checked by tools/parity_check.py; the accuracy
claims by tools/validate_xrd_fit.py, whose report these tests keep up to date."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import predict  # noqa: E402
import validate_xrd_fit  # noqa: E402
import xrdfit  # noqa: E402

P = predict.Predictor()
CLI = os.path.join(ROOT, "tools", "heo_predict.py")
SPINEL = {e: 0.2 for e in ("Co", "Cr", "Fe", "Mn", "Ni")}
A_TRUE = 8.328


def fit(proto, frac, a, **kw):
    tt, counts = xrdfit.synthetic_pattern(P, proto, frac, a, **kw)
    found, fits = xrdfit.compare_measured(P, tt, counts, frac, kw.get("radiation", "Cu"))
    return tt, counts, found, fits


def write_xy(path, tt, counts, sep=" ", header="", decimal="."):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for x, y in zip(tt, counts):
            fh.write(f"{x:.2f}{sep}{y:.1f}\n".replace(".", decimal))


def test_lattice_parameter_is_recovered_with_zero_shift_and_kalpha2():
    _tt, _c, found, fits = fit("spinel", SPINEL, A_TRUE, zero=0.05, seed=211)
    best = fits[0]
    assert best["proto"] == "spinel" and found["stripped"] and found["quality"] == "clean"
    assert abs(best["a"] / A_TRUE - 1) < 5e-4
    assert abs(best["zero"] - 0.05) < 0.02 and best["fit_zero"]
    assert best["unindexed"] == [] and best["strong_found"] == best["n_strong"]
    # the quoted uncertainty contains the true error and is never below the validated accuracy
    assert abs(best["a"] - A_TRUE) <= best["a_uncertainty"]
    assert best["a_uncertainty"] >= xrdfit.ACCURACY_CLEAN * best["a"]
    assert abs(best["density_g_cm3"] - P.xray_density("spinel", SPINEL, best["a"])) < 1e-12


def test_start_far_from_the_truth():
    """The geometric model starts this rocksalt 4.5 % too low; the search still finds it."""
    frac = {e: 0.2 for e in ("Co", "Fe", "Mg", "Mn", "Zn")}
    _tt, _c, _found, fits = fit("rocksalt", frac, 4.29, seed=223)
    best = fits[0]
    assert best["proto"] == "rocksalt" and abs(best["a0"] / 4.29 - 1) > 0.04
    assert abs(best["a"] / 4.29 - 1) < 5e-4


def test_foreign_lines_are_reported_and_do_not_move_the_lattice():
    _tt, _c, _found, fits = fit("spinel", SPINEL, A_TRUE, seed=227,
                                extra=[(33.15, 10.0), (49.48, 6.0)])
    best = fits[0]
    assert best["proto"] == "spinel" and abs(best["a"] / A_TRUE - 1) < 5e-4
    assert [round(u["tt"], 1) for u in best["unindexed"]] == [33.1, 49.5]
    assert best["explained"] < 1


def test_kalpha2_is_removed_only_when_present():
    with_k2 = fit("spinel", SPINEL, A_TRUE, seed=229)[2]
    without = fit("spinel", SPINEL, A_TRUE, seed=229, kalpha2=False)[2]
    assert with_k2["stripped"] and with_k2["asym"] > xrdfit.XFIT["asymmetry"]
    assert not without["stripped"] and abs(without["asym"]) < 0.05
    # no K-alpha2 peak left over: every peak found sits on a K-alpha1 reflection
    lines = [p["two_theta"] for p in P.simulate_pattern("spinel", SPINEL, A_TRUE)]
    for found in (with_k2, without):
        assert all(min(abs(p["tt"] - t) for t in lines) < 0.05 for p in found["peaks"])


def test_noise_bumps_on_the_flank_of_a_strong_peak_are_not_peaks():
    """Regression: a three-sigma bump on the tail of the (311) line passed for a peak and
    was reported as a secondary phase."""
    for seed in (103, 211, 229, 233):
        best = fit("spinel", SPINEL, A_TRUE, seed=seed)[3][0]
        assert best["unindexed"] == [], seed


def test_wrong_family_is_not_the_best_explanation():
    """A rocksalt pattern for a composition the model sends to spinel."""
    _tt, _c, _found, fits = fit("rocksalt", SPINEL, 4.25, seed=239)
    assert fits[0]["proto"] == "rocksalt" and P.predict(SPINEL)["family"] == "spinel"
    assert fits[0]["score"] > 0.9 > next(f for f in fits if f["proto"] == "spinel")["score"]


def test_noisy_data_get_the_wider_uncertainty():
    _tt, _c, found, fits = fit("spinel", SPINEL, A_TRUE, seed=241, counts=800.0,
                               background=400.0, fwhm=0.3, step=0.04)
    best = fits[0]
    assert found["quality"] == "noisy" and best["proto"] == "spinel"
    assert best["a_uncertainty"] >= xrdfit.ACCURACY_NOISY * best["a"]
    assert abs(best["a"] - A_TRUE) <= best["a_uncertainty"]


def test_cobalt_radiation():
    _tt, _c, _found, fits = fit("spinel", SPINEL, A_TRUE, seed=251, radiation="Co",
                                tt_min=20.0, tt_max=90.0)
    assert fits[0]["proto"] == "spinel" and abs(fits[0]["a"] / A_TRUE - 1) < 5e-4


@pytest.mark.parametrize("sep, header, decimal", [
    (" ", "# scan\n", "."), ("\t", "", "."), (";", "2theta;counts\n", ","), (",", "", "."),
])
def test_text_formats(tmp_path, sep, header, decimal):
    tt, counts = xrdfit.synthetic_pattern(P, "spinel", SPINEL, A_TRUE, seed=257)
    path = tmp_path / "scan.txt"
    write_xy(path, tt, counts, sep, header, decimal)
    got_tt, got = xrdfit.read_pattern(path)
    assert len(got_tt) == len(tt) and abs(got_tt[0] - 10) < 1e-9 and abs(got_tt[-1] - 80) < 1e-9
    assert max(abs(u - round(v, 1)) for u, v in zip(got, counts)) < 1e-9


def test_xrdml_and_garbage():
    xml = ('<xrdMeasurement><positions axis="2Theta" unit="deg"><startPosition>10</startPosition>'
           '<endPosition>12</endPosition></positions><intensities unit="counts">5 6 7 8 9'
           '</intensities></xrdMeasurement>')
    assert xrdfit.parse_pattern(xml) == ([10.0, 10.5, 11.0, 11.5, 12.0], [5.0, 6.0, 7.0, 8.0, 9.0])
    assert xrdfit.parse_pattern("ceci n'est pas une mesure\n\n; commentaire\n") == ([], [])


def test_validation_report_is_the_output_of_its_script():
    assert validate_xrd_fit.main(check=True) == 0
    report = open(os.path.join(ROOT, "QC", "xrd_fit_validation.txt"), encoding="utf-8").read()
    assert "NON ATTEINT" not in report and report.count("Bilan") == 2
    # the report says which set is no longer untouched, and why
    assert "ne vaut donc plus comme jeu vierge" in report


def test_cli_indexes_a_measured_pattern(tmp_path):
    tt, counts = xrdfit.synthetic_pattern(P, "spinel", SPINEL, A_TRUE, seed=263,
                                          extra=[(33.15, 10.0)])
    path = tmp_path / "scan.xy"
    write_xy(path, tt, counts)
    out = subprocess.run([sys.executable, CLI, "(FeCoNiCrMn)3O4", "--measured", str(path)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "s'indexe le mieux en spinelle — la famille prédite" in out.stdout
    assert "1 pic(s) non indexé(s) : 33,1" in out.stdout
    out = subprocess.run([sys.executable, CLI, "(FeCoNiCrMn)3O4", "--measured", str(path), "--json"],
                         capture_output=True, text=True)
    m = json.loads(out.stdout)["measured"]
    assert m["fits"][0]["proto"] == "spinel" and abs(m["fits"][0]["a"] / A_TRUE - 1) < 5e-4
    assert m["kalpha2_removed"] and m["quality"] == "clean"
    bad = tmp_path / "vide.txt"
    bad.write_text("rien\n", encoding="utf-8")
    out = subprocess.run([sys.executable, CLI, "(FeCoNiCrMn)3O4", "--measured", str(bad)],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "au moins 100" in out.stderr
