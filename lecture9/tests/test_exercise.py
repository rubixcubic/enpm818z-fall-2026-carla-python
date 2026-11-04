"""The in-class pieces: candidates.csv, the extra cost hook and l9_check on the stub."""
import csv

import numpy as np

import export_candidates as e
import l9_check
import l9common as c


def test_csv_matches_the_slides(tmp_path):
    out = tmp_path / "c.csv"
    e.main(["--out", str(out)])
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 31
    assert sum(1 for r in rows if r["dropped"]) == 22
    first = rows[0]
    assert first["name"] == "d_end 3.5 m, 40 m"
    assert float(first["peak_lat_accel_mps2"]) == 1.253
    assert float(first["smallest_gap_m"]) == 1.359


def test_committed_csv_is_current(tmp_path):
    out = tmp_path / "c.csv"
    e.main(["--out", str(out)])
    assert out.read_text() == (e.HERE / "candidates.csv").read_text()


def test_extra_term_adds_to_every_total():
    ref = c.load_scene()
    car = c.parked_car(ref)
    plain = c.plan(ref, c.SCENE["av_s"], (0.0, 0.0, 0.0), [car])
    plus = c.plan(ref, c.SCENE["av_s"], (0.0, 0.0, 0.0), [car], extra=lambda p: 1.0)
    kept = [sc["total"] for _, sc in plain if np.isfinite(sc["total"])]
    kept_plus = [sc["total"] for _, sc in plus if np.isfinite(sc["total"])]
    assert np.allclose(np.array(kept) + 1.0, kept_plus)
    assert all(sc["terms"]["extra"] == 1.0 for _, sc in plus)


def test_check_reports_the_stub_as_not_done(capsys):
    assert l9_check.main() == 1
    assert "Not done yet" in capsys.readouterr().out
