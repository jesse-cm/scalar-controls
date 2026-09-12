"""Tests: learned sun-sensor front end (small, fast training)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalar.learning.sun_mlp import (angular_error_deg, export_c_header, flight_like_sensor,  # noqa: E402
                                     make_dataset, train_sun_mlp)


def test_mlp_beats_analytic_under_albedo():
    sensor = flight_like_sensor(np.random.default_rng(1))
    mlp, val_err = train_sun_mlp(sensor, extra=True, n_train=30000, n_val=3000, epochs=4, seed=0)
    Xc, Xe, y = make_dataset(sensor, 2000, np.random.default_rng(9), albedo_max=0.5)
    fs = sensor.full_scale_current(1361.0)
    analytic = np.array([sensor.reconstruct(c * fs, 1361.0)[0] for c in Xc])
    e_an = np.median(angular_error_deg(analytic, y))
    e_nn = np.median(angular_error_deg(mlp(Xc, Xe), y))
    assert e_nn < e_an, (e_nn, e_an)
    assert mlp.n_params < 2000
    out = mlp(Xc[:5], Xe[:5])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_reconstruct_gate_and_export():
    sensor = flight_like_sensor(np.random.default_rng(2))
    mlp, _ = train_sun_mlp(sensor, extra=False, n_train=5000, n_val=500, epochs=1, seed=0)
    s_hat, valid = mlp.reconstruct(np.zeros(6), 1361.0, sensor)
    assert not valid and s_hat is None
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.h"); export_c_header(mlp, p)
        txt = open(p).read()
        assert "sun_mlp_W0" in txt and "SUN_MLP_LAYERS 3" in txt


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"  ok  {f.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {f.__name__}: {e}")
    print(f"{ok}/{len(fns)} passed")
