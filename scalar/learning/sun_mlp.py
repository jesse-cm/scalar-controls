"""Learned coarse-sun-sensor reconstruction (small MLP) for the MEKF front end.

WHY A NETWORK HERE AND NOWHERE ELSE
-----------------------------------
The MEKF already averages zero-mean sensor noise optimally; a network cannot
beat it there. What the MEKF cannot remove is *systematic* error, and for this
sensor suite the dominant sun-side systematic is Earth albedo: a second
Lambertian source (nadir) superposed on the six face photocurrents. The
analytic opposing-pair reconstruction  s ~ [I+x-I-x, I+y-I-y, I+z-I-z]  has no
way to separate the two sources, so the recovered sun vector is pulled toward
nadir by up to ~10 deg. Per-diode gain mismatch, cover-glass roll-off and face
tilt add a few more degrees of fixed error.

A tiny MLP from the raw currents to a unit sun vector learns exactly that
nonlinear, geometry-specific inverse. Two variants:

  * ``SunMLP(extra=False)``  — inputs: the 6 normalised currents only.
  * ``SunMLP(extra=True)``   — plus the *predicted* nadir direction in the body
    frame (from the MEKF's current attitude and the propagated orbit) and the
    dayside factor. This makes albedo separation well-posed; the network only
    has to learn the albedo *magnitude*, which the analytic model cannot.

The output feeds ``MEKF.update_sun`` unchanged, so the covariance machinery,
eclipse handling and gyro fusion are untouched. The network is a sensor model,
not an estimator.

SIZE
----
6(+4) -> 32 -> 32 -> 3 is ~1.4k parameters, ~1.4k MACs per sample. At 1 Hz that
is nothing on the RP2350 (Cortex-M33 + FPU) and still trivial on the
STM32L031 (Cortex-M0+, no FPU, ~30 us in soft float). Weights fit in 6 KB
float32 or 1.5 KB int8. ``export_c_header`` writes them out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sensors.sun_sensor import SunSensorArray
from ..sensors.veml6031 import VEML6031Array

__all__ = ["make_dataset", "SunMLP", "train_sun_mlp", "angular_error_deg",
           "export_c_header", "flight_like_sensor", "sensor_inputs", "annual_irradiance_range"]


def flight_like_sensor(rng, gain_sigma=0.05, rolloff=0.2, tilt_deg=2.0):
    """One physical sensor instance with fixed (unknown-to-us) systematics.

    The network is trained *for this unit*, the way a flight net would be
    trained from that unit's ground sun-simulator sweep plus a simulated
    albedo model. Returns the SunSensorArray whose ``true_*`` fields carry the
    systematics while ``normals`` stays nominal (what the analytic path uses).
    """
    s = SunSensorArray()
    n = len(s.normals)
    s.gains = 1.0 + gain_sigma * rng.standard_normal(n)
    s.rolloff = rolloff
    tilt = np.deg2rad(tilt_deg) * rng.standard_normal((n, 3))
    tn = s.normals + tilt
    s.true_normals = tn / np.linalg.norm(tn, axis=1, keepdims=True)
    return s


def _random_unit(rng, n):
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sensor_inputs(sensor, raw, irradiance):
    """Raw sensor output -> the network's inputs, one per sensor, face-on Sun = 1.

    SunSensorArray: photocurrents / full-scale current (as before).
    VEML6031Array (plan row 1f): the seven counts, converted to lux and divided
    by the face-on lux at ``irradiance``. A clipped sensor therefore reads
    65535 * resolution / face-on lux (0.43 at 25 ms and 1 AU), and that plateau
    moves with the irradiance; an unclipped one reads its cosine.
    """
    raw = np.asarray(raw, float)
    if np.ndim(irradiance) == 1:                # one irradiance per row of raw
        irradiance = np.asarray(irradiance, float)[:, None]
    if isinstance(sensor, VEML6031Array):
        return raw * sensor.resolution / sensor.face_on_lux(irradiance)
    return raw / sensor.full_scale_current(irradiance)


def annual_irradiance_range(n=366):
    """(min, max) of environment.sun.solar_irradiance over one year [W/m^2]."""
    from ..environment.sun import solar_irradiance
    E = [solar_irradiance(2451545.0 + 365.25 * k / n) for k in range(n)]
    return float(min(E)), float(max(E))


def make_dataset(sensor, n, rng, albedo_max=0.5, irradiance=1361.0):
    """(X_currents (n,6), X_extra (n,4), y_sun (n,3)) with random sun/nadir/albedo.

    ``irradiance`` is one value, or ``(lo, hi)`` for a uniform draw per sample
    (drawn after everything else, so a single value consumes ``rng`` as before).
    For a VEML6031Array X is (n,7), see ``sensor_inputs``.
    """
    s = _random_unit(rng, n)
    nadir = _random_unit(rng, n)
    coeff = rng.uniform(0.0, albedo_max, n)
    dayside = rng.uniform(0.0, 1.0, n)
    if np.ndim(irradiance) == 0:
        E = [irradiance] * n
    else:
        E = rng.uniform(irradiance[0], irradiance[1], n)
    X = np.empty((n, len(sensor.true_normals)))
    for i in range(n):
        X[i] = sensor.measure(s[i], E[i], True, rng,
                              albedo_dir_body=nadir[i],
                              albedo_irradiance=coeff[i] * E[i] * dayside[i])
    X = sensor_inputs(sensor, X, irradiance if np.ndim(irradiance) == 0 else E)
    extra = np.hstack([nadir, dayside[:, None]])
    return X.astype(np.float32), extra.astype(np.float32), s.astype(np.float32)


def angular_error_deg(a, b):
    d = np.clip(np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1)
                                          * np.linalg.norm(b, axis=-1)), -1, 1)
    return np.degrees(np.arccos(d))


@dataclass
class SunMLP:
    """NumPy inference wrapper (what flight would run) around trained weights."""

    weights: list          # [(W1,b1),(W2,b2),(W3,b3)] numpy float32
    extra: bool

    @property
    def n_params(self):
        return int(sum(W.size + b.size for W, b in self.weights))

    @property
    def n_macs(self):
        return int(sum(W.size for W, _ in self.weights))

    def __call__(self, x_currents, x_extra=None):
        x = np.asarray(x_currents, np.float32)
        if self.extra:
            x = np.concatenate([x, np.asarray(x_extra, np.float32)], axis=-1)
        for i, (W, b) in enumerate(self.weights):
            x = x @ W.T + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(n, 1e-9)

    def reconstruct(self, currents, irradiance, sensor, x_extra=None):
        """Drop-in for SunSensorArray.reconstruct: (s_hat, valid)."""
        c = np.asarray(currents, float)
        if isinstance(sensor, VEML6031Array):
            # Eclipse gate without the face layout (the opposing-pair difference below assumes
            # one): lit when the brightest sensor reads more than min_signal_frac of face-on Sun.
            x = sensor_inputs(sensor, c, irradiance)
            if np.max(x) <= sensor.base.min_signal_frac:
                return None, False
            return self(x, x_extra), True
        fs = sensor.full_scale_current(irradiance)
        v = np.array([c[0] - c[1], c[2] - c[3], c[4] - c[5]])
        if np.linalg.norm(v) <= sensor.min_signal_frac * fs:   # same eclipse gate
            return None, False
        return self(c / fs, x_extra), True


def train_sun_mlp(sensor, extra=False, n_train=200_000, n_val=20_000, hidden=32,
                  epochs=20, seed=0, albedo_max=0.5, verbose=False, irradiance=None):
    """Train the MLP with torch; return (SunMLP, val_error_deg).

    ``irradiance`` goes to ``make_dataset``. None: 1361.0 for the photodiode
    cube (its inputs do not depend on it); for a VEML6031Array the range over a
    year, because the clip plateau in the inputs moves with the irradiance.
    """
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if irradiance is None:
        irradiance = annual_irradiance_range() if isinstance(sensor, VEML6031Array) else 1361.0
    Xc, Xe, y = make_dataset(sensor, n_train + n_val, rng, albedo_max, irradiance)
    X = np.hstack([Xc, Xe]) if extra else Xc
    Xt, yt = torch.tensor(X), torch.tensor(y)
    Xtr, ytr, Xva, yva = Xt[:n_train], yt[:n_train], Xt[n_train:], yt[n_train:]

    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 3))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def loss_fn(out, tgt):
        out = out / out.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return (1.0 - (out * tgt).sum(dim=1)).mean()

    bs = 512
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        if verbose:
            with torch.no_grad():
                print(f"  epoch {ep+1:2d}  val loss {loss_fn(net(Xva), yva):.5f}")

    weights = []
    for m in net:
        if isinstance(m, torch.nn.Linear):
            weights.append((m.weight.detach().numpy().astype(np.float32),
                            m.bias.detach().numpy().astype(np.float32)))
    mlp = SunMLP(weights, extra)
    pred = mlp(Xva.numpy()[:, :Xc.shape[1]], Xva.numpy()[:, Xc.shape[1]:] if extra else None)
    return mlp, angular_error_deg(pred, yva.numpy())


def export_c_header(mlp, path, name="sun_mlp"):
    """Write the weights as a C header (float32) for the flight build."""
    with open(path, "w") as f:
        f.write(f"/* generated by scalar.learning.sun_mlp — {mlp.n_params} params, "
                f"{mlp.n_macs} MACs, extra_inputs={int(mlp.extra)} */\n")
        f.write(f"#define {name.upper()}_LAYERS {len(mlp.weights)}\n")
        for i, (W, b) in enumerate(mlp.weights):
            f.write(f"static const int {name}_shape{i}[2] = {{{W.shape[0]}, {W.shape[1]}}};\n")
            f.write(f"static const float {name}_W{i}[] = {{"
                    + ",".join(f"{v:.7g}f" for v in W.ravel()) + "};\n")
            f.write(f"static const float {name}_b{i}[] = {{"
                    + ",".join(f"{v:.7g}f" for v in b.ravel()) + "};\n")
