"""Step 6: sensor faults injected into the simulated measurements.

Each fault is a list of time intervals. The filter is never told about them:
it has to cope through its own innovation tests (gating), or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Faults:
    # GNSS receiver loses its fix: ((t0, t1), ...). The fixes are still drawn
    # (same random sequence as without the outage) but not delivered.
    gnss_outage: tuple = ()
    # GNSS position jumps, as with multipath or a bad satellite:
    # ((t0, t1, (dN, dE, dD) [m]), ...). The velocity is not affected.
    gnss_jump: tuple = ()
    # Local magnetic disturbance (power line, metal structure, motor current)
    # added to the Earth field, in NED: ((t0, t1, (mN, mE, mD) [gauss]), ...).
    mag_disturbance: tuple = ()

    def gnss_lost(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.gnss_outage)

    def gnss_offset(self, t: float) -> np.ndarray:
        return _sum_active(self.gnss_jump, t)

    def mag_offset(self, t: float) -> np.ndarray:
        return _sum_active(self.mag_disturbance, t)


def _sum_active(intervals, t: float) -> np.ndarray:
    out = np.zeros(3)
    for a, b, v in intervals:
        if a <= t < b:
            out += np.asarray(v, dtype=float)
    return out


NONE = Faults()
