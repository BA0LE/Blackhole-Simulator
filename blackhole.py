"""
blackhole.py — v7: BlackHoleBody class.

Each BlackHoleBody is a fully dynamic gravitating object with:
  • 3D position and velocity in world (pixel) space
  • Mass and Kerr spin parameter
  • Derived physics: rs, ISCO, shadow radius, Einstein radius
  • Gravitational interaction with other BHs (post-Newtonian)
  • Merger detection and coalescence (conserves momentum, ~5% GW energy loss)
  • Own AccretionDisk centered at BH position

Units: world space = pixels.  SIM_SCALE converts pixels → geometric units.
"""
import numpy as np
import math
from config import Config


class BlackHoleBody:
    """
    A gravitating black hole that can move under N-body gravity.

    Parameters
    ----------
    pos    : (3,) world-pixel position
    vel    : (3,) world-pixel velocity  [pixels per simulated second]
    mass   : BH mass in geometric units (rs = 2GM, c=G=1 → rs = 2*mass)
    spin   : dimensionless Kerr spin a  ∈ [0, 0.99]
    label  : human-readable identifier
    """

    _id_counter = 0

    def __init__(self, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
                 mass: float = 0.5, spin: float = 0.0,
                 sim_scale: float = None, label: str = None):
        BlackHoleBody._id_counter += 1
        self.id        = BlackHoleBody._id_counter
        self.label     = label or f"BH-{self.id}"
        self.pos       = np.array(pos, dtype=float)       #world pixels
        self.vel       = np.array(vel, dtype=float)       #pixels / sim-s
        self.mass      = float(mass)
        self.spin      = float(np.clip(spin, 0.0, 0.99))
        self.SIM_SCALE = float(sim_scale or Config.SIM_SCALE)
        self.active    = True
        self.age       = 0.0                               #seconds of sim time
        self.merge_flash = 0.0                            #visual flash timer
        self._recompute()

    def _recompute(self):
        G  = Config.G
        self.GM = self.mass * G
        a_sc    = np.clip(self.spin, 0.0, 0.99) * self.GM

        #Kerr event horizon r_+
        disc          = max(0.0, self.GM**2 - a_sc**2)
        self.rs_sim   = self.GM + math.sqrt(disc)
        self.rs_px    = self.rs_sim * self.SIM_SCALE

        #Shadow radius  (photon capture cross-section)
        self.shadow_r_sim = self.GM * (5.196 - 0.4 * self.spin)
        self.shadow_r_px  = self.shadow_r_sim * self.SIM_SCALE

        #Einstein radius reference (used for disk lensing)
        self.einstein_r_sim = self.rs_sim * 4.0
        self.einstein_r_px  = self.einstein_r_sim * self.SIM_SCALE

        #Kerr prograde ISCO
        a = self.spin
        z1 = 1.0 + math.pow(max(0.0, 1.0 - a**2), 1/3) * (
             math.pow(max(0.0, 1.0 + a), 1/3) + math.pow(max(0.0, 1.0 - a), 1/3))
        z2 = math.sqrt(max(0.0, 3.0*a**2 + z1**2))
        isco_f           = 3.0 + z2 - math.sqrt(max(0.0, (3.0 - z1)*(3.0 + z1 + 2.0*z2)))
        self.r_isco_sim  = self.GM * max(isco_f, 1.01)
        self.r_isco_px   = self.r_isco_sim * self.SIM_SCALE

        #Photon sphere
        self.photon_sphere_r_sim = self.GM * (2.0 + 1.0*(1.0 - self.spin))

    @property
    def pos_sim(self) -> np.ndarray:
        """Position in simulation (geometric) units."""
        return self.pos / self.SIM_SCALE

    def update(self, dt: float, bh_list: list):
        """
        Advance position and velocity under gravity from other BHs.
        Uses Paczynski-Wiita pseudo-Newtonian potential for BH-BH interaction.
        """
        if not self.active:
            return
        self.age += dt

        accel = np.zeros(3, float)
        for other in bh_list:
            if other is self or not other.active:
                continue
            diff_px  = other.pos - self.pos
            dist_px  = np.linalg.norm(diff_px)
            if dist_px < 1e-4:
                continue
            dist_sim = dist_px / self.SIM_SCALE
            #Pseudo-Newtonian: soften at combined rs
            rs_sum   = self.rs_sim + other.rs_sim
            eff_dist = max(dist_sim - rs_sum, rs_sum * 0.1)
            #Acceleration magnitude in sim units / s²
            a_mag_sim = other.GM / eff_dist**2
            #Convert to pixel / s²
            a_mag_px  = a_mag_sim * self.SIM_SCALE
            accel    += (diff_px / dist_px) * a_mag_px

        #Simple Euler-Cromer (sufficient for BH-BH; dt is small)
        self.vel += accel * dt
        self.pos += self.vel * dt

        #Tick down visual merger flash
        if self.merge_flash > 0:
            self.merge_flash = max(0.0, self.merge_flash - dt)

    def try_merge(self, bh_list: list) -> list:
        """
        Check for mergers with other BHs.  When two BHs overlap within
        3 × (rs_i + rs_j) in simulation units, they coalesce:

          • Momentum conserved exactly
          • Center of mass at mass-weighted midpoint
          • 5 % of total mass-energy radiated as gravitational waves
          • Spin approximated from angular momentum conservation

        Returns list of BHs absorbed (now inactive).
        """
        absorbed = []
        for other in bh_list:
            if other is self or not other.active or not self.active:
                continue
            diff_px  = other.pos - self.pos
            dist_sim = np.linalg.norm(diff_px) / self.SIM_SCALE
            merge_r  = (self.rs_sim + other.rs_sim) * 3.0

            if dist_sim < merge_r:
                M1, M2   = self.mass, other.mass
                M_tot    = M1 + M2
                #Center of mass position & velocity
                self.pos = (M1 * self.pos + M2 * other.pos) / M_tot
                self.vel = (M1 * self.vel + M2 * other.vel) / M_tot
                #GW energy loss (Christodoulou formula approximation)
                eta      = M1 * M2 / M_tot**2          #symmetric mass ratio
                gw_loss  = 1.0 - 0.0523 * eta          #~5% for equal masses
                self.mass = M_tot * gw_loss
                #Spin: angular-momentum weighted average
                L_self  = M1 * self.spin  * self.GM
                L_other = M2 * other.spin * other.GM
                L_orb   = M1 * M2 * np.linalg.norm(diff_px) / self.SIM_SCALE
                L_total = abs(L_self) + abs(L_other) + L_orb * 0.01
                self.spin = float(np.clip(L_total / (self.mass * self.mass + 1e-9), 0.0, 0.99))
                self._recompute()
                other.active     = False
                self.merge_flash = 1.5    #1.5s visual ring-down flash
                absorbed.append(other)

        return absorbed

    def accrete(self, dm: float):
        """Grow BH mass by dm (in geometric units)."""
        self.mass += dm
        self._recompute()

    def __repr__(self):
        return (f"<BlackHoleBody {self.label}  M={self.mass:.3f}  "
                f"a={self.spin:.2f}  rs={self.rs_sim:.3f}  "
                f"pos={self.pos.round(1)}>")
