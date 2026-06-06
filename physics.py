#v7
import numpy as np
import sys, math
sys.path.insert(0, '.')
from computer import Computer
from config   import Config


class Physics(Computer):
    def __init__(self):
        super().__init__()
        self.SIM_SCALE = Config.SIM_SCALE
        self.update_properties(Config.M_BH, getattr(Config, 'SPIN_A', 0.0))

    def update_properties(self, m_bh: float, a: float):
        self.GM = m_bh * Config.G
        a_sc    = np.clip(a, 0.0, 0.99) * self.GM
        disc    = max(0.0001, self.GM**2 - a_sc**2)
        self.rs_sim   = self.GM + math.sqrt(disc)
        self.rs_px    = self.rs_sim * self.SIM_SCALE

        z1 = 1.0 + math.pow(1.0 - a**2, 1/3) * (
             math.pow(1.0 + a, 1/3) + math.pow(1.0 - a, 1/3))
        z2  = math.sqrt(3.0 * a**2 + z1**2)
        isco_f         = 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0*z2))
        self.r_isco_sim = self.GM * max(isco_f, 1.01)
        self.r_isco_px  = self.r_isco_sim * self.SIM_SCALE

        self.photon_sphere_r_sim = self.GM * (2.0 + 1.0*(1.0 - a))
        self.shadow_r_sim  = self.GM * (5.196 - 0.4*a)
        self.einstein_r_sim = self.rs_sim * 4.0
        self.spin_a = a

    @staticmethod
    def multi_bh_accel(pos_sim: np.ndarray, bh_list: list) -> np.ndarray:
        accel = np.zeros_like(pos_sim)
        for bh in bh_list:
            if not bh.active: continue
            bh_sim = bh.pos / Config.SIM_SCALE
            diff   = bh_sim - pos_sim
            dist   = np.linalg.norm(diff, axis=1)
            r_eff  = np.maximum(dist - bh.rs_sim, bh.rs_sim * 0.05)
            a_mag  = bh.GM / r_eff**2
            safe   = dist > 1e-6
            unit   = np.where(safe[:,None], diff / np.maximum(dist, 1e-9)[:,None], 0.0)
            accel += unit * a_mag[:,None]
        return accel

    @staticmethod
    def multi_bh_accel_single(pos_sim: np.ndarray, bh_list: list) -> np.ndarray:
        accel = np.zeros(3, float)
        for bh in bh_list:
            if not bh.active: continue
            bh_sim  = bh.pos / Config.SIM_SCALE
            diff    = bh_sim - pos_sim
            dist    = np.linalg.norm(diff)
            if dist < 1e-6: continue
            r_eff   = max(dist - bh.rs_sim, bh.rs_sim * 0.05)
            a_mag   = bh.GM / r_eff**2
            accel  += diff / dist * a_mag
        return accel

    def circular_L(self, r):
        r = np.asarray(r, float)
        return r / np.sqrt(np.maximum(2.0*r - 3.0, 1e-6))
        

    def geodesic_accel(self, r, L):
        r  = np.maximum(np.asarray(r, float), 0.52)
        L  = np.asarray(L, float)
        GM = self.GM
        return -GM/r**2 + L**2/r**3 - 3.0*GM*L**2/r**4

    def rk4_step(self, r, phi, pr, L, dt):
        r = np.maximum(r, 0.52)
        def deriv(r_, phi_, pr_):
            rs = np.maximum(r_, 0.52)
            return pr_, L/rs**2, self.geodesic_accel(rs, L)
        k1 = deriv(r,               phi,               pr)
        k2 = deriv(r+.5*dt*k1[0],  phi+.5*dt*k1[1],  pr+.5*dt*k1[2])
        k3 = deriv(r+.5*dt*k2[0],  phi+.5*dt*k2[1],  pr+.5*dt*k2[2])
        k4 = deriv(r+   dt*k3[0],  phi+   dt*k3[1],  pr+   dt*k3[2])
        f  = dt/6.0
        return (r   + f*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
                phi + f*(k1[1]+2*k2[1]+2*k3[1]+k4[1]),
                pr  + f*(k1[2]+2*k2[2]+2*k3[2]+k4[2]))

    

    def disk_temperature(self, r_sim):
        r    = np.asarray(r_sim, float)
        r_in = self.r_isco_sim
        T    = np.zeros_like(r)
        v    = r > r_in * 1.001
        x    = np.sqrt(r_in / r[v])
        T[v] = (r_in / r[v])**0.75 * np.maximum(0.0, 1.0 - x)**0.25
        return T

    def temperature_to_rgb(self, T):
        T   = np.clip(np.ravel(np.asarray(T, float)), 0.0, 1.0)
        rgb = np.zeros((len(T), 3), float)
        def band(lo, hi, r0,g0,b0, r1,g1,b1):
            m = (T >= lo) & (T < hi)
            if not m.any(): return
            t = (T[m]-lo)/(hi-lo)
            rgb[m] = np.stack([r0+(r1-r0)*t, g0+(g1-g0)*t, b0+(b1-b0)*t], 1)
        band(0.00,0.08,   0, 0, 0,   80,  4,  0)
        band(0.08,0.20,  80, 4, 0,  210, 18,  0)
        band(0.20,0.38, 210,18, 0,  255, 85,  4)
        band(0.38,0.58, 255,85, 4,  255,200,  8)
        band(0.58,0.78, 255,200,8,  255,255,170)
        band(0.78,1.00, 255,255,170,215,228,255)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    #redshift

    def grav_redshift(self, r_sim):
        r = np.maximum(np.asarray(r_sim, float), self.rs_sim + 0.01)
        return np.sqrt(np.maximum(0.0, 1.0 - self.rs_sim / r))

    #Doppler

    def doppler_shift(self, colors, beta):
        beta = np.clip(np.asarray(beta, float), -0.97, 0.97)
        D    = np.sqrt((1.0+beta)/(1.0-beta))
        D3   = D**3
        c    = colors.astype(float)
        bm   = D > 1.0;  rm = ~bm
        c[bm,0] = c[bm,0]*D3[bm]*0.58
        c[bm,1] = c[bm,1]*D3[bm]*0.86
        c[bm,2] = c[bm,2]*D3[bm]*1.28 + 28*(D3[bm]-1)
        c[rm,0] = c[rm,0]*D3[rm]
        c[rm,1] = c[rm,1]*D3[rm]*0.68
        c[rm,2] = c[rm,2]*D3[rm]*0.38
        return np.clip(c, 0, 255).astype(np.uint8)

    #

    def lens_all(self, sx, sy, cx, cy, theta_E, shadow_r_px):
        θE   = theta_E
        dx   = sx-cx;  dy = sy-cy
        # Clamp β ≥ 0.5 like v6 — ensures |θn| is never numerically zero,
        # so p2 is always correctly positioned on the opposite side of the BH.
        β    = np.maximum(np.sqrt(dx*dx+dy*dy), 0.5)
        nx   = dx/β;   ny = dy/β
        disc = np.sqrt(β*β + 4*θE*θE)
        θp   = (β+disc)*0.5;  θn = (β-disc)*0.5   # θn < 0  → p2 on opposite side
        p1x  = cx+nx*θp;  p1y = cy+ny*θp
        p2x  = cx+nx*θn;  p2y = cy+ny*θn
        u    = np.maximum(β/θE, 0.01)
        tmp  = (u*u+2)/(2*u*np.sqrt(u*u+4))
        mu1  = np.minimum(tmp+0.5, 12.0)
        mu2  = np.minimum(np.maximum(tmp-0.5, 0.02), 6.0)
        # d2 = |θn| = distance from BH centre to p2 (since nx,ny is unit vector)
        d2   = np.abs(θn)
        vis2 = d2 > shadow_r_px + 2.0
        return p1x, p1y, p2x, p2y, mu1, mu2, vis2

    def lens_stars(self, sx, sy, cx, cy, theta_E):
        θE = theta_E
        dx = sx-cx;  dy = sy-cy
        β  = np.maximum(np.sqrt(dx*dx+dy*dy), 0.5)   # clamp like v6
        nx = dx/β;   ny = dy/β
        θp = (β+np.sqrt(β*β+4*θE*θE))*0.5
        u  = np.maximum(β/θE, 0.01)
        mu = np.minimum((u*u+2)/(u*np.sqrt(u*u+4)), 10.0)
        return (cx+nx*θp).astype(int), (cy+ny*θp).astype(int), mu
