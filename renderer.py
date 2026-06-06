import math
import numpy as np
import pygame
#v7
_TWO_PI        = 2.0 * math.pi
_STAR_SPHERE_R = 80_000.0


class Renderer:
    def __init__(self, W: int, H: int, physics):
        self.W    = W
        self.H    = H
        self.phys = physics

        pygame.font.init()
        self.font     = pygame.font.SysFont("monospace", 12)
        self.font_med = pygame.font.SysFont("monospace", 14)
        self.N_ang    = 720

        from config import Config
        self.rebuild_stars(getattr(Config, 'STAR_COUNT', 3400))

    # ── stars ────────────────────────────────────────────────────────────────

    def rebuild_stars(self, NS):
        NS  = max(0, int(NS))
        rng = np.random.default_rng(99 + NS)

        b    = np.clip(rng.exponential(0.18, NS), 0.0, 1.0)
        tp   = rng.choice(3, NS, p=[0.54, 0.28, 0.18])
        base = np.zeros((NS, 3), float)
        base[tp==0] = [255, 255, 255]
        base[tp==1] = [155, 200, 255]
        base[tp==2] = [255, 240, 165]
        self.s_raw = np.clip(base * b[:,None], 0, 255).astype(np.uint8)

        #unified 3-D positions - same data for both render modes
        cos_t = rng.uniform(-1.0, 1.0, NS)
        phi_s = rng.uniform(0.0,  _TWO_PI, NS)
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
        R     = _STAR_SPHERE_R
        self.s_pos3d = np.stack([
            R * sin_t * np.cos(phi_s),
            R * cos_t,
            R * sin_t * np.sin(phi_s),
        ], axis=1).astype(float)

        # 2-D fallback derived deterministically from the same angles → no flicker on toggle
        self.s_sx = ((phi_s / _TWO_PI) * self.W).astype(float)
        self.s_sy = ((cos_t * 0.5 + 0.5) * self.H).astype(float)

    # ── fast scatter ─────────────────────────────────────────────────────────

    def _scatter(self, buf, xs, ys, cols):
        x = xs.astype(int); y = ys.astype(int)
        v = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        if not v.any(): return
        np.add.at(buf, (x[v], y[v]), cols[v].astype(np.int16))

    # ── thin-lens deflection (shared helper) ─────────────────────────────────

    @staticmethod
    def _apply_lensing(lx, ly, bh_projs, dists=None, only_behind=False):
        """
        Branchless thin-lens: θp = (β + √(β²+4θE²))/2
        only_behind=True: weight by sign(dists - bh_dist) so only
        particles behind the BH are deflected.
        """
        for bhx, bhy, theta_E, bh_depth, _sr, _bh, bh_dist in bh_projs:
            if theta_E < 0.5:
                continue
            dx = lx - bhx
            dy = ly - bhy
            β  = np.sqrt(dx*dx + dy*dy + 1e-6)
            θp = (β + np.sqrt(β*β + 4.0*theta_E*theta_E)) * 0.5
            delta = θp - β                          # how much to shift outward

            if only_behind and dists is not None:
                # smooth weight: 1 if behind, 0 if in front - no branch
                w = np.clip((dists - bh_dist) * 1e6, 0.0, 1.0)
            else:
                w = 1.0

            lx = lx + (dx / β) * delta * w
            ly = ly + (dy / β) * delta * w
        return lx, ly

    # ── stars ─────────────────────────────────────────────────────────────────

    def _draw_stars(self, buf, camera, bh_projs):
        from config import Config
        B = getattr(Config, 'GLOBAL_BRIGHTNESS', 1.0)

        if getattr(Config, 'REALISTIC_STARS', True):
            sx, sy, _d, fwd = camera.project_batch(self.s_pos3d, self.W, self.H)
            lx = sx[fwd].copy(); ly = sy[fwd].copy()
            raw = self.s_raw[fwd]
        else:
            lx = self.s_sx.copy(); ly = self.s_sy.copy()
            raw = self.s_raw

        lx, ly = self._apply_lensing(lx, ly, bh_projs, only_behind=False)

        inb = (lx >= 0) & (lx < self.W) & (ly >= 0) & (ly < self.H)
        c   = np.clip(raw[inb].astype(float) * B, 0, 255).astype(np.int16)
        np.add.at(buf, (lx[inb].astype(int), ly[inb].astype(int)), c)

    # ── particles ─────────────────────────────────────────────────────────────

    def _draw_particles(self, buf, camera, disk_list, free, side, bh_projs):
        from config import Config
        show_disk = getattr(Config, 'USE_VIRTUAL_ACCRETION_DISK', True)
        ref_dist = bh_projs[0][6] if bh_projs else 99999

        passes = []
        if show_disk:
            for disk in disk_list:
                if disk.bh.active and disk.n > 0:
                    passes.append((disk.positions_3d(),
                                   disk.colors_frame(camera),
                                   disk.alpha.copy()))
        if free.n > 0:
            passes.append((free.positions_3d(),
                           free.colors_frame(),
                           np.full(free.n, 0.75)))

        for pos3, cols, alpha in passes:
            if len(pos3) == 0: continue
            sx, sy, depths, fwd = camera.project_batch(pos3, self.W, self.H)

            dists = np.linalg.norm(pos3 - camera.position, axis=1)
            mask = fwd & (dists >= ref_dist if side == 'far' else dists < ref_dist)
            idx  = np.where(mask)[0]
            if not len(idx): continue

            lx = sx[idx].copy(); ly = sy[idx].copy()

            if side == 'far':
                #deflect only particles that are behind the BH
                lx, ly = self._apply_lensing(lx, ly, bh_projs,
                                              dists=dists[idx], only_behind=True)
            # near side: NO suppression - dark centre comes from disk geometry
            # (no particles inside r_ISCO) + photon ring, not from painting black

            a = alpha[idx]
            c = (cols[idx].astype(float) * a[:,None]).astype(np.uint8)
            self._scatter(buf, lx, ly, c)

            # glow halo
            glow_w = np.clip((a - 0.40) / 0.60, 0.0, 1.0) * 0.25
            gc     = (c.astype(float) * glow_w[:,None]).astype(np.uint8)
            for ddx, ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                self._scatter(buf, lx+ddx, ly+ddy, gc)

    # ── photon ring ───────────────────────────────────────────────────────────

    def _draw_photon_ring(self, buf, camera, disk, bhx, bhy, shadow_r_px, bh_dist):
        N   = self.N_ang
        pos = disk.positions_3d()
        col = disk.colors_frame(camera)
        alp = disk.alpha

        sx, sy, depths, fwd = camera.project_batch(pos, self.W, self.H)
        dists = np.linalg.norm(pos - camera.position, axis=1)
        valid = np.where(fwd)[0]
        if not len(valid): return

        dx  = sx[valid] - bhx; dy = sy[valid] - bhy
        phi = np.arctan2(dy, dx) % _TWO_PI

        # far/near weights - continuous via distance sign, no branch
        t_far  = np.clip((dists[valid] - bh_dist) / (abs(bh_dist)*0.01 + 1.0), 0.0, 1.0)
        w_far  = t_far * 3.2
        w_near = (1.0 - t_far) * 0.6

        bins_far  = ((phi / _TWO_PI) * N).astype(int) % N
        bins_near = ((phi + math.pi*0.90) / _TWO_PI * N).astype(int) % N

        ring_col = np.zeros((N, 3), float)
        idm = valid
        for ch in range(3):
            np.add.at(ring_col[:,ch], bins_far,  alp[idm]*w_far *col[idm,ch].astype(float))
            np.add.at(ring_col[:,ch], bins_near, alp[idm]*w_near*col[idm,ch].astype(float))

        mx = ring_col.max()
        if mx < 1e-6: return
        ring_col = np.clip(ring_col / mx * 220.0, 0, 255)

        ang = np.linspace(0, _TWO_PI, N, endpoint=False)
        ca  = np.cos(ang); sa = np.sin(ang)
        for rr, wt in ((shadow_r_px-.5, 0.55),(shadow_r_px+.5, 1.00),(shadow_r_px+1.5, 0.40)):
            rx = (bhx + rr*ca).astype(int); ry = (bhy + rr*sa).astype(int)
            v  = (rx >= 0) & (rx < self.W) & (ry >= 0) & (ry < self.H)
            if v.any():
                np.add.at(buf, (rx[v],ry[v]), (ring_col[v]*wt).astype(np.int16))

    # ── celestial bodies ──────────────────────────────────────────────────────

    def _draw_celestial_body(self, surface, camera, obj, bh_projs, side):
        from config import Config

        pts_3d = obj.get_silhouette_points(camera, n_points=50)
        if pts_3d is None: return
        c_proj = camera.project_single(obj.pos, self.W, self.H)
        if c_proj is None: return
        _, _, obj_depth = c_proj

        if bh_projs:
            bh_depth = bh_projs[0][3]
            shadow_r = bh_projs[0][4]
            # Near layer is anything closer to camera than (BH depth - 1.5*Shadow Radius)
            # Matches v6 logic exactly
            is_near = obj_depth < (bh_depth - shadow_r * 1.5)
        else:
            is_near = True

        if (side=='near' and not is_near) or (side=='far' and is_near):
            return

        sx, sy, depths, fwd_mask = camera.project_batch(pts_3d, self.W, self.H)
        if not fwd_mask.all():
            return

        if bh_projs and side == 'far':
            bhx, bhy, theta_E, bh_depth, shadow_r_px, _bh, bh_dist = bh_projs[0]
            if bh_depth < 99999 and obj_depth > bh_depth and theta_E > 1e-4:
                d_ls = obj_depth - bh_depth   # lens–source distance
                d_os = max(obj_depth, 0.01)   # observer–source

                # Weak-field Einstein angle (standard thin-lens formula)
                lthE_weak = theta_E * math.sqrt(d_ls / d_os)

                # Strong-field correction: as d_ls → 0 (object near BH),
                # deflection angle diverges like α ≈ 4GM/b.  We add a boost
                # proportional to rs_px / d_ls so bending grows continuously.
                rs_px = shadow_r_px / 2.598  # shadow_r ≈ 2.598 * rs for Schwarzschild
                strong_boost = rs_px / max(d_ls, rs_px * 0.25)
                lthE = lthE_weak * (1.0 + 1.8 * strong_boost)

                if lthE > 1e-4:
                    p1x, p1y, p2x, p2y, _, _, _ = self.phys.lens_all(
                        sx, sy, bhx, bhy, lthE, shadow_r_px)
                    # Draw both p1 and p2 unconditionally — v6 behaviour.
                    # BH shadow (painted in render step 5) provides natural occlusion.
                    for px_arr, py_arr in ((p1x, p1y), (p2x, p2y)):
                        pts = np.vstack([np.column_stack((px_arr, py_arr)),
                                         np.column_stack((px_arr, py_arr))[:1]])
                        xf, yf = self.phys.CubicSpline(pts)
                        self._draw_alpha_polygon(surface, obj.color, list(zip(xf, yf)))
                    return

        pts = np.vstack([np.column_stack((sx, sy)),
                          np.column_stack((sx, sy))[:1]])
        xf, yf = self.phys.CubicSpline(pts)
        self._draw_alpha_polygon(surface, obj.color, list(zip(xf, yf)))

    def _draw_alpha_polygon(self, surface, color, points):
        if not points: return
        alpha = color[3] if len(color) >= 4 else 255
        if alpha == 0: return
        if alpha == 255:
            pygame.draw.polygon(surface, color[:3], points); return
        min_x = int(min(p[0] for p in points)); max_x = int(max(p[0] for p in points))
        min_y = int(min(p[1] for p in points)); max_y = int(max(p[1] for p in points))
        w = max_x-min_x+6; h = max_y-min_y+6
        if 0 < w < self.W and 0 < h < self.H:
            ov = pygame.Surface((w,h), pygame.SRCALPHA)
            pygame.draw.polygon(ov, color, [(p[0]-min_x+3,p[1]-min_y+3) for p in points])
            surface.blit(ov, (min_x-3, min_y-3))

    # ── merger flash ──────────────────────────────────────────────────────────

    def _draw_merger_flash(self, surface, bh, camera):
        res = camera.project_single(bh.pos, self.W, self.H)
        if res is None: return
        bhx, bhy, bh_depth = res
        r_flash = camera.get_screen_radius(
            bh.shadow_r_sim*bh.SIM_SCALE*4.0, bh_depth, self.W, self.H)
        alpha = int(bh.merge_flash / 1.5 * 200)
        if r_flash < 1: return
        fs = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for dr in range(5):
            pygame.draw.circle(fs, (255,200,100,max(0,alpha-dr*40)),
                               (int(bhx),int(bhy)), int(r_flash)+dr, 2)
        surface.blit(fs, (0,0))

    # ── bloom ─────────────────────────────────────────────────────────────────

    def _bloom(self, surface):
        W, H = self.W, self.H
        sm  = pygame.transform.smoothscale(surface,(W//4,H//4))
        bl  = pygame.transform.smoothscale(sm,(W,H)); bl.set_alpha(76)
        surface.blit(bl,(0,0),special_flags=pygame.BLEND_ADD)
        sm2 = pygame.transform.smoothscale(surface,(W//2,H//2))
        bl2 = pygame.transform.smoothscale(sm2,(W,H)); bl2.set_alpha(22)
        surface.blit(bl2,(0,0),special_flags=pygame.BLEND_ADD)

    # ── config panel ──────────────────────────────────────────────────────────

    def draw_config_panel(self, surface, panel_state):
        if not panel_state.get('visible'): return
        from config import Config
        params = panel_state['params']
        cursor = panel_state['cursor']
        
        max_visible = max(5, (self.H - 80) // 18)
        
        if len(params) > max_visible:
            half = max_visible // 2
            start_idx = max(0, min(cursor - half, len(params) - max_visible))
            end_idx = start_idx + max_visible
        else:
            start_idx = 0
            end_idx = len(params)
            
        visible_params = params[start_idx:end_idx]

        PW, PH = 420, min(len(visible_params)*18+50, self.H-20)
        px0, py0 = self.W-PW-8, 8

        overlay = pygame.Surface((PW,PH), pygame.SRCALPHA)
        overlay.fill((10,10,30,210))
        surface.blit(overlay,(px0,py0))
        pygame.draw.rect(surface,(80,120,255),(px0,py0,PW,PH),1)

        title = self.font_med.render(f"CONFIG ({start_idx+1}-{end_idx}/{len(params)})  (Tab=close  ↑↓=select  ←→=change)",
                                     True,(180,220,255))
        surface.blit(title,(px0+6,py0+4))

        for visual_i, p in enumerate(visible_params):
            i = start_idx + visual_i
            y    = py0+22+visual_i*18
            val  = getattr(Config, p['attr'], '?')
            unit = p.get('unit','')

            if isinstance(val, bool):
                val_str = "ON " if val else "OFF"
                col_v   = (100,255,140) if val else (255,100,100)
            elif isinstance(val, float):
                val_str = f"{val:.4f}" if val < 0.01 and val > 0 else f"{val:.3f}"
                col_v   = (255,240,160)
            elif isinstance(val, int):
                val_str = str(val); col_v = (255,240,160)
            else:
                val_str = str(val); col_v = (200,200,255)

            if unit: val_str = f"{val_str} {unit}"

            if i == cursor:
                hl = pygame.Surface((PW-2,17), pygame.SRCALPHA)
                hl.fill((40,60,120,180))
                surface.blit(hl,(px0+1,y-1))
            arrow = "▶ " if i==cursor else "  "
            label = self.font.render(f"{arrow}{p['name']:<28}", True,
                                     (255,255,255) if i==cursor else (180,180,200))
            value = self.font.render(val_str, True, col_v)
            surface.blit(label,(px0+4,y))
            surface.blit(value,(px0+PW-90,y))

        hint = self.font.render("H=BH  B=planet  V=body  O=pause  ESC=quit",
                                True,(120,140,180))
        surface.blit(hint,(px0+4,py0+PH-14))

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _hud(self, surface, fps, bh_list, show_particles, selected_bh_idx):
        from config import Config
        active = [b for b in bh_list if b.active]
        bh_str = "  ".join(
            f"{'[►]' if i==selected_bh_idx else '   '}{b.label} M={b.mass:.2f} a={b.spin:.2f}"
            for i,b in enumerate(bh_list) if b.active)
        lines = [
            (f"FPS {fps:3.0f} | BHs:{len(active)} | "
             f"Time×{Config.TIME_LAPSE:.1f} | "
             f"{'3D★' if Config.REALISTIC_STARS else '2D★'} | "
             f"Phys:{Config.PHYSICS_MODE} | "
             f"Disk:{'ON' if Config.USE_VIRTUAL_ACCRETION_DISK else 'OFF'} | "
             f"Part:{'ON' if show_particles else 'OFF'} | Tab=cfg"),
            bh_str,
        ]
        
        if getattr(Config, 'SHOW_UNIT', False):
            # Convert geometric units to physical units roughly
            M_sun = 1.0  # reference mass
            c_km_s = 299792.458
            M_str = f"M = {M_sun * Config.M_BH:.2f} M_sun"
            V_str = f"c = {c_km_s} km/s"
            L_str = f"Rs = {3.0 * Config.M_BH:.1f} km"
            lines.insert(1, f"Physical Units: [ {M_str} | {V_str} | {L_str} ]")

        legend = self.font.render(
            "Units: Mass [M_geo]  Dist [R_s]  Vel [c]  Time [s_sim]", True,(150,160,180))
        surface.blit(legend,(10,10))
        for i, line in enumerate(lines):
            s = self.font.render(line, True,(200,220,255))
            surface.blit(s,(10, self.H-14*(len(lines)-i)))

    # ── main render ───────────────────────────────────────────────────────────

    def render(self, surface, camera, disk_list, free, objects,
               bh_list, fps, show_particles=True,
               selected_bh_idx=0, panel_state=None):

        buf = np.zeros((self.W, self.H, 3), dtype=np.int16)
        from config import Config

        bh_projs = []
        for bh in bh_list:
            if not bh.active: continue
            res = camera.project_single(bh.pos, self.W, self.H)
            if res is None: continue
            bhx, bhy, bh_depth = res
            
            bh_dist = np.linalg.norm(bh.pos - camera.position)

            dist_fc    = math.hypot(bhx-self.W/2, bhy-self.H/2)
            max_r      = max(self.W,self.H)*0.7
            dist_fade  = np.clip(1.0-(dist_fc-max_r)/(max_r+1.0), 0.0, 1.0)
            depth_fade = np.clip((bh_depth-0.1)/5.0, 0.0, 1.0)
            fade       = dist_fade * depth_fade

            shadow_r_px   = camera.get_screen_radius(
                bh.shadow_r_sim*bh.SIM_SCALE, bh_depth, self.W, self.H) * fade
            einstein_r_px = camera.get_screen_radius(
                bh.einstein_r_sim*bh.SIM_SCALE, bh_depth, self.W, self.H) * fade
            bh_projs.append((bhx, bhy, einstein_r_px, bh_depth, shadow_r_px, bh, bh_dist))

        bh_projs.sort(key=lambda x: x[6])

        # 1 stars
        self._draw_stars(buf, camera, bh_projs)

        if show_particles:
            # 2 far particles
            self._draw_particles(buf, camera, disk_list, free, 'far', bh_projs)
            # 3 photon rings
            if Config.USE_VIRTUAL_ACCRETION_DISK:
                for bhx, bhy, _te, bh_depth, shadow_r_px, bh, bh_dist in bh_projs:
                    if shadow_r_px <= 0: continue
                    for disk in disk_list:
                        if disk.bh is bh and disk.n > 0:
                            self._draw_photon_ring(buf, camera, disk,
                                                   bhx, bhy, shadow_r_px, bh_dist)

        # 4 flush - nothing black painted
        np.clip(buf, 0, 255, out=buf)
        px = pygame.surfarray.pixels3d(surface); px[:] = buf.astype(np.uint8); del px

        # 4b far bodies
        for obj in objects:
            self._draw_celestial_body(surface, camera, obj, bh_projs, 'far')

        # 5 BH shadows + merger flashes (draw shadow to mask out far objects overlapping the horizon)
        for bhx, bhy, theta_E, bh_depth, shadow_r_px, bh, bh_dist in bh_projs:
            if shadow_r_px > 0:
                pygame.draw.circle(surface, (0,0,0),
                                   (int(bhx),int(bhy)), int(shadow_r_px)+1)
            if bh.merge_flash > 0:
                self._draw_merger_flash(surface, bh, camera)

        # 6 near particles
        if show_particles:
            px2  = pygame.surfarray.pixels3d(surface)
            buf2 = px2.astype(np.int16); del px2
            self._draw_particles(buf2, camera, disk_list, free, 'near', bh_projs)
            np.clip(buf2, 0, 255, out=buf2)
            px3  = pygame.surfarray.pixels3d(surface)
            px3[:] = buf2.astype(np.uint8); del px3

        # 6b near bodies
        for obj in objects:
            self._draw_celestial_body(surface, camera, obj, bh_projs, 'near')

        # 7 bloom
        self._bloom(surface)

        # 8 HUD
        self._hud(surface, fps, bh_list, show_particles, selected_bh_idx)
        if panel_state:
            self.draw_config_panel(surface, panel_state)