#v7
import sys, math, random
import numpy as np
import pygame

from physics    import Physics
from camera     import Camera3D
from simulation import AccretionDisk, FreeParticles, CelestialBody, GasPlanet
from renderer   import Renderer
from blackhole  import BlackHoleBody
from config     import Config, CONFIG_PANEL_PARAMS, loader

settings = loader.settings
W = int(settings.get("window_width", 1280))
H = int(settings.get("window_height", 720))
TARGET_FPS = int(settings.get("target_fps", 60))



def build_initial_bh_list() -> list:
    bh_list = []
    if Config.RANDOM_BH_COUNT is not None:
        n   = int(Config.RANDOM_BH_COUNT)
        rng = random.Random(42)
        spread = 250.0
        for i in range(n):
            mass  = rng.uniform(0.2, 0.8)
            spin  = rng.uniform(0.0, 0.6)
            px    = rng.uniform(-spread, spread)
            pz    = rng.uniform(-spread, spread)
            py    = rng.uniform(-20, 20)
            r     = math.sqrt(px*px+pz*pz) / Config.SIM_SCALE + 0.01
            v_c   = math.sqrt(Config.G * 0.5 / r) * Config.SIM_SCALE * 0.6
            angle = math.atan2(pz, px) + math.pi/2
            vx    = v_c * math.cos(angle) * rng.uniform(0.6, 1.2)
            vz    = v_c * math.sin(angle) * rng.uniform(0.6, 1.2)
            bh_list.append(BlackHoleBody(pos=(px,py,pz), vel=(vx,0,vz),
                                         mass=mass, spin=spin,
                                         sim_scale=Config.SIM_SCALE,
                                         label=f"BH-{i+1}"))
        return bh_list

    for i, cfg in enumerate(Config.INITIAL_BLACK_HOLES):
        bh_list.append(BlackHoleBody(
            pos=cfg.get('pos',(0,0,0)), vel=cfg.get('vel',(0,0,0)),
            mass=cfg.get('mass',0.5),   spin=cfg.get('spin',0.0),
            sim_scale=Config.SIM_SCALE, label=f"BH-{i+1}"))
    return bh_list


def spawn_bh_at_camera(camera, bh_list, mass=0.3, spin=None):
    forward, right, _ = camera._basis()
    pos = camera.position + forward*120.0 + right*random.uniform(-25,25)
    vel = right*random.uniform(-10,10) + forward*random.uniform(-3,3)
    if spin is None: spin = random.uniform(0.0, 0.5)
    bh = BlackHoleBody(pos=tuple(pos), vel=tuple(vel),
                       mass=mass, spin=spin,
                       sim_scale=Config.SIM_SCALE,
                       label=f"BH-{BlackHoleBody._id_counter+1}")
    bh_list.append(bh)
    return bh


def _panel_adjust(param, delta):
    attr  = param['attr']
    ptype = param['type']
    cur   = getattr(Config, attr)
    if ptype == 'bool':
        setattr(Config, attr, not cur)
    elif ptype in ('float', 'int'):
        step = param.get('step', 1)
        lo   = param.get('min', -1e9)
        hi   = param.get('max',  1e9)
        new  = cur + delta * step
        if ptype == 'int':
            new = int(round(new))
        setattr(Config, attr,
                float(np.clip(new, lo, hi)) if ptype=='float'
                else int(np.clip(new, lo, hi)))
    elif ptype == 'choice':
        choices = param['choices']
        idx = choices.index(cur) if cur in choices else 0
        setattr(Config, attr, choices[(idx + int(delta)) % len(choices)])


def _resync_disk(bh_idx, disk_list, bh_list):
    if bh_idx >= len(bh_list): return
    bh = bh_list[bh_idx]
    for disk in disk_list:
        if disk.bh is bh:
            disk.resync()


def _on_config_change(param, disk_list, free, rend=None):
    attr = param['attr']
    if attr == 'USE_VIRTUAL_ACCRETION_DISK' and not Config.USE_VIRTUAL_ACCRETION_DISK:
        for disk in disk_list:
            if disk.n > 0: free.absorb_disk(disk)
    if attr == 'DISK_PARTICLE_COUNT':
        for disk in disk_list: disk.rebuild()
    if attr == 'STAR_COUNT' and rend:
        rend.rebuild_stars(Config.STAR_COUNT)


# Main 

def main():
    pygame.init()
    flags = pygame.FULLSCREEN if settings.get("fullscreen", False) else 0
    try:
        screen = pygame.display.set_mode((W, H), flags, vsync=1 if settings.get("vsync", False) else 0)
    except TypeError:
        screen = pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption("Black Hole Simulator 3D  —  v6")
    clock  = pygame.time.Clock()

    phys   = Physics()
    camera = Camera3D(pos=(150,240,680), pitch=-18, yaw=12)
    rend   = Renderer(W, H, phys)

    bh_list  = build_initial_bh_list()
    phys.update_properties(bh_list[0].mass, bh_list[0].spin)

    disk_list = [AccretionDisk(bh, n=2500, physics=Physics()) for bh in bh_list]
    free      = FreeParticles(phys, n=360)
    selected  = 0

    # Auto-transfer particles if the current loaded config wants the disk disabled
    startup_frames_elapsed = 0
    startup_auto_transfer_needed = not Config.USE_VIRTUAL_ACCRETION_DISK

    celestial_objects = [
        GasPlanet(phys, pos=(40,0,700), radius=1.5,
                  color=(100,200,255), vel=(0.035,0,0.015), respawn=False),
        CelestialBody(phys, pos=(20,0,-500), radius=1.0,
                      color=(255,150,50), vel=(0.04,0,-0.01), respawn=False),
    ]

    panel = {'visible': False, 'cursor': 0, 'params': CONFIG_PANEL_PARAMS}
    dragging       = False
    last_mouse     = (0,0)
    paused         = False
    show_particles = True

    def active_bhs():
        return [b for b in bh_list if b.active]

    while True:
        dt_real = min(clock.tick(TARGET_FPS)/1000.0, 0.05)

        time_lapse_bh = 1.0
        if Config.ENABLE_TIME_DILATION_CAMERA:
            nearest_r = nearest_rs = 1e18
            for bh in active_bhs():
                d = np.linalg.norm(camera.position - bh.pos) / phys.SIM_SCALE
                if d < nearest_r:
                    nearest_r  = d
                    nearest_rs = bh.rs_sim
            if nearest_r < 1e17:
                r_ratio = nearest_rs / max(nearest_r, nearest_rs*1.01)
                time_lapse_bh = 1.0 / math.sqrt(max(0.0001, 1.0-r_ratio))

        dt = dt_real * Config.TIME_LAPSE * time_lapse_bh

        # Auto-transfer on startup if needed
        if startup_auto_transfer_needed:
            startup_frames_elapsed += 1
            if startup_frames_elapsed > 3:
                for disk in disk_list:
                    if disk.n > 0: free.absorb_disk(disk)
                Config.USE_VIRTUAL_ACCRETION_DISK = False
                startup_auto_transfer_needed = False

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.KEYDOWN:
                k = ev.key

                if k in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit()

                #Config panel
                if k == pygame.K_TAB:
                    panel['visible'] = not panel['visible']

                if panel['visible']:
                    n_params = len(panel['params'])
                    if k == pygame.K_UP:
                        panel['cursor'] = (panel['cursor']-1) % n_params
                    elif k == pygame.K_DOWN:
                        panel['cursor'] = (panel['cursor']+1) % n_params
                    elif k == pygame.K_LEFT:
                        _panel_adjust(panel['params'][panel['cursor']], -1)
                        _on_config_change(panel['params'][panel['cursor']], disk_list, free, rend)
                    elif k == pygame.K_RIGHT:
                        _panel_adjust(panel['params'][panel['cursor']], +1)
                        _on_config_change(panel['params'][panel['cursor']], disk_list, free, rend)
                    if k not in (pygame.K_TAB, pygame.K_o, pygame.K_ESCAPE, pygame.K_q,
                                  pygame.K_h, pygame.K_b, pygame.K_v,
                                  pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        continue

                #BH selection
                if k == pygame.K_RIGHTBRACKET:
                    if bh_list: selected = (selected+1) % len(bh_list)
                if k == pygame.K_LEFTBRACKET:
                    if bh_list: selected = (selected-1) % len(bh_list)

                #BH mass / spin
                if k == pygame.K_m and bh_list:
                    bh_list[selected].mass = max(0.0, bh_list[selected].mass+0.05)
                    bh_list[selected]._recompute()
                    _resync_disk(selected, disk_list, bh_list)
                if k == pygame.K_n and bh_list:
                    target_bh = bh_list[selected]
                    target_bh.mass -= 0.05
                    if target_bh.mass <= 0.01:
                        target_bh.mass = 0.0
                        target_bh.active = False
                    else:
                        target_bh._recompute()
                        _resync_disk(selected, disk_list, bh_list)
                if k == pygame.K_k and bh_list:
                    bh_list[selected].spin = min(0.99, bh_list[selected].spin+0.05)
                    bh_list[selected]._recompute()
                    _resync_disk(selected, disk_list, bh_list)
                if k == pygame.K_j and bh_list:
                    bh_list[selected].spin = max(0.0, bh_list[selected].spin-0.05)
                    bh_list[selected]._recompute()
                    _resync_disk(selected, disk_list, bh_list)

                #Quick toggles
                if k == pygame.K_o:  paused = not paused
                if k == pygame.K_c:  show_particles = not show_particles
                if k == pygame.K_z:  Config.TIME_LAPSE = max(0.0, Config.TIME_LAPSE+0.5)
                if k == pygame.K_x:  Config.TIME_LAPSE = max(0.0, Config.TIME_LAPSE-0.5)
                if k == pygame.K_p:
                    Config.PHYSICS_MODE = "2-body" if Config.PHYSICS_MODE=="realistic" else "realistic"
                if k == pygame.K_t:
                    Config.ENABLE_TIME_DILATION_CAMERA = not Config.ENABLE_TIME_DILATION_CAMERA
                if k == pygame.K_e:
                    Config.ENERGY_LOST = not Config.ENERGY_LOST
                if k == pygame.K_r:
                    Config.ENABLE_REDSHIFT_FADING = not Config.ENABLE_REDSHIFT_FADING
                if k == pygame.K_i:
                    Config.PARTICLE_TEMP_GLOW = not Config.PARTICLE_TEMP_GLOW
                if k == pygame.K_u:
                    Config.PLANET_SPAGHETTIFICATION = not Config.PLANET_SPAGHETTIFICATION
                if k == pygame.K_g:
                    Config.BH_GRAVITY_ON = not Config.BH_GRAVITY_ON
                if k == pygame.K_f:
                    Config.BH_MERGERS_ON = not Config.BH_MERGERS_ON
                if k == pygame.K_s:       #★ NEW: toggle realistic 3D stars
                    Config.REALISTIC_STARS = not Config.REALISTIC_STARS
                if k == pygame.K_y:
                    Config.USE_VIRTUAL_ACCRETION_DISK = not Config.USE_VIRTUAL_ACCRETION_DISK
                    #if not Config.USE_VIRTUAL_ACCRETION_DISK:
                    for disk in disk_list:
                        if disk.n > 0: free.absorb_disk(disk)

                #Spawn
                if k == pygame.K_h:
                    new_bh   = spawn_bh_at_camera(camera, bh_list)
                    new_disk = AccretionDisk(new_bh, n=2000, physics=Physics())
                    disk_list.append(new_disk)
                    selected = len(bh_list)-1

                if k in (pygame.K_b, pygame.K_v):
                    forward, right, _ = camera._basis()
                    pos = camera.position + forward*80.0
                    vel = forward*0.15
                    if k == pygame.K_b:
                        celestial_objects.append(
                            GasPlanet(phys, pos=pos, radius=1.5,
                                      color=(random.randint(100,255),200,255),
                                      vel=vel, respawn=False))
                    else:
                        celestial_objects.append(
                            CelestialBody(phys, pos=pos, radius=0.8,
                                          color=(255,100,random.randint(50,150)),
                                          vel=vel, respawn=False))

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                dragging = True;  last_mouse = ev.pos
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            if ev.type == pygame.MOUSEMOTION and dragging:
                dx = ev.pos[0]-last_mouse[0];  dy = ev.pos[1]-last_mouse[1]
                sensitivity = float(settings.get("mouse_sensitivity", 0.28))
                camera.rotate(-dx * sensitivity, -dy * sensitivity)
                last_mouse = ev.pos

        if not panel['visible']:
            keys  = pygame.key.get_pressed()
            speed = float(settings.get("camera_speed", 250.0)) * dt_real
            if keys[pygame.K_LSHIFT]: speed *= 3.0
            fwd = right = up = 0
            if keys[pygame.K_w]: fwd   += speed
            if keys[pygame.K_s]: fwd   -= speed
            if keys[pygame.K_d]: right += speed
            if keys[pygame.K_a]: right -= speed
            if keys[pygame.K_SPACE]:  up += speed
            if keys[pygame.K_LCTRL]: up -= speed
            if fwd or right or up: camera.move(right, up, fwd)
            if keys[pygame.K_LEFT]:  camera.rotate( 80*dt_real, 0)
            if keys[pygame.K_RIGHT]: camera.rotate(-80*dt_real, 0)
            if keys[pygame.K_UP]:    camera.rotate(0,  80*dt_real)
            if keys[pygame.K_DOWN]:  camera.rotate(0, -80*dt_real)
        else:
            keys  = pygame.key.get_pressed()
            speed = float(settings.get("camera_speed", 250.0)) * dt_real
            if keys[pygame.K_LSHIFT]: speed *= 3.0
            fwd = right = up = 0
            if keys[pygame.K_w]: fwd   += speed
            if keys[pygame.K_s]: fwd   -= speed
            if keys[pygame.K_d]: right += speed
            if keys[pygame.K_a]: right -= speed
            if keys[pygame.K_SPACE]:  up += speed
            if keys[pygame.K_LCTRL]: up -= speed
            if fwd or right or up: camera.move(right, up, fwd)

            # allow left/right hold to adjust panel values continuously for non-bool params
            current_param = panel['params'][panel['cursor']]
            if current_param['type'] in ('float', 'int', 'choice'):
                if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                    delta = -1 if keys[pygame.K_LEFT] else 1
                    _panel_adjust(current_param, delta)
                    _on_config_change(current_param, disk_list, free, rend)

        if not paused:
            #Hawking evaporation (dM/dt ∝ 1/M²)
            if Config.ENERGY_LOST:
                for bh in active_bhs():
                    evap = Config.HAWKING_EVAPORATION_RATE / max(bh.mass**2, 0.01)
                    bh.mass -= evap * dt
                    if bh.mass <= 0.01:
                        bh.mass = 0.0; bh.active = False
                    else:
                        bh._recompute()

            #N-body
            if Config.BH_GRAVITY_ON:
                for bh in active_bhs():
                    bh.update(dt, bh_list)

            #Mergers
            if Config.BH_MERGERS_ON:
                for bh in list(active_bhs()):
                    dead = bh.try_merge(bh_list)
                    for d in dead:
                        disk_list[:] = [dk for dk in disk_list if dk.bh is not d]

            #Particles & disks
            if show_particles:
                for disk in list(disk_list):
                    if not disk.bh.active:
                        if disk.n > 0: free.absorb_disk(disk)
                        disk_list.remove(disk)
                    elif disk.n > 0:
                        disk.update(dt)
                free.update(dt, bh_list=active_bhs())

            #Celestial objects
            celestial_objects[:] = [o for o in celestial_objects if o.active]
            for obj in celestial_objects:
                obj.update(dt, speed=1.5, free_system=free, bh_list=active_bhs())

        active_disks = [d for d in disk_list if d.bh.active and d.n > 0]
        rend.render(screen, camera, active_disks, free,
                    celestial_objects, bh_list, clock.get_fps(),
                    show_particles, selected, panel)

        if paused:
            msg = rend.font.render("  PAUSED - O to resume  ", True, (255,220,80))
            screen.blit(msg, msg.get_rect(center=(W//2, H-55)))

        pygame.display.flip()


if __name__ == "__main__":
    main()
