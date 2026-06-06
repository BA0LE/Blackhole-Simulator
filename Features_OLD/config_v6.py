#v6
class Config:
    G_SI = 6.67430e-11
    C_SI = 299792458
    G    = 1.0
    C    = 1.0

    M_BH   = 0.5
    SPIN_A = 0.0

    # ── Multi-BH scenarios ────────────────────────────────────────────────────
    # Each entry: dict(pos, vel, mass, spin) — pos/vel in pixel coords
    INITIAL_BLACK_HOLES = [
        dict(pos=(0,0,0), vel=(0,0,0), mass=0.5, spin=0.0),
    ]
    RANDOM_BH_COUNT = None   # set to int for random cluster

    # ── Scale ─────────────────────────────────────────────────────────────────
    SIM_SCALE = 55.0

    # ── Multi-BH dynamics ─────────────────────────────────────────────────────
    BH_GRAVITY_ON  = True
    BH_MERGERS_ON  = True

    # ── Brightness & Counts ──────────────────────────────────────────────────
    DISK_BRIGHTNESS             = 1.5
    GLOBAL_BRIGHTNESS           = 1.0
    DISK_PARTICLE_COUNT         = 3200
    STAR_COUNT                  = 3400

    # ── Visual / mechanics toggles ────────────────────────────────────────────
    USE_VIRTUAL_ACCRETION_DISK  = True
    PLANET_SPAGHETTIFICATION    = True
    PARTICLE_TEMP_GLOW          = True
    PHYSICS_MODE                = "realistic"   # "realistic" or "2-body"
    TIME_LAPSE                  = 1.0
    ENABLE_TIME_DILATION_CAMERA   = True
    ENABLE_REDSHIFT_FADING        = True
    ENABLE_TIME_DILATION_OBJECTS  = True

    PLANET_COHESION         = 1.0
    GAS_EMISSION_RATE       = 2.0
    ROCHE_LIMIT_BASE        = 15.0
    PARTICLE_PLUNGE_FACTOR  = 1.0
    OBJECT_GRAVITY          = True
    ENERGY_LOST             = True

    ORBIT_ENERGY_DECAY      = 0.010
    MULTI_BODY_NOISE        = 0.002
    PARTICLE_DYNAMICS_NOISE = 0.04
    HAWKING_EVAPORATION_RATE = 0.0001
    GR_CURVATURE_EFFECT     = True
    BH_SPIN_LENSE_THIRRING  = True


# ── Config panel parameter definitions ───────────────────────────────────────
# Each dict: name, attr, type ('bool'/'float'/'int'/'choice'),
#            min, max, step  (only for numeric), choices (only for 'choice')

CONFIG_PANEL_PARAMS = [
    # ─ Time ──────────────────────────────────────────────────────────────────
    dict(name="Time Lapse",             attr="TIME_LAPSE",
         type="float", min=0.0, max=30.0, step=0.5),
    dict(name="Time Dilation (camera)", attr="ENABLE_TIME_DILATION_CAMERA",  type="bool"),
    dict(name="Time Dilation (objects)",attr="ENABLE_TIME_DILATION_OBJECTS", type="bool"),
    # ─ Black Holes ───────────────────────────────────────────────────────────
    dict(name="BH Gravity",             attr="BH_GRAVITY_ON",   type="bool"),
    dict(name="BH Mergers",             attr="BH_MERGERS_ON",   type="bool"),
    dict(name="Hawking Evaporation",    attr="ENERGY_LOST",      type="bool"),
    # ─ Physics mode ──────────────────────────────────────────────────────────
    dict(name="Physics Mode",           attr="PHYSICS_MODE",
         type="choice", choices=["realistic","2-body"]),
    dict(name="GR Curvature",           attr="GR_CURVATURE_EFFECT",      type="bool"),
    dict(name="Lense-Thirring Spin",    attr="BH_SPIN_LENSE_THIRRING",   type="bool"),
    dict(name="Object Gravity",         attr="OBJECT_GRAVITY",           type="bool"),
    dict(name="Hawking Evap Rate",      attr="HAWKING_EVAPORATION_RATE",
         type="float", min=0.0, max=0.01, step=0.0001),
    dict(name="Orbit Energy Decay",     attr="ORBIT_ENERGY_DECAY",
         type="float", min=0.0, max=0.1, step=0.002),
    dict(name="Multi-Body Noise",       attr="MULTI_BODY_NOISE",
         type="float", min=0.0, max=0.02, step=0.001),
    dict(name="Particle Chaos",         attr="PARTICLE_DYNAMICS_NOISE",
         type="float", min=0.0, max=0.2, step=0.01),
    # ─ Accretion / visual ────────────────────────────────────────────────────
    dict(name="Global Brightness",      attr="GLOBAL_BRIGHTNESS",
         type="float", min=0.1, max=4.0, step=0.1),
    dict(name="Disk Brightness",        attr="DISK_BRIGHTNESS",
         type="float", min=0.1, max=5.0, step=0.1),
    dict(name="Star Count",             attr="STAR_COUNT",
         type="int", min=0, max=15000, step=500),
    dict(name="Disk Particle Count",    attr="DISK_PARTICLE_COUNT",
         type="int", min=0, max=15000, step=500),
    dict(name="Virtual Accretion Disk", attr="USE_VIRTUAL_ACCRETION_DISK", type="bool"),
    dict(name="Redshift Fading",        attr="ENABLE_REDSHIFT_FADING",    type="bool"),
    dict(name="Particle Temp Glow",     attr="PARTICLE_TEMP_GLOW",        type="bool"),
    # ─ Spaghettification ─────────────────────────────────────────────────────
    dict(name="Spaghettification",      attr="PLANET_SPAGHETTIFICATION",  type="bool"),
    dict(name="Roche Limit Base",       attr="ROCHE_LIMIT_BASE",
         type="float", min=1.0, max=60.0, step=1.0),
    dict(name="Planet Cohesion",        attr="PLANET_COHESION",
         type="float", min=0.05, max=5.0, step=0.05),
    dict(name="Gas Emission Rate",      attr="GAS_EMISSION_RATE",
         type="float", min=0.1, max=10.0, step=0.1),
    dict(name="Plunge Factor",          attr="PARTICLE_PLUNGE_FACTOR",
         type="float", min=0.0, max=2.0, step=0.1),
]
