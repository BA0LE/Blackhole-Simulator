

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from copy   import deepcopy
from dataclasses import dataclass, field
from typing import Any

def _get_bundle_dir() -> str:
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            #onefile: assets extracted to temp _MEI* dir
            return sys._MEIPASS
        else:
            #onedir: assets sit alongside the exe
            return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _get_data_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BUNDLE_DIR : str = _get_bundle_dir()
DATA_DIR   : str = _get_data_dir()

@dataclass
class FileSpec:
    path:     str               #relative to DATA_DIR
    defaults: dict = field(default_factory=dict)
    readonly: bool = False    #skip writes if True
    autosave: bool = True             #save after every set_() call


FILE_REGISTRY: dict[str, FileSpec] = {

    #Simulation configuration
    "config": FileSpec(
        path     = "config.json",
        autosave = True,
        defaults = {
            #Physical constants (never change at runtime)
            "G_SI":  6.67430e-11,
            "C_SI":  299_792_458,
            "G":     1.0,
            "C":     1.0,

            #Reference BH
            "M_BH":  0.5,
            "SPIN_A": 0.0,

            #Initial scene
            "INITIAL_BLACK_HOLES": [
                {"pos": [0, 0, 0], "vel": [0, 0, 0], "mass": 0.5, "spin": 0.0}
            ],
            "RANDOM_BH_COUNT": None,

            #Scale
            "SIM_SCALE": 55.0,

            #N-body
            "BH_GRAVITY_ON":  True,
            "BH_MERGERS_ON":  True,

            #Brightness / counts
            "DISK_BRIGHTNESS":        1.5,
            "GLOBAL_BRIGHTNESS":      1.0,
            "DISK_PARTICLE_COUNT":    25000,
            "STAR_COUNT":             3400,

            #Visual toggles
            "USE_VIRTUAL_ACCRETION_DISK": False,
            "PLANET_SPAGHETTIFICATION":   True,
            "PARTICLE_TEMP_GLOW":         True,
            "REALISTIC_STARS":            True,
            "PHYSICS_MODE":               "realistic",
            "TIME_LAPSE":                 1.0,
            "ENABLE_TIME_DILATION_CAMERA":  True,
            "ENABLE_REDSHIFT_FADING":       True,
            "ENABLE_TIME_DILATION_OBJECTS": True,

            #bject mechanics
            "PLANET_COHESION":        1.0,
            "GAS_EMISSION_RATE":      2.0,
            "ROCHE_LIMIT_BASE":       15.0,
            "PARTICLE_PLUNGE_FACTOR": 1.0,
            "OBJECT_GRAVITY":         True,
            "ENERGY_LOST":            True,

            #Physics constants (tunable)
            "ORBIT_ENERGY_DECAY":       0.010,
            "MULTI_BODY_NOISE":         0.002,
            "PARTICLE_DYNAMICS_NOISE":  0.04,
            "HAWKING_EVAPORATION_RATE": 0.0001,
            "GR_CURVATURE_EFFECT":      True,
            "BH_SPIN_LENSE_THIRRING":   True,

            #Accretion
            "ACCRETION_DM_PARTICLE": 0.00005,
            "ACCRETION_DM_BODY":     0.005,

            #Particle self-gravity
            "PARTICLE_FORCE":          False,
            "PARTICLE_FORCE_STRENGTH": 0.0008,

            #Particle auto-zoom
            "PARTICLE_AUTO_ZOOM": True,
            "PARTICLE_BASE_SIZE": 1.0,

            #v7+ features
            "REAL_LIGHT_BEND":        True,
            "RENDER_DISTANCE":        0.0,
            "RANDOM_PARTICLE_SIZE":   True,
            "MIN_SIZE_RANDOM":        0.5,
            "MAX_SIZE_RANDOM":        2.5,
            "BRIGHTNESS_BY_VELOCITY": True,
            "PARTICLE_ABSORBER":      True,
            "SHOW_UNIT":              True,
        },
    ),

    #User preferences 
    "settings": FileSpec(
        path     = "settings.json",
        autosave = True,
        defaults = {
            "window_width":       1280,
            "window_height":      720,
            "target_fps":         60,
            "fullscreen":         False,
            "vsync":              False,
            "mouse_sensitivity":  0.28,
            "camera_speed":       250.0,
            "keybinds":           {},       #pygame key-name overrides
            "config_panel_width": 420,
            "hud_scale":          1.0,
        },
    ),

    #Achievements 
    "achievements": FileSpec(
        path     = "achievements.json",
        autosave = True,
        defaults = {
            #key → {"unlocked": bool, "progress": number, "desc": str}
            "first_merger":     {"unlocked": False, "progress": 0,
                                 "desc": "Witness your first black hole merger"},
            "singularity":      {"unlocked": False, "progress": 0,
                                 "desc": "Grow a BH to mass > 5.0"},
            "tidal_disruption": {"unlocked": False, "progress": 0,
                                 "desc": "Spaghettify a gas planet"},
            "hawking_ghost":    {"unlocked": False, "progress": 0,
                                 "desc": "Let a BH fully evaporate"},
            "donut_world":      {"unlocked": False, "progress": 0,
                                 "desc": "Watch a planet form a torus ring"},
        },
    ),

    #Add more files here 
    #"leaderboard": FileSpec(
    #path     = "leaderboard.json",
    #defaults = {"entries": [], "max_keep": 100},
    #),
}



#GameLoader
#

class GameLoader:

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir   = os.path.abspath(data_dir)
        self.bundle_dir = BUNDLE_DIR
        self._lock      = threading.Lock()
        self._data:  dict[str, dict] = {}
        self._dirty: dict[str, bool] = {}

    #Lifecycle 

    def load_all(self) -> "GameLoader":
    
        for name in FILE_REGISTRY:
            self.reload(name)
        return self

    def reload(self, name: str) -> dict:
        spec = self._spec(name)
        path = self._abs(spec.path)
        data: dict = {}

        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[Loader] WARNING: could not read {path!r}: {exc}")

        #Merge: defaults first so new keys are always present
        merged = deepcopy(spec.defaults)
        merged.update(data)

        with self._lock:
            self._data[name]  = merged
            self._dirty[name] = False

        return merged

    #Read 

    def get(self, name: str) -> dict:
        if name not in self._data:
            raise KeyError(f"[Loader] Unknown file '{name}'. "
                           f"Add it to FILE_REGISTRY first.")
        return self._data[name]

    def get_value(self, name: str, key: str, default: Any = None) -> Any:
        return self._data.get(name, {}).get(key, default)

    #Write 

    def set(self, name: str, key: str, value: Any) -> None:
     
        spec = self._spec(name)
        if spec.readonly:
            raise PermissionError(f"[Loader] '{name}' is read-only.")
        with self._lock:
            self._data[name][key] = value
            self._dirty[name]     = True
        if spec.autosave:
            self.save(name)

    def update(self, name: str, patch: dict) -> None:
        spec = self._spec(name)
        if spec.readonly:
            raise PermissionError(f"[Loader] '{name}' is read-only.")
        with self._lock:
            self._data[name].update(patch)
            self._dirty[name] = True
        if spec.autosave:
            self.save(name)

    def save(self, name: str) -> None:
    
        spec = self._spec(name)
        if spec.readonly:
            return
        path = self._abs(spec.path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with self._lock:
            snapshot       = deepcopy(self._data[name])
            self._dirty[name] = False
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=2, ensure_ascii=False)
            shutil.move(tmp, path)
        except OSError as exc:
            print(f"[Loader] ERROR saving {path!r}: {exc}")

    def save_all(self) -> None:
        for name, spec in FILE_REGISTRY.items():
            if not spec.readonly:
                self.save(name)

    def save_all_dirty(self) -> None:
        for name in FILE_REGISTRY:
            if self._dirty.get(name):
                self.save(name)

    #Convenience shortcuts 

    @property
    def config(self) -> dict:
        return self.get("config")

    @property
    def settings(self) -> dict:
        return self.get("settings")

    @property
    def achievements(self) -> dict:
        return self.get("achievements")

    def set_config(self, key: str, value: Any) -> None:
        self.set("config", key, value)

    def set_setting(self, key: str, value: Any) -> None:
        self.set("settings", key, value)

    def set_achievement(self, key: str, value: Any) -> None:
        self.set("achievements", key, value)

    #Achievement helpers 

    def unlock_achievement(self, key: str) -> bool:
        ach = deepcopy(self.achievements.get(key))
        if ach is None:
            print(f"[Loader] Achievement '{key}' not in registry.")
            return False
        if ach.get("unlocked"):
            return False
        ach["unlocked"] = True
        self.set_achievement(key, ach)
        return True

    def increment_achievement(self, key: str, by: float = 1.0) -> float:
        ach = deepcopy(self.achievements.get(key, {}))
        ach["progress"] = ach.get("progress", 0) + by
        self.set_achievement(key, ach)
        return ach["progress"]

    #Config preset helpers 

    def export_config(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(deepcopy(self._data["config"]), fh, indent=2,
                      ensure_ascii=False)
        print(f"[Loader] Config exported -> {path}")

    def import_config(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[Loader] Preset not found: {path!r}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        merged = deepcopy(FILE_REGISTRY["config"].defaults)
        merged.update(data)
        with self._lock:
            self._data["config"]  = merged
            self._dirty["config"] = True
        self.save("config")
        print(f"[Loader] Config imported <- {path}")

    def reset_to_defaults(self, name: str) -> None:
 
        spec = self._spec(name)
        with self._lock:
            self._data[name]  = deepcopy(spec.defaults)
            self._dirty[name] = True
        if not spec.readonly:
            self.save(name)

    #Path helpers 

    def _spec(self, name: str) -> FileSpec:
        if name not in FILE_REGISTRY:
            raise KeyError(f"[Loader] Unknown file '{name}'.")
        return FILE_REGISTRY[name]

    def _abs(self, rel: str) -> str:
        return os.path.join(self.data_dir, rel)

    def asset(self, rel: str) -> str:
        """
        Resolve a path to a READ -ONLY bundled asset (images, fonts, etc.).
        Works both from source and inside a PyInstaller bundle.

        Usage:
            img = pygame.image.load(loader.asset("assets/logo.png"))
        """
        return os.path.join(self.bundle_dir, rel)

    #Debug 

    def __repr__(self) -> str:
        return (f"<GameLoader  data={self.data_dir!r}"
                f"  bundle={self.bundle_dir!r}"
                f"  loaded={list(self._data.keys())}>")

    def print_paths(self) -> None:
        """Print all resolved paths - useful for debugging packaging issues."""
        frozen = getattr(sys, "frozen", False)
        print(f"[Loader] frozen      = {frozen}")
        print(f"[Loader] bundle_dir  = {self.bundle_dir}")
        print(f"[Loader] data_dir    = {self.data_dir}")
        for name, spec in FILE_REGISTRY.items():
            full = self._abs(spec.path)
            exists = "V" if os.path.isfile(full) else "X (will create)"
            print(f"[Loader]   {name:15s} -> {full}  {exists}")


#
#Module-level singleton
#

loader = GameLoader().load_all()


#
#_ConfigBridge  -  drop-in for old "Config" class
#All other .py files keep:  from config import Config
#and nothing breaks.
#

class _ConfigBridge:

    def __getattr__(self, key: str) -> Any:
        #avoid infinite recursion on dunder/private
        if key.startswith("_"):
            raise AttributeError(key)
        cfg = loader.config
        if key in cfg:
            return cfg[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        loader.set_config(key, value)

    def __repr__(self) -> str:
        return f"<Config (JSON-backed) - {len(loader.config)} keys>"


Config = _ConfigBridge()