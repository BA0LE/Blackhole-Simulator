# BlackHoleSimulator.spec  —  --onedir build
# ════════════════════════════════════════════════════════════════════════════
# Build:
#   pyinstaller BlackHoleSimulator.spec
#
# Output structure:
#   dist/
#     BlackHoleSimulator/
#       BlackHoleSimulator.exe   ← chạy file này
#       _internal/               ← libs, .pyc (PyInstaller 6+)
#       config.json              ← tự tạo lần đầu chạy, user có thể edit
#       settings.json
#       achievements.json
#       assets/                  ← assets bundle (nếu có)
#
# JSON files được Loader.py ghi vào cùng thư mục với .exe
# nên user có thể backup / edit chúng dễ dàng.
# ════════════════════════════════════════════════════════════════════════════

from PyInstaller.building.api       import PYZ, EXE, COLLECT
from PyInstaller.building.build_main import Analysis

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],

    # ── Bundled read-only assets ──────────────────────────────────────────
    # Dùng loader.asset("assets/font.ttf") trong code để trỏ đúng path.
    datas=[
        # ('assets', 'assets'),   # uncomment nếu có folder assets/
        # ('fonts',  'fonts'),
    ],

    # ── Hidden imports (PyInstaller hay miss các thư viện này) ────────────
    hiddenimports=[
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        'scipy.interpolate',
        'scipy.interpolate._interpolate',
        'pygame',
    ],

    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries = True,        # --onedir: libs go into COLLECT
    name             = 'BlackHoleSimulator',
    debug            = False,
    strip            = False,
    upx              = False,       # set True nếu đã cài UPX (nhỏ hơn ~30%)
    console          = False,       # True = hiện terminal (tốt khi debug)
    icon           = 'Blackhole.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip       = False,
    upx         = False,
    upx_exclude = [],
    name        = 'BlackHoleSimulator',
)
