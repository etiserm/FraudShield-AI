# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

def keep_runtime_modules(module_name):
    """Exclude optional Android support and test-only imports from the bundle."""
    return ".tests" not in module_name and ".android" not in module_name

datas = [('site', 'site'), ('fraudshield_rf_model.joblib', '.')]
binaries = []
hiddenimports = []
hiddenimports += collect_submodules('uvicorn')
tmp_ret = collect_all('webview', filter_submodules=keep_runtime_modules)
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn', filter_submodules=keep_runtime_modules)
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy', filter_submodules=keep_runtime_modules)
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FraudShield AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX-compressed macOS binaries can fail Apple Silicon code-signature checks.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/fraudshield_icon.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='FraudShield AI',
)
app = BUNDLE(
    coll,
    name='FraudShield AI.app',
    icon='assets/fraudshield_icon.icns',
    bundle_identifier='org.veritas.fraudshieldai',
)
