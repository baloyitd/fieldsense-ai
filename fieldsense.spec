# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification file for FieldSense AI v3.0

Build executable:
    pyinstaller fieldsense.spec

Output:
    dist/fieldsense.exe (Windows)
    dist/fieldsense (Linux/Mac)
"""

block_cipher = None

a = Analysis(
    ['bin/ship.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('ui', 'ui'),
        ('data', 'data'),
        ('models', 'models'),
        ('README.md', '.'),
    ],
    hiddenimports=[
        'numpy',
        'cv2',
        'torch',
        'reportlab',
        'sklearn',
        'pandas',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='fieldsense',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ui/public/icon.ico' if os.path.exists('ui/public/icon.ico') else None,
)
