# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['vaes_gui.py'],
             pathex=['.'],
             binaries=[],
             datas=[('__init__.py', './'),
                    ('icons\\aueb_logo.png', 'icons'),
                    ('icons\\help.ico', 'icons'),
                    ('icons\\info.ico', 'icons'),
                    ('icons\\vaes.ico', 'icons')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['tk'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='vaes_gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon='icons\\vaes.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='vaes_gui')
