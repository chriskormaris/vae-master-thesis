import sys

import cx_Freeze

base = None

if sys.platform == 'win64':
    base = 'Win64GUI'
elif sys.platform == 'win32':
    base = 'Win32GUI'

icons_path = 'icons\\'
executables = [cx_Freeze.Executable('vaes_gui.py', base=base, icon=icons_path + 'vaes.png')]
# executables = [cx_Freeze.Executable('vaes_gui.py', icon='icons/vaes.png')]
include_files = [icons_path + x for x in ['vaes.png', 'help.png', 'info.png', 'aueb_logo_256.png']]
includes = []
excludes = ['Tkinter']
packages = ['numpy', 'matplotlib', 'tensorflow', 'keras', 'torchvision']
build_exe_options = {'includes': includes, 'packages': packages, 'excludes': excludes, 'include_files': include_files}

cx_Freeze.setup(
    name='vaes',
    options={'build_exe': build_exe_options},
    version='1.0.0',
    description='Variational Autoencoders & Missing Values Completion Algorithm',
    executables=executables
)
