#!/usr/bin/env python3

import sys
import platform

# Get platform system
plat_system = platform.system().lower()

if plat_system == 'darwin':
    # On macOS, check for Apple Silicon
    mac_machine = platform.machine()
    if mac_machine == 'arm64':
        print('macosx_12_0_arm64')
    elif mac_machine == 'x86_64' or mac_machine == 'i386':
        print('macosx_10_15_x86_64')
    else:
        print('Unsupported macOS architecture:', mac_machine, file=sys.stderr)
        sys.exit(1)
elif plat_system == 'linux':
    if platform.machine().startswith('x86_64'):
        print('manylinux_2_17_x86_64.manylinux2014_x86_64')
    elif platform.machine().startswith('aarch64'):
        print('manylinux_2_17_aarch64.manylinux2014_aarch64')
    else:
        print('Unsupported Linux architecture:', platform.machine(), file=sys.stderr)
        sys.exit(1)
elif plat_system == 'windows':
    win_machine = platform.machine()
    if win_machine.endswith('64'):
        print('win_amd64')
    elif win_machine == 'arm64':
        print('win_arm64')
    else:
        print('Unsupported Windows architecture:', win_machine, file=sys.stderr)
        sys.exit(1)
else:
    print('Unsupported system:', plat_system, file=sys.stderr)
    sys.exit(1)

sys.exit(0)