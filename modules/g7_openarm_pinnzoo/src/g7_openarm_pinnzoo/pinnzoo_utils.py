import platform


class NotSupportedArchitecture(Exception):
    pass

def get_arch():
    machine = platform.machine().lower()
    
    if machine in ('x86_64', 'amd64'):
        return 'x86_64'
    elif machine in ('aarch64', 'arm64'):
        return 'aarch64'

    raise NotSupportedArchitecture(f'{platform.machine().lower()} is not support')
