'''
Coop-Drive launcher (what CoopDrive.bat runs): starts the CARLA server when it is not running, opens the dashboard and
reopens it whenever the dashboard asks (restart / mode / pipeline commands), with a fresh Traffic Manager port each time.
Crashes are retried 3 times (the server is restarted if it died); a queued pipeline job resumes where it was.

  python app.py              run
  python app.py --shortcut   (re)create the desktop shortcut
'''
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
import appcfg  # noqa: E402

RESTART, SERVER_RESTART = 75, 76  # = dashboard.RESTART, dashboard.SERVER_RESTART
SERVER = 'CarlaUE4-Win64-Shipping.exe'


def server_running():
    out = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {SERVER}', '/NH'], capture_output=True, text=True).stdout
    return 'carlaue4-win64' in out.lower()  # tasklist cuts long image names ("CarlaUE4-Win64-Shipping.e")


def server_answers():
    probe = "import sys; sys.path.insert(0, '.'); from simulation.connection import carla; " \
            "c = carla.Client('localhost', 2000); c.set_timeout(3); c.get_server_version()"
    return subprocess.run([sys.executable, '-c', probe], capture_output=True).returncode == 0


def start_server(cfg):
    exe = os.path.join(cfg['carla_root'], 'CarlaUE4.exe')
    while not os.path.exists(exe):
        print(f'CARLA 0.9.10 not found at {cfg["carla_root"]}')
        cfg['carla_root'] = input('CARLA folder (the one with CarlaUE4.exe): ').strip().strip('"')
        exe = os.path.join(cfg['carla_root'], 'CarlaUE4.exe')
        appcfg.save(cfg)
    print('starting the CARLA server…')
    subprocess.Popen([exe, f"-quality-level={cfg['carla_quality']}", '-windowed', '-ResX=640', '-ResY=480'], cwd=cfg['carla_root'])


def stop_server():
    subprocess.run(['taskkill', '/F', '/IM', SERVER], capture_output=True)
    subprocess.run(['taskkill', '/F', '/IM', 'CarlaUE4.exe'], capture_output=True)


def ensure_server(cfg):
    if not server_running():
        start_server(cfg)
    print('waiting for the CARLA server', end='', flush=True)
    for _ in range(90):
        if server_answers():
            print(' ready')
            return True
        print('.', end='', flush=True)
        time.sleep(2)
    print(' no answer')
    return False


def shortcut():
    '''Desktop shortcut "Coop-Drive" with the project icon, pointing at CoopDrive.bat.'''
    ps = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\\Coop-Drive.lnk'); "
          f"$s.TargetPath = '{os.path.join(HERE, 'CoopDrive.bat')}'; $s.WorkingDirectory = '{HERE}'; "
          f"$s.IconLocation = '{os.path.join(HERE, 'assets', 'coopdrive.ico')}'; $s.WindowStyle = 7; $s.Save()")
    subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True)


def single_instance():
    '''True for the first launcher; a second one (double click, two shortcuts) would drive the same CARLA world.
    A named mutex dies with its process, so a crash leaves no stale lock.'''
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
    except Exception:  # shortcut: no guard off Windows, add a lock file if the app is ever run elsewhere
        return True
    single_instance.handle = k32.CreateMutexW(None, False, 'Local\CoopDriveLauncher')
    return k32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def main():
    if '--shortcut' in sys.argv:
        return shortcut()
    if not single_instance():
        print('Coop-Drive is already running · this window closes in 5 s')
        time.sleep(5)
        return
    if not os.path.exists(os.path.join(os.path.expanduser('~'), 'Desktop', 'Coop-Drive.lnk')):
        shortcut()
    try:  # no idle sleep while the app runs: overnight collection / training must not stop (display may still turn off)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except Exception:
        pass
    port, fails = int(os.environ.get('COOP_TM_PORT', '8100')), 0
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    while True:
        cfg = appcfg.load()
        if not ensure_server(cfg):
            stop_server()
            time.sleep(5)
            continue
        port += 1  # the Traffic Manager of a finished process can linger on its port
        env['COOP_TM_PORT'] = str(port)
        print(f'dashboard · mode {cfg["mode"]} · TM port {port}')
        started = time.time()
        rc = subprocess.call([sys.executable, '-X', 'faulthandler', 'dashboard.py', '--app'], env=env)
        if time.time() - started > 600:  # it made progress before failing (CARLA 0.9.10 crashes now and then): fresh retries
            fails = 0
        if rc == 0:
            return
        if rc == SERVER_RESTART:
            stop_server()
            time.sleep(5)
            continue
        if rc == RESTART:
            fails = 0
            continue
        fails += 1
        print(f'dashboard exited with code {rc} ({fails}/3)')
        if fails > 3:
            input('stopped after 3 failed restarts · see logs/ · press Enter to close')
            return
        if not server_answers():
            stop_server()
        time.sleep(5)


if __name__ == '__main__':
    main()
