import os
import sys
import winshell
from win32com.client import Dispatch

def create_shortcut(startup=False):
    desktop = winshell.desktop()
    startup_dir = winshell.startup()
    
    label = "Whispr Flow"
    if startup:
        path = os.path.join(startup_dir, f"{label}.lnk")
    else:
        path = os.path.join(desktop, f"{label}.lnk")
        
    target = sys.executable.replace("python.exe", "pythonw.exe")
    if not os.path.exists(target):
        target = sys.executable # Fallback
    # Running without console: use pythonw.exe if available or just python.exe
    # For now, we point to the main.py
    main_script = os.path.abspath("main.py")
    w_dir = os.path.abspath(".")
    icon = target # Use python icon for now

    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(path)
    shortcut.Targetpath = target
    shortcut.Arguments = f'"{main_script}"'
    shortcut.WorkingDirectory = w_dir
    shortcut.IconLocation = target
    shortcut.save()
    
    print(f"Shortcut created on Desktop: {path}")

if __name__ == "__main__":
    # Ensure pywin32 and winshell are installed
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        print("Installing required packages for shortcut creation...")
        os.system(f"{sys.executable} -m pip install pywin32 winshell")
        import winshell
        from win32com.client import Dispatch
        
    create_shortcut(startup=False) # Desktop
    create_shortcut(startup=True)  # Startup
    create_shortcut(startup=False)
