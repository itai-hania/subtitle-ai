import os
import sys
import subprocess
import platform


def check_venv_python_version(venv_python: str):
    """Verify the venv Python is 3.10+ (required for latest yt-dlp)."""
    try:
        result = subprocess.run(
            [venv_python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True
        )
        version_str = result.stdout.strip()
        major, minor = map(int, version_str.split('.'))
        if major < 3 or (major == 3 and minor < 10):
            print(f"Error: venv Python is {version_str}, but 3.10+ is required.")
            print("Latest yt-dlp (needed for working YouTube downloads) requires Python 3.10+.")
            print("\nTo fix, recreate the venv with a newer Python:")
            print("  rm -rf venv")
            print("  python3.11 -m venv venv")
            print("  venv/bin/pip install -r requirements.txt")
            sys.exit(1)
    except Exception:
        pass  # If we can't check, let it proceed and fail naturally


def main():
    """
    Cross-platform launcher for SubtitleAI.
    Detects the operating system and runs app.py using the virtual environment's Python interpreter.
    """
    system = platform.system()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths based on OS
    if system == "Windows":
        venv_python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    else:
        # macOS / Linux
        venv_python = os.path.join(project_dir, "venv", "bin", "python")
        
    app_script = os.path.join(project_dir, "app.py")

    # validation
    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment python not found at: {venv_python}")
        print("Please ensure the 'venv' directory exists and is valid.")
        sys.exit(1)
        
    if not os.path.exists(app_script):
        print(f"Error: App script not found at: {app_script}")
        sys.exit(1)

    check_venv_python_version(venv_python)

    print(f"🚀 Starting SubtitleAI on {system}...")
    
    try:
        # Execute app.py using the venv python
        # This is equivalent to 'source venv/bin/activate && python app.py'
        subprocess.check_call([venv_python, app_script])
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Application crashed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
