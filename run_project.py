import os
import subprocess
import sys
import webbrowser

VENV_DIR = "venv"
PYTHON_PATH = r"venv\Scripts\python.exe"
PIP_PATH = r"venv\Scripts\pip.exe"

def create_virtual_env():
    if not os.path.exists(VENV_DIR):
        print("\n📦 Creating virtual environment (one-time setup)...\n")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("✔ Virtual environment already exists. Skipping creation.")

def is_streamlit_installed():
    try:
        subprocess.check_output([PYTHON_PATH, "-m", "pip", "show", "streamlit"])
        return True
    except:
        return False

def install_requirements():
    if not is_streamlit_installed():
        print("\n📦 Installing dependencies...\n")
        subprocess.check_call([PIP_PATH, "install", "-r", "requirements.txt"])
    else:
        print("✔ Streamlit & dependencies already installed.")

def start_streamlit():
    print("\n🚀 Launching Exercise Form Analyzer...\n")

    url = "http://localhost:8501"
    webbrowser.open(url)

    cmd = f"{PYTHON_PATH} -m streamlit run humantrack.py --server.headless false --server.address localhost"
    os.system(cmd)

if __name__ == "__main__":
    create_virtual_env()
    install_requirements()
    start_streamlit()
