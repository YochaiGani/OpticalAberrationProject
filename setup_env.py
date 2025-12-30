import subprocess
import sys
import os
import pkg_resources

def install_package(package):
    """Installs a single package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed: {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install: {package}")

def ensure_tool_installed(tool_name):
    """Ensures a specific tool (like pipreqs) is installed."""
    try:
        pkg_resources.get_distribution(tool_name)
    except pkg_resources.DistributionNotFound:
        print(f"🔹 Tool '{tool_name}' not found. Installing...")
        install_package(tool_name)

def generate_requirements():
    """Scans the 'src' folder and generates requirements.txt automatically."""
    print("\n[1/2] Scanning project imports and updating requirements.txt...")
    ensure_tool_installed("pipreqs")
    cmd = [
        "pipreqs", 
        "src", 
        "--force", 
        "--savepath", "requirements.txt",
        "--ignore", ".venv,venv,__pycache__"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✅ requirements.txt has been updated based on your current code!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating requirements: {e}")
    except FileNotFoundError:
        # במקרה ש-pipreqs לא נמצא ב-PATH (קורה בווינדוס לפעמים), מנסים להריץ דרך פייתון
        subprocess.check_call([sys.executable, "-m", "pipreqs.pipreqs", "src", "--force", "--savepath", "requirements.txt"])
        print("✅ requirements.txt has been updated!")

def install_requirements():
    """Installs everything listed in requirements.txt."""
    print("\n[2/2] Installing dependencies from requirements.txt...")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found! Did the scan fail?")
        return

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n🎉 Success! All libraries are installed and up to date.")
    except subprocess.CalledProcessError:
        print("\n⚠️ Some packages failed to install. Check the error log above.")

def main():
    print("=== 🛠️  Optical Project Environment Manager 🛠️  ===")
    print("This script will scan your code and install necessary libraries.")
    generate_requirements()
    install_requirements()
    print("\nDone. You can now run the project.")

if __name__ == "__main__":
    main()