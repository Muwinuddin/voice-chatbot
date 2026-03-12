import subprocess
import sys


def main() -> None:
    command = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
