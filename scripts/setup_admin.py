"""Set up Diya's one-owner web login without echoing the password."""
from __future__ import annotations

from getpass import getpass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
USERNAME_KEY = "BRAIN_ADMIN_USERNAME"
PASSWORD_KEY = "BRAIN_ADMIN_PASSWORD"


def main() -> None:
    username = input("Admin username [admin]: ").strip() or "admin"
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,64}", username):
        raise SystemExit("Use 3–64 letters, numbers, dots, underscores, or hyphens for the username.")
    while True:
        password = getpass("Choose an admin password (at least 10 characters): ")
        if len(password) < 10:
            print("Use at least 10 characters.")
            continue
        if password != password.strip():
            print("Do not start or end the password with spaces.")
            continue
        confirmation = getpass("Enter it again: ")
        if password != confirmation:
            print("Those passwords did not match. Try again.")
            continue
        break

    existing = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    kept = [line for line in existing
            if not line.lstrip().startswith(f"{USERNAME_KEY}=")
            and not line.lstrip().startswith(f"{PASSWORD_KEY}=")]
    kept.extend((f"{USERNAME_KEY}={username}", f"{PASSWORD_KEY}={password}"))
    ENV_FILE.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"Single-owner login saved in {ENV_FILE}. The password was not displayed.")
    print("Restart Diya, then open the app and sign in with this username and password.")


if __name__ == "__main__":
    main()
