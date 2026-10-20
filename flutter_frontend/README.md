# Diya Flutter frontend

This Flutter client connects to the Diya API. It includes owner sign-in, chat,
saved conversations, temporary document reading, a persistent knowledge library,
memory management, and the admin status panel.

## Run on Windows

From the repository root, start the API and Flutter desktop app with
`run_diya.bat`. To run the Flutter client in development, open a terminal in
this folder and run:

```powershell
flutter pub get
flutter run -d windows
```

The backend address defaults to `http://127.0.0.1:8400`. Sign in with the
owner username and password saved in the repository root `.env` file.

## Run on an Android emulator

Keep the backend running on the Windows host, start an Android emulator, then list devices and run the app:

```powershell
flutter devices
flutter run -d <device-id>
```

The Android emulator address defaults to `http://10.0.2.2:8400`, which routes to the Windows host's local backend. For a physical phone, the backend must be made reachable on the local network and the app's **Backend address** changed to the host's LAN address.

The username and password stay in memory for the current app session and are cleared when the app closes or disconnects.

## Run on the web

The API does not enable cross-origin requests. Build with the backend's static path and open the Flutter page on the same origin:

```powershell
flutter build web --base-href /static/flutter/
New-Item -ItemType Directory -Force ..\brain\static\flutter | Out-Null
Copy-Item -Recurse -Force build\web\* ..\brain\static\flutter\
```

Then open `http://127.0.0.1:8400/static/flutter/index.html` while Diya is running. The web client uses the backend origin by default.
