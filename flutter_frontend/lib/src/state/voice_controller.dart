import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';

import '../../api_client.dart';
import '../voice/mic_capture.dart';

/// Piper playback + Vosk capture state for the chat surface.
///
/// [api] is late-bound through a getter because the controller outlives
/// individual connections.
class VoiceController extends ChangeNotifier {
  VoiceController(this._api);

  final DiyaApiClient? Function() _api;
  final AudioPlayer _player = AudioPlayer();

  bool speaking = false;
  String? playingId;
  bool recording = false;
  int recordingSeconds = 0;
  String? error;

  MicRecording? _activeRecording;
  Timer? _recordingTicker;

  bool get hasAudio => _api() != null;

  Future<void> toggleSpeak(String id, String text) async {
    if (speaking && playingId == id) {
      await stopSpeaking();
      return;
    }
    final client = _api();
    if (client == null || text.trim().isEmpty) return;
    await stopSpeaking();
    try {
      final bytes = await client.speak(text);
      speaking = true;
      playingId = id;
      error = null;
      notifyListeners();
      unawaited(_player.play(BytesSource(bytes)));
    } on DiyaApiException catch (e) {
      error = e.message;
      notifyListeners();
    }
  }

  Future<void> stopSpeaking() async {
    if (!speaking) return;
    speaking = false;
    playingId = null;
    await _player.stop();
    notifyListeners();
  }

  /// Toggles the mic. Returns transcript text when a recording finishes,
  /// null otherwise (start/cancel paths).
  Future<String?> toggleRecording() async {
    if (recording) {
      return _finishRecording();
    }
    final client = _api();
    if (client == null) return null;
    final session = await startMicRecording();
    if (session == null) {
      error = 'The microphone is not available on this device.';
      notifyListeners();
      return null;
    }
    _activeRecording = session;
    recording = true;
    recordingSeconds = 0;
    error = null;
    _recordingTicker = Timer.periodic(const Duration(seconds: 1), (timer) {
      recordingSeconds = timer.tick;
      notifyListeners();
    });
    notifyListeners();
    return null;
  }

  Future<String?> _finishRecording() async {
    final session = _activeRecording;
    _activeRecording = null;
    _recordingTicker?.cancel();
    _recordingTicker = null;
    recording = false;
    notifyListeners();
    if (session == null) return null;
    final bytes = await session.stop();
    if (bytes == null) return null;
    try {
      final result = await _api()!.transcribe(bytes);
      return result['text']?.toString();
    } on DiyaApiException catch (e) {
      error = e.message;
      notifyListeners();
      return null;
    }
  }

  void clearError() {
    error = null;
    notifyListeners();
  }

  @override
  void dispose() {
    unawaited(_player.dispose());
    _recordingTicker?.cancel();
    super.dispose();
  }
}
