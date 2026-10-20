import 'dart:typed_data';

/// A started microphone recording; [stop] finishes and returns WAV bytes.
abstract class MicRecording {
  Future<Uint8List?> stop();
}

/// Starts a microphone recording where the platform supports it.
/// Returns null where dart:io (and the mic plugin) is unavailable.
Future<MicRecording?> startMicRecording() async => null;
