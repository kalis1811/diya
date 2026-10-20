import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'mic_capture_stub.dart';

export 'mic_capture_stub.dart';

class _IoMicRecording implements MicRecording {
  _IoMicRecording(this._recorder);

  final AudioRecorder _recorder;
  bool _stopped = false;

  @override
  Future<Uint8List?> stop() async {
    if (_stopped) return null;
    _stopped = true;
    try {
      final saved = await _recorder.stop();
      if (saved == null) return null;
      final bytes = await File(saved).readAsBytes();
      await File(saved).delete();
      return bytes.isEmpty ? null : bytes;
    } finally {
      _recorder.dispose();
    }
  }
}

Future<MicRecording?> startMicRecording() async {
  final recorder = AudioRecorder();
  if (!await recorder.hasPermission()) {
    recorder.dispose();
    return null;
  }
  final dir = await getTemporaryDirectory();
  final path =
      '${dir.path}${Platform.pathSeparator}diya_voice_'
      '${DateTime.now().millisecondsSinceEpoch}.wav';
  await recorder.start(
    const RecordConfig(
      encoder: AudioEncoder.wav,
      sampleRate: 16000,
      numChannels: 1,
    ),
    path: path,
  );
  return _IoMicRecording(recorder);
}
