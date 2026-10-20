import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;

class DiyaApiException implements Exception {
  const DiyaApiException(this.message, {this.statusCode});
  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

/// One server-sent event from POST /query/stream.
class QueryEvent {
  const QueryEvent._(this.type, {this.text = '', this.sources = const [], this.detail = '', this.timing = const {}});

  final String type;
  final String text;
  final List<Map<String, dynamic>> sources;
  final String detail;
  final Map<String, dynamic> timing;

  bool get isDelta => type == 'delta';
  bool get isFinal => type == 'final';
  bool get isError => type == 'error';

  factory QueryEvent.fromJson(Map<String, dynamic> json) => QueryEvent._(
    json['type']?.toString() ?? 'error',
    text: json['text']?.toString() ?? '',
    sources: (json['sources'] as List? ?? [])
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e))
        .toList(),
    detail: json['detail']?.toString() ?? '',
    timing: Map<String, dynamic>.from(json['timing'] as Map? ?? {}),
  );
}

class DiyaApiClient {
  DiyaApiClient({
    required this.baseUrl,
    required this.username,
    required this.password,
    http.Client? client,
  }) : _client = client ?? http.Client();

  final String baseUrl;
  final String username;
  final String password;
  final http.Client _client;

  Uri _uri(String path) => Uri.parse(baseUrl).resolve(path);

  Map<String, String> get _headers => {
    'Authorization':
        'Basic ${base64Encode(utf8.encode('$username:$password'))}',
    'Accept': 'application/json',
  };

  Future<Map<String, dynamic>> get(String path) async {
    final response = await _client.get(_uri(path), headers: _headers);
    return _decode(response);
  }

  Future<Map<String, dynamic>> post(
    String path,
    Map<String, dynamic> body,
  ) async {
    final response = await _client.post(
      _uri(path),
      headers: {..._headers, 'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    return _decode(response);
  }

  Future<Map<String, dynamic>> delete(String path) async {
    final response = await _client.delete(_uri(path), headers: _headers);
    return _decode(response);
  }

  Future<Map<String, dynamic>> upload(
    PlatformFile file, {
    String? conversationId,
  }) async {
    final bytes = await file.readAsBytes();
    final request = http.MultipartRequest('POST', _uri('/read-file'))
      ..headers.addAll(_headers)
      ..files.add(
        http.MultipartFile.fromBytes('file', bytes, filename: file.name),
      );
    if (conversationId != null) {
      request.fields['conversation_id'] = conversationId;
    }
    final response = await http.Response.fromStream(await request.send());
    return _decode(response);
  }

  Future<Map<String, dynamic>> listKnowledge() => get('/knowledge');

  Future<Map<String, dynamic>> uploadKnowledge(PlatformFile file) async {
    final bytes = await file.readAsBytes();
    final request = http.MultipartRequest('POST', _uri('/knowledge'))
      ..headers.addAll(_headers)
      ..files.add(
        http.MultipartFile.fromBytes('file', bytes, filename: file.name),
      );
    final response = await http.Response.fromStream(await request.send());
    return _decode(response);
  }

  Future<Map<String, dynamic>> removeKnowledge(String filename) =>
      delete('/knowledge/${Uri.encodeComponent(filename)}');

  /// Speaks [text] through Piper. Returns WAV audio bytes.
  Future<Uint8List> speak(String text) async {
    final response = await _client.post(
      _uri('/voice/tts'),
      headers: {
        ..._headers,
        'Content-Type': 'application/json',
        'Accept': 'audio/wav',
      },
      body: jsonEncode({'text': text}),
    );
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw _decodeException(response);
    }
    return response.bodyBytes;
  }

  /// Transcribes a locally recorded 16-bit PCM WAV with Vosk.
  Future<Map<String, dynamic>> transcribe(Uint8List wavBytes) async {
    final request = http.MultipartRequest('POST', _uri('/voice/transcribe'))
      ..headers.addAll(_headers)
      ..files.add(
        http.MultipartFile.fromBytes('file', wavBytes, filename: 'voice.wav'),
      );
    final response = await http.Response.fromStream(await request.send());
    return _decode(response);
  }

  /// Streams Diya's answer token-by-token over SSE. The caller may cancel
  /// the returned subscription to stop generation early.
  Stream<QueryEvent> queryStream({
    required String question,
    bool deepThink = false,
    String? conversationId,
  }) {
    final payload = <String, dynamic>{
      'question': question,
      'deep_think': deepThink,
      'conversation_id': ?conversationId,
    };
    late StreamController<QueryEvent> controller;
    http.StreamedResponse? response;

    controller = StreamController<QueryEvent>(
      onListen: () async {
        final request = http.Request('POST', _uri('/query/stream'))
          ..headers.addAll({
            ..._headers,
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
          })
          ..body = jsonEncode(payload);
        try {
          response = await _client.send(request);
          if (response!.statusCode != 200) {
            final body = await response!.stream.bytesToString();
            controller.addError(_exceptionFromBody(response!.statusCode, body));
            await controller.close();
            return;
          }
          final lines = utf8.decoder
              .bind(response!.stream)
              .transform(const LineSplitter());
          await for (final line in lines) {
            if (!line.startsWith('data:')) continue;
            final json = jsonDecode(line.substring(5).trim());
            controller.add(QueryEvent.fromJson(json));
          }
          await controller.close();
        } catch (e, stack) {
          if (!controller.isClosed) {
            controller.addError(
              e is DiyaApiException
                  ? e
                  : DiyaApiException('The stream failed: $e'),
              stack,
            );
            await controller.close();
          }
        }
      },
      onCancel: () async {
        // Dropping the connection tells the backend to stop generating.
        response?.stream.listen((_) {}).cancel();
        await controller.close();
      },
    );
    return controller.stream;
  }

  DiyaApiException _decodeException(http.Response response) {
    dynamic decoded;
    try {
      decoded = jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException {
      decoded = null;
    }
    final detail = decoded is Map ? decoded['detail'] : null;
    return DiyaApiException(
      detail?.toString() ??
          switch (response.statusCode) {
            401 => 'Username or password is incorrect.',
            _ => 'Diya returned an error (${response.statusCode}).',
          },
      statusCode: response.statusCode,
    );
  }

  DiyaApiException _exceptionFromBody(int statusCode, String body) {
    dynamic decoded;
    try {
      decoded = jsonDecode(body);
    } on FormatException {
      decoded = null;
    }
    final detail = decoded is Map ? decoded['detail'] : null;
    return DiyaApiException(
      detail?.toString() ??
          (statusCode == 401
              ? 'Username or password is incorrect.'
              : 'Diya returned an error ($statusCode).'),
      statusCode: statusCode,
    );
  }

  Map<String, dynamic> _decode(http.Response response) {
    dynamic decoded;
    try {
      decoded = jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException {
      decoded = null;
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = decoded is Map ? decoded['detail'] : null;
      final message =
          detail?.toString() ??
          switch (response.statusCode) {
            401 => 'Username or password is incorrect.',
            503 => utf8.decode(response.bodyBytes),
            _ => 'Diya returned an error (${response.statusCode}).',
          };
      throw DiyaApiException(message, statusCode: response.statusCode);
    }
    if (decoded is Map<String, dynamic>) return decoded;
    return <String, dynamic>{};
  }

  void close() => _client.close();
}
