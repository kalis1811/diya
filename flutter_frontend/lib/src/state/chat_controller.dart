import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';

import '../../api_client.dart';

/// One turn in the chat surface. Mutable so a streaming answer can grow
/// token-by-token without rebuilding the whole list.
class ChatMessage {
  ChatMessage({
    required this.role,
    required this.content,
    List<Map<String, dynamic>>? sources,
    this.streaming = false,
    this.timing = const {},
  }) : sources = sources ?? [];

  final String role;
  String content;
  List<Map<String, dynamic>> sources;
  bool streaming;
  Map<String, dynamic> timing;

  bool get isUser => role == 'user';
}

/// Session state for one Diya connection: health, conversations, messages,
/// streaming query lifecycle.
class ChatController extends ChangeNotifier {
  DiyaApiClient? _api;
  StreamSubscription<QueryEvent>? _querySub;
  DateTime _lastFlush = DateTime.fromMillisecondsSinceEpoch(0);
  final StringBuffer _pendingDeltas = StringBuffer();
  ChatMessage? _streamingMessage;

  List<ChatMessage> messages = [];
  List<Map<String, dynamic>> conversations = [];
  Map<String, dynamic>? health;
  String? conversationId;
  String? error;
  bool connecting = false;
  bool busy = false;
  bool deepThink = false;

  /// Bumped whenever the message list should scroll down (new content).
  int scrollTick = 0;

  static const _flushInterval = Duration(milliseconds: 80);

  DiyaApiClient? get api => _api;
  bool get connected => _api != null;
  bool get mongoReady =>
      (health?['storage']?.toString() ?? '').startsWith('MongoDB connected');
  bool get ttsReady => health?['voice']?['tts'] == 'ready';
  bool get sttReady => health?['voice']?['stt'] == 'ready';
  String get model => health?['model']?.toString() ?? '';
  int get indexedChunks => health?['indexed_chunks'] as int? ?? 0;

  Future<void> connect(String baseUrl, String username, String password) async {
    connecting = true;
    error = null;
    notifyListeners();
    final normalized = baseUrl.trim().replaceFirst(RegExp(r'/+$'), '');
    final client = DiyaApiClient(
      baseUrl: normalized,
      username: username.trim(),
      password: password,
    );
    try {
      await client.get('/about');
      health = await client.get('/health');
      _api = client;
      connecting = false;
      notifyListeners();
      await refreshConversations();
      await refreshHealth();
    } catch (e) {
      client.close();
      connecting = false;
      error = e is DiyaApiException
          ? e.message
          : 'Could not reach Diya. Check the server address and try again.';
      notifyListeners();
    }
  }

  void disconnect() {
    _querySub?.cancel();
    _api?.close();
    _api = null;
    messages = [];
    conversations = [];
    health = null;
    conversationId = null;
    error = null;
    busy = false;
    notifyListeners();
  }

  Future<void> refreshHealth() async {
    final client = _api;
    if (client == null) return;
    try {
      health = await client.get('/health');
      notifyListeners();
    } on DiyaApiException {
      // A transient health failure must not tear down the chat surface.
    }
  }

  Future<void> refreshConversations() async {
    final client = _api;
    if (client == null || !mongoReady) {
      conversations = [];
      notifyListeners();
      return;
    }
    try {
      final result = await client.get('/conversations');
      conversations = (result['conversations'] as List? ?? [])
          .cast<Map<String, dynamic>>();
      notifyListeners();
    } on DiyaApiException {
      // Keep the last known list; the next refresh will retry.
    }
  }

  Future<void> newConversation() async {
    if (busy) return;
    _querySub?.cancel();
    conversationId = null;
    if (mongoReady && _api != null) {
      try {
        final result = await _api!.post('/conversations', {});
        conversationId = result['id']?.toString();
      } on DiyaApiException catch (e) {
        error = e.message;
      }
    }
    messages = [];
    error = null;
    notifyListeners();
    await refreshConversations();
  }

  Future<void> openConversation(String id) async {
    if (busy || _api == null) return;
    try {
      final result = await _api!.get('/conversations/${Uri.encodeComponent(id)}');
      conversationId = id;
      messages = (result['messages'] as List? ?? [])
          .whereType<Map>()
          .map(
            (m) => ChatMessage(
              role: m['role']?.toString() ?? 'assistant',
              content: m['content']?.toString() ?? '',
              sources: (m['sources'] as List? ?? [])
                  .whereType<Map>()
                  .map((s) => Map<String, dynamic>.from(s))
                  .toList(),
            ),
          )
          .toList();
      error = null;
      notifyListeners();
      scrollTick++;
    } on DiyaApiException catch (e) {
      error = e.message;
      notifyListeners();
    }
  }

  Future<void> deleteConversation(String id) async {
    try {
      await _api?.delete('/conversations/${Uri.encodeComponent(id)}');
      if (conversationId == id) {
        conversationId = null;
        messages = [];
      }
      await refreshConversations();
    } on DiyaApiException catch (e) {
      error = e.message;
      notifyListeners();
    }
  }

  void setDeepThink(bool value) {
    deepThink = value;
    notifyListeners();
  }

  /// Sends [question] and streams Diya's reply into a new assistant message.
  Future<void> send(String question) async {
    final client = _api;
    final text = question.trim();
    if (text.isEmpty || busy || client == null) return;
    messages.add(ChatMessage(role: 'user', content: text));
    busy = true;
    error = null;
    _streamingMessage = ChatMessage(role: 'assistant', content: '', streaming: true);
    messages.add(_streamingMessage!);
    _lastFlush = DateTime.now();
    scrollTick++;
    notifyListeners();

    final stream = client.queryStream(
      question: text,
      deepThink: deepThink,
      conversationId: mongoReady ? conversationId : null,
    );
    _querySub = stream.listen(_onEvent, onError: _onStreamError, onDone: _onStreamDone);
  }

  /// Stops the in-flight answer early; the partial text stays visible.
  void stop() {
    if (!busy) return;
    _querySub?.cancel();
    _querySub = null;
    busy = false;
    _flushDeltas(force: true);
    _streamingMessage?.streaming = false;
    _streamingMessage = null;
    notifyListeners();
  }

  Future<void> uploadFile(PlatformFile file) async {
    final client = _api;
    if (client == null || busy) return;
    messages.add(
      ChatMessage(role: 'user', content: 'Please read this file: ${file.name}'),
    );
    busy = true;
    error = null;
    scrollTick++;
    notifyListeners();
    try {
      final result = await client.upload(
        file,
        conversationId: mongoReady ? conversationId : null,
      );
      messages.add(
        ChatMessage(
          role: 'assistant',
          content:
              result['summary']?.toString() ??
              'The file was read, but no summary was returned.',
          sources: (result['sources'] as List? ?? [])
              .whereType<Map>()
              .map((s) => Map<String, dynamic>.from(s))
              .toList(),
        ),
      );
      await refreshConversations();
      unawaited(refreshHealth());
    } on DiyaApiException catch (e) {
      error = e.message;
    } catch (_) {
      error = 'Could not read the file. Try another file.';
    } finally {
      busy = false;
      scrollTick++;
      notifyListeners();
    }
  }

  void clearError() {
    error = null;
    notifyListeners();
  }

  void _onEvent(QueryEvent event) {
    if (event.isDelta) {
      _pendingDeltas.write(event.text);
      _flushDeltas();
    } else if (event.isFinal) {
      _flushDeltas(force: true);
      final message = _streamingMessage;
      if (message != null) {
        message.streaming = false;
        message.sources = event.sources;
        message.timing = event.timing;
      }
      _streamingMessage = null;
      busy = false;
      _querySub = null;
      unawaited(refreshConversations());
      unawaited(refreshHealth());
      scrollTick++;
      notifyListeners();
    } else if (event.isError) {
      _finishWithError(event.detail);
    }
  }

  void _onStreamError(Object e) {
    _finishWithError(e is DiyaApiException ? e.message : 'The stream failed: $e');
  }

  void _onStreamDone() {
    if (!busy) return;
    busy = false;
    _flushDeltas(force: true);
    _streamingMessage?.streaming = false;
    _streamingMessage = null;
    notifyListeners();
  }

  void _finishWithError(String message) {
    busy = false;
    _flushDeltas(force: true);
    _streamingMessage?.streaming = false;
    _streamingMessage = null;
    _querySub = null;
    error = message;
    notifyListeners();
  }

  void _flushDeltas({bool force = false}) {
    final now = DateTime.now();
    if (!force && now.difference(_lastFlush) < _flushInterval) return;
    if (_pendingDeltas.isEmpty) return;
    _streamingMessage?.content += _pendingDeltas.toString();
    _pendingDeltas.clear();
    _lastFlush = now;
    scrollTick++;
    notifyListeners();
  }

  @override
  void dispose() {
    _querySub?.cancel();
    _api?.close();
    super.dispose();
  }
}
