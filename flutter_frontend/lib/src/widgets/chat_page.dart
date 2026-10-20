import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../state/chat_controller.dart';
import '../state/voice_controller.dart';
import '../theme/diya_tokens.dart';
import 'chat_header.dart';
import 'composer_bar.dart';
import 'error_note.dart';
import 'message_bubble.dart';
import 'thinking_indicator.dart';
import 'welcome_view.dart';

/// The chat surface: header, message list or welcome hero, composer.
class ChatPage extends StatefulWidget {
  const ChatPage({
    required this.chat,
    required this.voice,
    required this.compact,
    required this.themeModeIcon,
    required this.onToggleTheme,
    required this.onShowAdmin,
    required this.onShowMemories,
    super.key,
  });

  final ChatController chat;
  final VoiceController voice;
  final bool compact;
  final IconData themeModeIcon;
  final VoidCallback onToggleTheme;
  final VoidCallback onShowAdmin;
  final VoidCallback onShowMemories;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _composer = TextEditingController();
  final _composerFocus = FocusNode();
  final _scroll = ScrollController();
  int _seenTick = 0;

  ChatController get chat => widget.chat;
  VoiceController get voice => widget.voice;

  @override
  void initState() {
    super.initState();
    chat.addListener(_onChatChanged);
    voice.addListener(_onVoiceChanged);
  }

  @override
  void dispose() {
    chat.removeListener(_onChatChanged);
    voice.removeListener(_onVoiceChanged);
    _composer.dispose();
    _composerFocus.dispose();
    _scroll.dispose();
    super.dispose();
  }

  void _onChatChanged() {
    if (chat.scrollTick == _seenTick) return;
    _seenTick = chat.scrollTick;
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToEnd());
  }

  void _onVoiceChanged() {
    if (voice.error != null) {
      final message = voice.error!;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        ScaffoldMessenger.of(context)
          ..hideCurrentSnackBar()
          ..showSnackBar(SnackBar(content: Text(message)));
        voice.clearError();
      });
    }
  }

  void _scrollToEnd() {
    if (!_scroll.hasClients) return;
    final nearBottom =
        _scroll.position.pixels > _scroll.position.maxScrollExtent - 260;
    if (!nearBottom) return;
    _scroll.animateTo(
      _scroll.position.maxScrollExtent,
      duration: const Duration(milliseconds: 220),
      curve: Curves.easeOut,
    );
  }

  void _send() {
    final text = _composer.text.trim();
    if (text.isEmpty) return;
    _composer.clear();
    chat.send(text);
    _composerFocus.requestFocus();
  }

  void _usePrompt(String prompt) {
    _composer.value = TextEditingValue(
      text: prompt,
      selection: TextSelection.collapsed(offset: prompt.length),
    );
    _composerFocus.requestFocus();
  }

  Future<void> _attach() async {
    if (chat.busy) return;
    final picked = await FilePicker.pickFiles(
      type: FileType.custom,
      allowedExtensions: [
        'pdf',
        'png',
        'jpg',
        'jpeg',
        'tif',
        'tiff',
        'bmp',
        'webp',
        'txt',
        'md',
        'csv',
        'docx',
        'html',
        'htm',
      ],
    );
    if (picked.isEmpty) return;
    await chat.uploadFile(picked.single);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        ChatHeader(
          model: chat.model,
          indexedChunks: chat.indexedChunks,
          mongoReady: chat.mongoReady,
          ttsReady: chat.ttsReady,
          sttReady: chat.sttReady,
          themeModeIcon: widget.themeModeIcon,
          onToggleTheme: widget.onToggleTheme,
          onShowAdmin: widget.onShowAdmin,
          compact: widget.compact,
        ),
        Expanded(
          child: chat.messages.isEmpty
              ? WelcomeView(
                  onPrompt: _usePrompt,
                  onAttach: _attach,
                  voiceReady: chat.sttReady || chat.ttsReady,
                )
              : _messageList(),
        ),
        if (chat.error != null)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: DiyaSpacing.s4),
            child: ErrorNote(
              message: chat.error!,
              onDismiss: chat.clearError,
            ),
          ),
        ComposerBar(
          controller: _composer,
          focusNode: _composerFocus,
          busy: chat.busy,
          recording: voice.recording,
          recordingSeconds: voice.recordingSeconds,
          deepThink: chat.deepThink,
          sttReady: chat.sttReady,
          onSend: _send,
          onStop: chat.stop,
          onAttach: _attach,
          onMicPressed: voice.toggleRecording,
          onDeepThinkChanged: chat.setDeepThink,
        ),
      ],
    );
  }

  Widget _messageList() {
    final showThinking =
        chat.busy && chat.messages.isNotEmpty && chat.messages.last.isUser;
    final itemCount = chat.messages.length + (showThinking ? 1 : 0);
    return ListView.builder(
      controller: _scroll,
      padding: const EdgeInsets.fromLTRB(
        DiyaSpacing.s5,
        DiyaSpacing.s5,
        DiyaSpacing.s5,
        DiyaSpacing.s5,
      ),
      itemCount: itemCount,
      itemBuilder: (context, index) {
        if (index == chat.messages.length) {
          return const ThinkingIndicator();
        }
        final message = chat.messages[index];
        final speakId = 'm$index';
        return MessageBubble(
          message: message,
          speakId: speakId,
          ttsReady: chat.ttsReady,
          speaking: voice.speaking && voice.playingId == speakId,
          onSpeak: () => voice.toggleSpeak(speakId, message.content),
        );
      },
    );
  }
}
