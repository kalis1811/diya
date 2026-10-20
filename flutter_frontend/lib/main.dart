import 'package:flutter/material.dart';

import 'src/state/chat_controller.dart';
import 'src/state/voice_controller.dart';
import 'src/theme/diya_theme.dart';
import 'src/theme/diya_tokens.dart';
import 'src/widgets/admin_dialog.dart';
import 'src/widgets/app_shell.dart';
import 'src/widgets/connect_screen.dart';
import 'src/widgets/memory_sheet.dart';
import 'src/widgets/knowledge_sheet.dart';

void main() => runApp(const DiyaApp());

class DiyaApp extends StatefulWidget {
  const DiyaApp({super.key});

  @override
  State<DiyaApp> createState() => _DiyaAppState();
}

class _DiyaAppState extends State<DiyaApp> {
  bool _dark = true;

  void _toggleTheme() => setState(() => _dark = !_dark);

  IconData get _themeIcon =>
      _dark ? Icons.light_mode_outlined : Icons.dark_mode_outlined;

  @override
  Widget build(BuildContext context) {
    final mode = _dark ? Brightness.dark : Brightness.light;
    return DiyaTokens(
      brightness: mode,
      child: MaterialApp(
        title: 'Diya',
        debugShowCheckedModeBanner: false,
        builder: (context, child) => DefaultTextStyle.merge(
          style: const TextStyle(decoration: TextDecoration.none),
          child: child ?? const SizedBox.shrink(),
        ),
        theme: buildDiyaTheme(Brightness.light),
        darkTheme: buildDiyaTheme(Brightness.dark),
        themeMode: _dark ? ThemeMode.dark : ThemeMode.light,
        home: DiyaHome(
          themeModeIcon: _themeIcon,
          onToggleTheme: _toggleTheme,
        ),
      ),
    );
  }
}

/// Root switch: connect screen until a server session exists, then the
/// chat shell.
class DiyaHome extends StatefulWidget {
  const DiyaHome({
    required this.themeModeIcon,
    required this.onToggleTheme,
    super.key,
  });

  final IconData themeModeIcon;
  final VoidCallback onToggleTheme;

  @override
  State<DiyaHome> createState() => _DiyaHomeState();
}

class _DiyaHomeState extends State<DiyaHome> {
  late final ChatController _chat = ChatController();
  late final VoiceController _voice = VoiceController(() => _chat.api);

  @override
  void dispose() {
    _voice.dispose();
    _chat.dispose();
    super.dispose();
  }

  Future<void> _showMemories() async {
    final api = _chat.api;
    if (api == null) return;
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (context) => MemorySheet(api: api),
    );
  }

  Future<void> _showAdmin() async {
    final api = _chat.api;
    if (api == null) return;
    await showDialog<void>(
      context: context,
      builder: (context) => AdminDialog(api: api),
    );
  }

  Future<void> _showKnowledge() async {
    final api = _chat.api;
    if (api == null) return;
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (context) => KnowledgeSheet(
        api: api,
        onChanged: () => _chat.refreshHealth(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
    animation: _chat,
    builder: (context, child) {
      if (!_chat.connected) {
        return ConnectScreen(
          connecting: _chat.connecting,
          error: _chat.error,
          onConnect: _chat.connect,
        );
      }
      return AppShell(
        chat: _chat,
        voice: _voice,
        themeModeIcon: widget.themeModeIcon,
        onToggleTheme: widget.onToggleTheme,
        onShowMemories: _showMemories,
        onShowKnowledge: _showKnowledge,
        onShowAdmin: _showAdmin,
      );
    },
  );
}
