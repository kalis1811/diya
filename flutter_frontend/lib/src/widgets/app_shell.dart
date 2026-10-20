import 'package:flutter/material.dart';

import '../state/chat_controller.dart';
import '../state/voice_controller.dart';
import '../theme/diya_tokens.dart';
import 'app_sidebar.dart';
import 'chat_page.dart';
import 'voice_mark.dart';

/// Connected shell: brand sidebar (fixed on desktop, drawer on mobile)
/// next to the chat surface.
class AppShell extends StatelessWidget {
  const AppShell({
    required this.chat,
    required this.voice,
    required this.themeModeIcon,
    required this.onToggleTheme,
    required this.onShowMemories,
    required this.onShowKnowledge,
    required this.onShowAdmin,
    super.key,
  });

  final ChatController chat;
  final VoiceController voice;
  final IconData themeModeIcon;
  final VoidCallback onToggleTheme;
  final VoidCallback onShowMemories;
  final VoidCallback onShowKnowledge;
  final VoidCallback onShowAdmin;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final desktop = constraints.maxWidth >= DiyaSpacing.navBreakpoint;
        final sidebar = AppSidebar(
          conversations: chat.conversations,
          activeId: chat.conversationId,
          mongoReady: chat.mongoReady,
          onNew: chat.newConversation,
          onMemories: onShowMemories,
          onKnowledge: onShowKnowledge,
          onAdmin: onShowAdmin,
          onOpen: chat.openConversation,
          onDelete: chat.deleteConversation,
          onDisconnect: chat.disconnect,
          themeModeIcon: themeModeIcon,
          onToggleTheme: onToggleTheme,
        );
        final chatPage = ChatPage(
          chat: chat,
          voice: voice,
          compact: !desktop,
          themeModeIcon: themeModeIcon,
          onToggleTheme: onToggleTheme,
          onShowAdmin: onShowAdmin,
          onShowMemories: onShowMemories,
        );
        if (desktop) {
          return Row(
            children: [
              SizedBox(width: 276, child: sidebar),
              Expanded(child: chatPage),
            ],
          );
        }
        return Scaffold(
          drawer: Drawer(width: 298, child: sidebar),
          appBar: AppBar(
            titleSpacing: 0,
            title: const Row(
              children: [
                VoiceMark(size: 28),
                SizedBox(width: DiyaSpacing.s2 + 1),
                Text('Diya', style: TextStyle(fontWeight: FontWeight.w700)),
              ],
            ),
          ),
          body: chatPage,
        );
      },
    );
  }
}
