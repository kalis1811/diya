import 'package:flutter/material.dart';

import '../theme/diya_tokens.dart';
import 'voice_mark.dart';

/// Brand navigation panel: logo, new chat, memories/admin entries, the
/// conversation history, and the storage status footer.
class AppSidebar extends StatelessWidget {
  const AppSidebar({
    required this.conversations,
    required this.activeId,
    required this.mongoReady,
    required this.onNew,
    required this.onMemories,
    required this.onKnowledge,
    required this.onAdmin,
    required this.onOpen,
    required this.onDelete,
    required this.onDisconnect,
    required this.themeModeIcon,
    required this.onToggleTheme,
    super.key,
  });

  final List<Map<String, dynamic>> conversations;
  final String? activeId;
  final bool mongoReady;
  final VoidCallback onNew;
  final VoidCallback onMemories;
  final VoidCallback onKnowledge;
  final VoidCallback onAdmin;
  final ValueChanged<String> onOpen;
  final ValueChanged<String> onDelete;
  final VoidCallback onDisconnect;
  final IconData themeModeIcon;
  final VoidCallback onToggleTheme;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Container(
      color: c.navBg,
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(
                DiyaSpacing.s5,
                DiyaSpacing.s5,
                DiyaSpacing.s4,
                DiyaSpacing.s6,
              ),
              child: Row(
                children: [
                  const VoiceMark(size: 38),
                  const SizedBox(width: DiyaSpacing.s3),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'diya',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 25,
                          height: 1,
                          fontWeight: FontWeight.w800,
                          letterSpacing: -1.2,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        'your AI assistant',
                        style: TextStyle(
                          color: c.muted,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                  const Spacer(),
                  IconButton(
                    tooltip: 'Toggle light / dark',
                    onPressed: onToggleTheme,
                    icon: Icon(themeModeIcon, color: c.muted, size: 19),
                  ),
                  IconButton(
                    tooltip: 'Connection settings',
                    onPressed: onDisconnect,
                    icon: Icon(
                      Icons.settings_outlined,
                      color: c.muted,
                      size: 19,
                    ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: DiyaSpacing.s4),
              child: FilledButton.tonalIcon(
                onPressed: onNew,
                icon: const Icon(Icons.add_rounded),
                label: const Align(
                  alignment: Alignment.centerLeft,
                  child: Text('New conversation'),
                ),
                style: FilledButton.styleFrom(
                  backgroundColor: c.accent,
                  foregroundColor: c.onAccent,
                  minimumSize: const Size.fromHeight(44),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(DiyaSpacing.rSm),
                  ),
                ),
              ),
            ),
            const _SectionLabel('Your space'),
            _NavTile(
              icon: Icons.bookmark_border_rounded,
              label: 'Things Diya remembers',
              onTap: onMemories,
            ),
            _NavTile(
              icon: Icons.library_books_outlined,
              label: 'Knowledge base',
              onTap: onKnowledge,
            ),
            _NavTile(
              icon: Icons.tune_rounded,
              label: 'Owner status',
              onTap: onAdmin,
            ),
            const _SectionLabel('Recent conversations'),
            if (!mongoReady)
              Padding(
                padding: EdgeInsets.fromLTRB(
                  DiyaSpacing.s5,
                  DiyaSpacing.s1,
                  DiyaSpacing.s4,
                  DiyaSpacing.s2,
                ),
                child: Text(
                  'Start MongoDB to save chats here.',
                  style: TextStyle(
                    color: c.muted,
                    fontSize: 12,
                    height: 1.4,
                  ),
                ),
              ),
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.symmetric(horizontal: DiyaSpacing.s2),
                itemCount: conversations.length,
                itemBuilder: (context, index) {
                  final item = conversations[index];
                  final id = item['id']?.toString() ?? '';
                  final title = item['title']?.toString().trim();
                  final active = id == activeId;
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 1),
                    child: Row(
                      children: [
                        Expanded(
                          child: TextButton.icon(
                            onPressed: () => onOpen(id),
                            icon: const Icon(
                              Icons.chat_bubble_outline_rounded,
                              size: 14,
                            ),
                            label: Align(
                              alignment: Alignment.centerLeft,
                              child: Text(
                                title?.isNotEmpty == true
                                    ? title!
                                    : 'New conversation',
                                overflow: TextOverflow.ellipsis,
                                maxLines: 1,
                              ),
                            ),
                            style: TextButton.styleFrom(
                              foregroundColor: active ? c.text : c.muted,
                              backgroundColor: active
                                  ? c.surfaceHigh
                                  : Colors.transparent,
                              alignment: Alignment.centerLeft,
                              padding: const EdgeInsets.symmetric(
                                horizontal: DiyaSpacing.s3,
                                vertical: DiyaSpacing.s3,
                              ),
                              textStyle: const TextStyle(fontSize: 13),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(DiyaSpacing.rSm),
                              ),
                            ),
                          ),
                        ),
                        PopupMenuButton<String>(
                          tooltip: 'Conversation actions',
                          icon: Icon(
                            Icons.more_horiz_rounded,
                            size: 18,
                            color: c.muted,
                          ),
                          onSelected: (value) {
                            if (value == 'delete') onDelete(id);
                          },
                          itemBuilder: (context) => const [
                            PopupMenuItem(
                              value: 'delete',
                              child: Text('Delete conversation'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
            _StorageFooter(mongoReady: mongoReady),
          ],
        ),
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);

  final String text;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.fromLTRB(
      DiyaSpacing.s5,
      DiyaSpacing.s5,
      DiyaSpacing.s4,
      DiyaSpacing.s2,
    ),
    child: Text(
      text,
      style: TextStyle(
        color: DiyaTokens.of(context).muted,
        fontSize: 12,
        fontWeight: FontWeight.w600,
      ),
    ),
  );
}

class _NavTile extends StatelessWidget {
  const _NavTile({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(horizontal: DiyaSpacing.s4, vertical: 1),
    child: TextButton(
      onPressed: onTap,
      style: TextButton.styleFrom(
        foregroundColor: DiyaTokens.of(context).text,
        alignment: Alignment.centerLeft,
        padding: const EdgeInsets.symmetric(
          horizontal: DiyaSpacing.s3,
          vertical: DiyaSpacing.s3,
        ),
        textStyle: const TextStyle(fontSize: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
      child: Row(
        children: [
          Icon(icon, color: DiyaTokens.of(context).teal, size: 18),
          const SizedBox(width: DiyaSpacing.s3),
          Text(label),
        ],
      ),
    ),
  );
}

class _StorageFooter extends StatelessWidget {
  const _StorageFooter({required this.mongoReady});

  final bool mongoReady;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Container(
      margin: const EdgeInsets.fromLTRB(
        DiyaSpacing.s5,
        DiyaSpacing.s2,
        DiyaSpacing.s5,
        0,
      ),
      padding: const EdgeInsets.symmetric(vertical: DiyaSpacing.s3),
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: c.border)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 8,
            height: 8,
            margin: const EdgeInsets.only(top: 4),
            decoration: BoxDecoration(
              color: mongoReady ? c.success : c.warning,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: DiyaSpacing.s2 + 2),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  mongoReady ? 'Chats saved' : 'History not saved',
                  style: TextStyle(
                    color: c.text,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  mongoReady ? 'MongoDB connected' : 'MongoDB is offline',
                  style: TextStyle(color: c.muted, fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
