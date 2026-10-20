import 'package:flutter/material.dart';

import '../theme/diya_tokens.dart';

/// Status bar above the conversation: model, knowledge size, voice state,
/// save state, theme toggle, and owner status entry.
class ChatHeader extends StatelessWidget {
  const ChatHeader({
    required this.model,
    required this.indexedChunks,
    required this.mongoReady,
    required this.ttsReady,
    required this.sttReady,
    required this.themeModeIcon,
    required this.onToggleTheme,
    required this.onShowAdmin,
    this.compact = false,
    super.key,
  });

  final String model;
  final int indexedChunks;
  final bool mongoReady;
  final bool ttsReady;
  final bool sttReady;
  final IconData themeModeIcon;
  final VoidCallback onToggleTheme;
  final VoidCallback onShowAdmin;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Container(
      height: compact ? 56 : 62,
      padding: const EdgeInsets.symmetric(horizontal: DiyaSpacing.s5),
      decoration: BoxDecoration(
        color: c.surface,
        border: Border(bottom: BorderSide(color: c.border)),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final tight = constraints.maxWidth < 520;
          return Row(
            children: [
              Flexible(
                child: Text(
                  model.isEmpty ? 'Local assistant' : model,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: c.muted,
                    fontSize: 14,
                  ),
                ),
              ),
              if (!compact) ...[
                const SizedBox(width: DiyaSpacing.s3),
                Container(width: 1, height: 18, color: c.border),
                const SizedBox(width: DiyaSpacing.s3),
                Text(
                  '$indexedChunks knowledge pieces',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: c.muted,
                    fontSize: 14,
                  ),
                ),
              ],
              const Spacer(),
              if (!tight) ...[
                _VoicePill(
                  icon: Icons.volume_up_rounded,
                  ready: ttsReady,
                  tooltip: ttsReady ? 'Voice output ready (Piper)' : 'Voice output unavailable',
                ),
                const SizedBox(width: DiyaSpacing.s2),
                _VoicePill(
                  icon: Icons.mic_none_rounded,
                  ready: sttReady,
                  tooltip: sttReady ? 'Voice input ready (Vosk)' : 'Voice input unavailable',
                ),
                const SizedBox(width: DiyaSpacing.s3),
              ],
              if (!tight) _SavePill(ready: mongoReady),
              IconButton(
                tooltip: 'Toggle light / dark',
                onPressed: onToggleTheme,
                icon: Icon(themeModeIcon, size: 18),
              ),
              IconButton(
                tooltip: 'Owner status',
                onPressed: onShowAdmin,
                icon: const Icon(Icons.tune_rounded, size: 18),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _VoicePill extends StatelessWidget {
  const _VoicePill({
    required this.icon,
    required this.ready,
    required this.tooltip,
  });

  final IconData icon;
  final bool ready;
  final String tooltip;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Tooltip(
      message: tooltip,
      child: Container(
        width: 26,
        height: 26,
        decoration: BoxDecoration(
          color: ready ? c.tealSoft : c.surfaceHigh,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(
          icon,
          size: 13,
          color: ready ? c.teal : c.faint,
        ),
      ),
    );
  }
}

class _SavePill extends StatelessWidget {
  const _SavePill({required this.ready});

  final bool ready;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: DiyaSpacing.s2 + 2,
        vertical: 6,
      ),
      decoration: BoxDecoration(
        color: ready ? c.tealSoft : c.accentSoft,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              color: ready ? c.success : c.warning,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            ready ? 'Saved' : 'Not saved',
            style: TextStyle(
              color: ready ? c.teal : c.accent,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
