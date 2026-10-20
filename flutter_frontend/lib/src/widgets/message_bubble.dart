import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import '../state/chat_controller.dart';
import '../theme/diya_tokens.dart';
import 'source_sheet.dart';
import 'voice_mark.dart';

/// One chat turn: user bubble or Diya's markdown answer with actions.
class MessageBubble extends StatefulWidget {
  const MessageBubble({
    required this.message,
    required this.speakId,
    this.ttsReady = false,
    this.speaking = false,
    this.onSpeak,
    super.key,
  });

  final ChatMessage message;
  final String speakId;
  final bool ttsReady;
  final bool speaking;
  final VoidCallback? onSpeak;

  @override
  State<MessageBubble> createState() => _MessageBubbleState();
}

class _MessageBubbleState extends State<MessageBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _cursor = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 900),
  )..repeat(reverse: true);
  bool _copied = false;

  @override
  void dispose() {
    _cursor.dispose();
    super.dispose();
  }

  Future<void> _copy() async {
    await Clipboard.setData(ClipboardData(text: widget.message.content));
    setState(() => _copied = true);
    await Future<void>.delayed(const Duration(milliseconds: 1600));
    if (mounted) setState(() => _copied = false);
  }

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    final message = widget.message;
    if (message.isUser) return _UserBubble(message: message);

    return Padding(
      padding: const EdgeInsets.only(bottom: DiyaSpacing.s5 + 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const VoiceMark(size: 30),
          const SizedBox(width: DiyaSpacing.s3),
          Flexible(
            child: ConstrainedBox(
              constraints: const BoxConstraints(
                maxWidth: DiyaSpacing.contentMax - 60,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.only(left: 2, bottom: 6),
                    child: Text(
                      'Diya',
                      style: TextStyle(color: c.faint, fontSize: 11),
                    ),
                  ),
                  _MarkdownAnswer(message: message, cursor: _cursor),
                  if (message.sources.isNotEmpty) ...[
                    const SizedBox(height: DiyaSpacing.s2 + 2),
                    SourceChips(sources: message.sources),
                  ],
                  _ActionRow(
                    copied: _copied,
                    speaking: widget.speaking,
                    canSpeak: widget.ttsReady,
                    timing: message.timing,
                    streaming: message.streaming,
                    onCopy: _copy,
                    onSpeak: widget.onSpeak,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _UserBubble extends StatelessWidget {
  const _UserBubble({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: DiyaSpacing.s5 + 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          Flexible(
            child: ConstrainedBox(
              constraints: const BoxConstraints(
                maxWidth: DiyaSpacing.contentMax - 60,
              ),
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: DiyaSpacing.s4,
                  vertical: DiyaSpacing.s3,
                ),
                decoration: BoxDecoration(
                  color: c.bubbleUser,
                  borderRadius: BorderRadius.circular(DiyaSpacing.rMd)
                      .copyWith(topRight: const Radius.circular(5)),
                ),
                child: SelectableText(
                  message.content,
                  style: TextStyle(
                    color: c.text,
                    fontSize: 16,
                    height: 1.55,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MarkdownAnswer extends StatelessWidget {
  const _MarkdownAnswer({required this.message, required this.cursor});

  final ChatMessage message;
  final AnimationController cursor;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    final style = MarkdownStyleSheet(
      p: TextStyle(color: c.text, fontSize: 16, height: 1.7),
      h1: TextStyle(
        color: c.text,
        fontSize: 23,
        fontWeight: FontWeight.w700,
      ),
      h2: TextStyle(
        color: c.text,
        fontSize: 19,
        fontWeight: FontWeight.w700,
      ),
      h3: TextStyle(
        color: c.text,
        fontSize: 16,
        fontWeight: FontWeight.w700,
      ),
      listBullet: TextStyle(color: c.teal, fontSize: 16),
      a: TextStyle(color: c.teal, decoration: TextDecoration.none),
      code: TextStyle(
        color: c.accent,
        backgroundColor: c.codeBg,
        fontFamily: 'Consolas',
        fontSize: 12.5,
      ),
      codeblockDecoration: BoxDecoration(
        color: c.codeBg,
        borderRadius: BorderRadius.circular(DiyaSpacing.rSm),
        border: Border.all(color: c.border),
      ),
      blockquoteDecoration: BoxDecoration(
        color: c.surfaceHigh,
        border: Border(left: BorderSide(color: c.teal, width: 3)),
      ),
      blockquote: TextStyle(color: c.muted, fontSize: 16, height: 1.6),
      tableBorder: TableBorder.all(color: c.border),
      tableBody: TextStyle(color: c.text, fontSize: 14),
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        MarkdownBody(data: message.content, selectable: true, styleSheet: style),
        if (message.streaming)
          AnimatedBuilder(
            animation: cursor,
            builder: (context, child) => Container(
              margin: const EdgeInsets.only(top: 4),
              width: 8,
              height: 15,
              color: c.teal.withValues(alpha: .35 + .55 * cursor.value),
            ),
          ),
      ],
    );
  }
}

class _ActionRow extends StatelessWidget {
  const _ActionRow({
    required this.copied,
    required this.speaking,
    required this.canSpeak,
    required this.streaming,
    required this.timing,
    required this.onCopy,
    required this.onSpeak,
  });

  final bool copied;
  final bool speaking;
  final bool canSpeak;
  final bool streaming;
  final Map<String, dynamic> timing;
  final VoidCallback onCopy;
  final VoidCallback? onSpeak;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    final tokens = timing['tokens'] as num?;
    final seconds = timing['generate_ms'] != null
        ? ((timing['generate_ms'] as num) / 1000).toStringAsFixed(1)
        : null;
    final meta = [
      if (seconds != null && !streaming) '${seconds}s',
      if (tokens != null && !streaming) '$tokens tokens',
    ].join(' · ');
    return Padding(
      padding: const EdgeInsets.only(top: DiyaSpacing.s2),
      child: Row(
        children: [
          _ActionIcon(
            tooltip: copied ? 'Copied' : 'Copy answer',
            icon: copied ? Icons.check_rounded : Icons.copy_rounded,
            color: copied ? c.success : c.faint,
            onTap: onCopy,
          ),
          if (canSpeak && !streaming)
            _ActionIcon(
              tooltip: speaking ? 'Stop speaking' : 'Speak aloud',
              icon: speaking
                  ? Icons.stop_rounded
                  : Icons.volume_up_rounded,
              color: speaking ? c.accent : c.faint,
              onTap: onSpeak,
            ),
          if (meta.isNotEmpty) ...[
            const SizedBox(width: DiyaSpacing.s2),
            Text(meta, style: TextStyle(color: c.faint, fontSize: 10.5)),
          ],
        ],
      ),
    );
  }
}

class _ActionIcon extends StatelessWidget {
  const _ActionIcon({
    required this.tooltip,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  final String tooltip;
  final IconData icon;
  final Color color;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => IconButton(
    tooltip: tooltip,
    onPressed: onTap,
    visualDensity: VisualDensity.compact,
    icon: Icon(icon, size: 16, color: color),
  );
}
