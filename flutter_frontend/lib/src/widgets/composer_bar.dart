import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/diya_tokens.dart';

/// The message composer: autogrow input, attach, mic, deep-think, send/stop.
class ComposerBar extends StatefulWidget {
  const ComposerBar({
    required this.controller,
    required this.focusNode,
    required this.busy,
    required this.recording,
    required this.recordingSeconds,
    required this.deepThink,
    required this.sttReady,
    required this.onSend,
    required this.onStop,
    required this.onAttach,
    required this.onMicPressed,
    required this.onDeepThinkChanged,
    super.key,
  });

  final TextEditingController controller;
  final FocusNode focusNode;
  final bool busy;
  final bool recording;
  final int recordingSeconds;
  final bool deepThink;
  final bool sttReady;
  final VoidCallback onSend;
  final VoidCallback onStop;
  final VoidCallback onAttach;
  final Future<String?> Function() onMicPressed;
  final ValueChanged<bool> onDeepThinkChanged;

  @override
  State<ComposerBar> createState() => _ComposerBarState();
}

class _ComposerBarState extends State<ComposerBar> {
  KeyEventResult _onKey(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent ||
        event.logicalKey != LogicalKeyboardKey.enter) {
      return KeyEventResult.ignored;
    }
    final mobile =
        defaultTargetPlatform == TargetPlatform.android ||
        defaultTargetPlatform == TargetPlatform.iOS;
    if (mobile || HardwareKeyboard.instance.isShiftPressed) {
      return KeyEventResult.ignored;
    }
    widget.onSend();
    return KeyEventResult.handled;
  }

  Future<void> _mic() async {
    final text = await widget.onMicPressed();
    if (text == null || text.isEmpty) return;
    final field = widget.controller;
    final base = field.text;
    final insert = base.trim().isEmpty ? text : '$base $text';
    field.value = TextEditingValue(
      text: insert,
      selection: TextSelection.collapsed(offset: insert.length),
    );
    widget.focusNode.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Padding(
      padding: EdgeInsets.fromLTRB(
        DiyaSpacing.s4,
        DiyaSpacing.s2,
        DiyaSpacing.s4,
        DiyaSpacing.s3 + MediaQuery.paddingOf(context).bottom,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            decoration: BoxDecoration(
              color: c.surface,
              border: Border.all(color: c.border),
              borderRadius: BorderRadius.circular(DiyaSpacing.rLg),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(
                    DiyaSpacing.s4,
                    DiyaSpacing.s2,
                    DiyaSpacing.s4,
                    0,
                  ),
                  child: Focus(
                    onKeyEvent: _onKey,
                    child: TextField(
                      controller: widget.controller,
                      focusNode: widget.focusNode,
                      minLines: 1,
                      maxLines: 6,
                      maxLength: 8000,
                      textInputAction: TextInputAction.newline,
                      decoration: const InputDecoration(
                        hintText: 'Ask Diya anything…',
                        counterText: '',
                        border: InputBorder.none,
                        enabledBorder: InputBorder.none,
                        focusedBorder: InputBorder.none,
                        filled: false,
                        contentPadding: EdgeInsets.symmetric(
                          vertical: DiyaSpacing.s2 + 2,
                        ),
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(
                    DiyaSpacing.s2,
                    0,
                    DiyaSpacing.s2,
                    DiyaSpacing.s2,
                  ),
                  child: Row(
                    children: [
                      IconButton(
                        tooltip: 'Read a file',
                        onPressed: widget.busy ? null : widget.onAttach,
                        icon: Icon(
                          Icons.add_circle_outline_rounded,
                          size: 20,
                          color: widget.busy ? c.faint : c.muted,
                        ),
                      ),
                      if (widget.sttReady)
                        _MicButton(
                          recording: widget.recording,
                          seconds: widget.recordingSeconds,
                          onPressed: widget.busy ? null : _mic,
                        ),
                      const Spacer(),
                      _ThinkToggle(
                        active: widget.deepThink,
                        onChanged: widget.busy
                            ? null
                            : widget.onDeepThinkChanged,
                      ),
                      const SizedBox(width: DiyaSpacing.s2),
                      _SendButton(
                        busy: widget.busy,
                        field: widget.controller,
                        onSend: widget.onSend,
                        onStop: widget.onStop,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: DiyaSpacing.s2),
          Text(
            'Diya can make mistakes. Check important details.',
            style: TextStyle(color: c.faint, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

class _MicButton extends StatelessWidget {
  const _MicButton({
    required this.recording,
    required this.seconds,
    required this.onPressed,
  });

  final bool recording;
  final int seconds;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    if (!recording) {
      return IconButton(
        tooltip: 'Speak your question',
        onPressed: onPressed,
        icon: Icon(Icons.mic_none_rounded, size: 20, color: c.muted),
      );
    }
    return TextButton.icon(
      onPressed: onPressed,
      icon: TweenAnimationBuilder(
        tween: Tween(begin: 0.6, end: 1.0),
        duration: const Duration(milliseconds: 700),
        builder: (context, double t, child) => Icon(
          Icons.fiber_manual_record_rounded,
          size: 14,
          color: c.danger.withValues(alpha: t),
        ),
      ),
      label: Text(
        '${seconds}s · tap to stop',
        style: TextStyle(color: c.danger, fontSize: 13),
      ),
    );
  }
}

class _ThinkToggle extends StatelessWidget {
  const _ThinkToggle({required this.active, required this.onChanged});

  final bool active;
  final ValueChanged<bool>? onChanged;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Semantics(
      button: true,
      toggled: active,
      label: 'Think mode',
      child: Material(
        color: active ? c.accentSoft : Colors.transparent,
        shape: StadiumBorder(
          side: BorderSide(color: active ? c.accent : c.border),
        ),
        child: InkWell(
          customBorder: const StadiumBorder(),
          onTap: onChanged == null ? null : () => onChanged!(!active),
          child: Padding(
            padding: const EdgeInsets.symmetric(
              horizontal: DiyaSpacing.s3,
              vertical: 6,
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.auto_awesome_outlined,
                  size: 13,
                  color: active ? c.accent : c.faint,
                ),
                const SizedBox(width: 5),
                Text(
                  'Think',
                  style: TextStyle(
                    color: active ? c.accent : c.muted,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _SendButton extends StatelessWidget {
  const _SendButton({
    required this.busy,
    required this.field,
    required this.onSend,
    required this.onStop,
  });

  final bool busy;
  final TextEditingController field;
  final VoidCallback onSend;
  final VoidCallback onStop;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    if (busy) {
      return IconButton.filled(
        tooltip: 'Stop answering',
        onPressed: onStop,
        style: IconButton.styleFrom(backgroundColor: c.danger),
        icon: const Icon(Icons.stop_rounded, size: 18),
      );
    }
    return ValueListenableBuilder(
      valueListenable: field,
      builder: (context, value, child) {
        final hasText = value.text.trim().isNotEmpty;
        return IconButton.filled(
          tooltip: 'Send',
          onPressed: hasText ? onSend : null,
          style: IconButton.styleFrom(
            backgroundColor: c.teal,
            disabledBackgroundColor: c.border.withValues(alpha: .4),
          ),
          icon: Icon(
            Icons.arrow_upward_rounded,
            size: 18,
            color: hasText ? Colors.white : c.faint,
          ),
        );
      },
    );
  }
}
