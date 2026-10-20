import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme/diya_tokens.dart';
import 'voice_mark.dart';

/// "Diya is thinking" row with a soft three-dot wave and a glowing lamp.
class ThinkingIndicator extends StatefulWidget {
  const ThinkingIndicator({super.key});

  @override
  State<ThinkingIndicator> createState() => _ThinkingIndicatorState();
}

class _ThinkingIndicatorState extends State<ThinkingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1200),
  )..repeat();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: DiyaSpacing.s5),
      child: Row(
        children: [
          const VoiceMarkGlow(size: 30),
          const SizedBox(width: DiyaSpacing.s3),
          AnimatedBuilder(
            animation: _controller,
            builder: (context, child) {
              return Row(
                children: List.generate(3, (index) {
                  final phase = (_controller.value * 2 * 3.14) + index * 1.1;
                  final lift = (math.sin(phase) + 1) / 2;
                  return Container(
                    width: 7,
                    height: 7,
                    margin: const EdgeInsets.only(right: 5),
                    decoration: BoxDecoration(
                      color: c.teal.withValues(alpha: .35 + .55 * lift),
                      shape: BoxShape.circle,
                    ),
                  );
                }),
              );
            },
          ),
          const SizedBox(width: DiyaSpacing.s2),
          Text(
            'Diya is thinking',
            style: TextStyle(color: c.muted, fontSize: 13),
          ),
        ],
      ),
    );
  }
}

/// Small glowing lamp used inside [ThinkingIndicator] and the welcome hero.
class VoiceMarkGlow extends StatelessWidget {
  const VoiceMarkGlow({required this.size, super.key});

  final double size;

  @override
  Widget build(BuildContext context) => _GlowingLamp(size: size);
}

class _GlowingLamp extends StatefulWidget {
  const _GlowingLamp({required this.size});

  final double size;

  @override
  State<_GlowingLamp> createState() => _GlowingLampState();
}

class _GlowingLampState extends State<_GlowingLamp>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1700),
  )..repeat(reverse: true);

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        final t = _controller.value;
        return Container(
          width: widget.size,
          height: widget.size,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(widget.size * .3),
            boxShadow: [
              BoxShadow(
                color: c.accent.withValues(alpha: .18 + .22 * t),
                blurRadius: 10 + 12 * t,
                spreadRadius: .5,
              ),
            ],
          ),
          child: child,
        );
      },
      child: VoiceMark(size: widget.size),
    );
  }
}
