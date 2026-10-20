import 'package:flutter/material.dart';

import '../theme/diya_tokens.dart';

/// The Diya lamp mark. Set [glowing] for the warm breathing-glow variant
/// used on the welcome screen and while Diya is thinking.
class VoiceMark extends StatelessWidget {
  const VoiceMark({required this.size, this.glowing = false, super.key});

  final double size;
  final bool glowing;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return glowing
        ? TweenAnimationBuilder(
            tween: Tween(begin: 0.0, end: 1.0),
            duration: const Duration(milliseconds: 1800),
            builder: (context, double t, child) {
              final pulse = 0.5 + 0.5 * (1 - (2 * t - 1).abs());
              return Container(
                width: size,
                height: size,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(size * .3),
                  boxShadow: [
                    BoxShadow(
                      color: c.accent.withValues(alpha: .25 + .2 * pulse),
                      blurRadius: 18 + 14 * pulse,
                      spreadRadius: 1,
                    ),
                  ],
                ),
                child: child,
              );
            },
            child: _lamp(c),
          )
        : _lamp(c);
  }

  Widget _lamp(DiyaColors c) => Container(
    width: size,
    height: size,
    decoration: BoxDecoration(
      color: c.surfaceHigh,
      borderRadius: BorderRadius.circular(size * .3),
    ),
    child: CustomPaint(painter: _DiyaMarkPainter(colors: c)),
  );
}

class _DiyaMarkPainter extends CustomPainter {
  const _DiyaMarkPainter({required this.colors});

  final DiyaColors colors;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.scale(size.width / 48, size.height / 48);

    final flamePath = Path()
      ..moveTo(24, 7)
      ..cubicTo(29, 13, 32, 18, 31, 23)
      ..cubicTo(30, 28, 27, 31, 23, 31)
      ..cubicTo(18, 31, 15, 27, 16, 23)
      ..cubicTo(17, 18, 21, 15, 20, 11)
      ..cubicTo(23, 13, 25, 17, 25, 19)
      ..cubicTo(28, 15, 27, 11, 24, 7)
      ..close();
    canvas.drawPath(flamePath, Paint()..color = colors.accent);

    final innerFlame = Path()
      ..moveTo(23, 20)
      ..cubicTo(20, 24, 21, 28, 24, 28)
      ..cubicTo(27, 28, 28, 24, 24, 20)
      ..close();
    canvas.drawPath(innerFlame, Paint()..color = colors.accentBright);

    final bowlPath = Path()
      ..moveTo(9, 31)
      ..cubicTo(15, 35, 33, 35, 39, 31)
      ..cubicTo(37, 39, 32, 43, 24, 43)
      ..cubicTo(16, 43, 11, 39, 9, 31)
      ..close();
    canvas.drawPath(bowlPath, Paint()..color = colors.teal);

    canvas.drawLine(
      const Offset(11, 31.5),
      const Offset(37, 31.5),
      Paint()
        ..color = colors.surfaceHigh.withValues(alpha: .75)
        ..strokeWidth = 1.5
        ..strokeCap = StrokeCap.round,
    );
    canvas.restore();
  }

  @override
  bool shouldRepaint(_DiyaMarkPainter oldDelegate) =>
      colors != oldDelegate.colors;
}
