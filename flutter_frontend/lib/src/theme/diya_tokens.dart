import 'package:flutter/material.dart';

/// Design tokens for Diya. Every color a widget needs comes from here via
/// `DiyaTokens.of(context)` — never an inline color literal.
@immutable
class DiyaColors {
  const DiyaColors._({
    required this.canvas,
    required this.surface,
    required this.surfaceHigh,
    required this.border,
    required this.text,
    required this.muted,
    required this.faint,
    required this.accent,
    required this.accentBright,
    required this.onAccent,
    required this.accentSoft,
    required this.teal,
    required this.tealSoft,
    required this.bubbleUser,
    required this.codeBg,
    required this.success,
    required this.warning,
    required this.danger,
    required this.dangerBg,
    required this.navBg,
  });

  final Color canvas;
  final Color surface;
  final Color surfaceHigh;
  final Color border;
  final Color text;
  final Color muted;
  final Color faint;
  final Color accent;
  final Color accentBright;
  final Color onAccent;
  final Color accentSoft;
  final Color teal;
  final Color tealSoft;
  final Color bubbleUser;
  final Color codeBg;
  final Color success;
  final Color warning;
  final Color danger;
  final Color dangerBg;
  final Color navBg;

  /// A quiet, warm night palette with enough separation for long chat sessions.
  static const DiyaColors dark = DiyaColors._(
    canvas: Color(0xFF121714),
    surface: Color(0xFF1A211D),
    surfaceHigh: Color(0xFF242D27),
    border: Color(0xFF303B34),
    text: Color(0xFFF1F0E8),
    muted: Color(0xFFB2B8AE),
    faint: Color(0xFF858F84),
    accent: Color(0xFFE8AA55),
    accentBright: Color(0xFFFFD28A),
    onAccent: Color(0xFF2A1B04),
    accentSoft: Color(0xFF352A1B),
    teal: Color(0xFF83C4A8),
    tealSoft: Color(0xFF20362D),
    bubbleUser: Color(0xFF2B3931),
    codeBg: Color(0xFF151A17),
    success: Color(0xFF83C4A8),
    warning: Color(0xFFE2B966),
    danger: Color(0xFFE88B75),
    dangerBg: Color(0xFF39221D),
    navBg: Color(0xFF171E1A),
  );

  /// Daylight variant of the same identity.
  static const DiyaColors light = DiyaColors._(
    canvas: Color(0xFFF6F4EF),
    surface: Color(0xFFFFFEFB),
    surfaceHigh: Color(0xFFF0EEE7),
    border: Color(0xFFE5E0D5),
    text: Color(0xFF29342E),
    muted: Color(0xFF6E796F),
    faint: Color(0xFF92998F),
    accent: Color(0xFFC9822D),
    accentBright: Color(0xFFE8AA55),
    onAccent: Color(0xFFFFFFFF),
    accentSoft: Color(0xFFFBF1DF),
    teal: Color(0xFF34785C),
    tealSoft: Color(0xFFE4EFE7),
    bubbleUser: Color(0xFFE8EEE6),
    codeBg: Color(0xFFF2F0E9),
    success: Color(0xFF34785C),
    warning: Color(0xFFB98A2E),
    danger: Color(0xFFB3563F),
    dangerBg: Color(0xFFFDF0EC),
    navBg: Color(0xFF263A30),
  );
}

/// Spacing scale (multiples of 4) and corner radii used across the app.
abstract final class DiyaSpacing {
  static const double s1 = 4;
  static const double s2 = 8;
  static const double s3 = 12;
  static const double s4 = 16;
  static const double s5 = 24;
  static const double s6 = 32;
  static const double s7 = 48;

  static const double rSm = 9;
  static const double rMd = 14;
  static const double rLg = 20;

  /// Below this width the app switches to a mobile drawer + compact chat.
  static const double navBreakpoint = 920;

  /// Max content width for chat columns on wide screens.
  static const double contentMax = 780;
}

/// Inherited access point so widgets read `DiyaTokens.of(context).<token>`
/// without threading palette objects through every constructor.
class DiyaTokens extends InheritedWidget {
  const DiyaTokens({
    required this.brightness,
    required super.child,
    super.key,
  });

  final Brightness brightness;

  DiyaColors get colors =>
      brightness == Brightness.dark ? DiyaColors.dark : DiyaColors.light;

  static DiyaColors of(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<DiyaTokens>()!.colors;

  @override
  bool updateShouldNotify(DiyaTokens oldWidget) =>
      brightness != oldWidget.brightness;
}
