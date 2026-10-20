import 'package:flutter/material.dart';

import 'diya_tokens.dart';

/// Builds the full ThemeData for one brightness from the Diya tokens.
ThemeData buildDiyaTheme(Brightness brightness) {
  final c =
      brightness == Brightness.dark ? DiyaColors.dark : DiyaColors.light;

  final textTheme = TextTheme(
    displaySmall: TextStyle(
      color: c.text,
      fontSize: 38,
      height: 1.12,
      fontWeight: FontWeight.w700,
      letterSpacing: -1.1,
    ),
    headlineMedium: TextStyle(
      color: c.text,
      fontSize: 25,
      height: 1.25,
      fontWeight: FontWeight.w700,
      letterSpacing: -0.4,
    ),
    titleMedium: TextStyle(
      color: c.text,
      fontSize: 16,
      height: 1.3,
      fontWeight: FontWeight.w600,
    ),
    bodyMedium: TextStyle(color: c.text, fontSize: 15, height: 1.65),
    bodySmall: TextStyle(color: c.muted, fontSize: 14, height: 1.5),
    labelLarge: TextStyle(
      color: c.text,
      fontSize: 14,
      fontWeight: FontWeight.w600,
    ),
    labelSmall: TextStyle(
      color: c.faint,
      fontSize: 12,
      fontWeight: FontWeight.w600,
      letterSpacing: 1.1,
    ),
  );

  final border = OutlineInputBorder(
    borderRadius: BorderRadius.circular(DiyaSpacing.rMd),
    borderSide: BorderSide(color: c.border),
  );

  return ThemeData(
    useMaterial3: true,
    fontFamily: 'Segoe UI',
    fontFamilyFallback: const ['Noto Sans', 'Arial'],
    brightness: brightness,
    scaffoldBackgroundColor: c.canvas,
    colorScheme: ColorScheme(
      brightness: brightness,
      primary: c.teal,
      onPrimary: Colors.white,
      secondary: c.accent,
      onSecondary: c.onAccent,
      surface: c.surface,
      onSurface: c.text,
      surfaceContainerHighest: c.surfaceHigh,
      outline: c.border,
      error: c.danger,
      onError: Colors.white,
    ),
    textTheme: textTheme,
    dividerTheme: DividerThemeData(color: c.border, thickness: 1, space: 1),
    appBarTheme: AppBarTheme(
      backgroundColor: c.surface,
      foregroundColor: c.text,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      titleTextStyle: textTheme.titleMedium,
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: c.teal,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(DiyaSpacing.rSm),
        ),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: c.surface,
      hintStyle: TextStyle(color: c.faint),
      contentPadding: const EdgeInsets.symmetric(
        horizontal: DiyaSpacing.s4,
        vertical: DiyaSpacing.s3,
      ),
      border: border,
      enabledBorder: border,
      focusedBorder: border.copyWith(
        borderSide: BorderSide(color: c.teal, width: 1.5),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size.fromHeight(46),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(DiyaSpacing.rMd),
        ),
        textStyle: textTheme.labelLarge,
      ),
    ),
    iconButtonTheme: IconButtonThemeData(
      style: IconButton.styleFrom(foregroundColor: c.muted),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: c.surface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(DiyaSpacing.rLg),
      ),
    ),
    progressIndicatorTheme: ProgressIndicatorThemeData(
      color: c.teal,
      linearTrackColor: c.surfaceHigh,
      circularTrackColor: c.surfaceHigh,
    ),
    snackBarTheme: SnackBarThemeData(
      backgroundColor: c.surfaceHigh,
      contentTextStyle: textTheme.bodyMedium,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(DiyaSpacing.rMd),
      ),
    ),
    tooltipTheme: TooltipThemeData(
      decoration: BoxDecoration(
        color: c.surfaceHigh,
        borderRadius: BorderRadius.circular(DiyaSpacing.rSm),
        border: Border.all(color: c.border),
      ),
      textStyle: textTheme.bodySmall,
    ),
    popupMenuTheme: PopupMenuThemeData(
      color: c.surfaceHigh,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(DiyaSpacing.rMd),
      ),
    ),
    bottomSheetTheme: BottomSheetThemeData(
      backgroundColor: c.surface,
      surfaceTintColor: Colors.transparent,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
    ),
  );
}
