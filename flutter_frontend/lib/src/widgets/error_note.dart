import 'package:flutter/material.dart';

import '../theme/diya_tokens.dart';

/// Inline error banner with optional dismiss.
class ErrorNote extends StatelessWidget {
  const ErrorNote({super.key, required this.message, this.onDismiss});

  final String message;
  final VoidCallback? onDismiss;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: DiyaSpacing.s3,
        vertical: DiyaSpacing.s2 + 2,
      ),
      decoration: BoxDecoration(
        color: c.dangerBg,
        border: Border.all(color: c.danger.withValues(alpha: .35)),
        borderRadius: BorderRadius.circular(DiyaSpacing.rSm + 1),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline_rounded, color: c.danger, size: 18),
          const SizedBox(width: DiyaSpacing.s2 + 1),
          Expanded(
            child: Text(
              message,
              style: TextStyle(
                color: c.danger,
                fontSize: 12,
                height: 1.4,
              ),
            ),
          ),
          if (onDismiss != null)
            IconButton(
              visualDensity: VisualDensity.compact,
              onPressed: onDismiss,
              icon: Icon(Icons.close_rounded, size: 16, color: c.danger),
            ),
        ],
      ),
    );
  }
}
