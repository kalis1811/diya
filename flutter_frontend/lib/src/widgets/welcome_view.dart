import 'package:flutter/material.dart';

import '../theme/diya_tokens.dart';
import 'thinking_indicator.dart';

/// Empty-chat hero: glowing lamp, headline, and suggestion cards.
class WelcomeView extends StatelessWidget {
  const WelcomeView({
    required this.onPrompt,
    required this.onAttach,
    required this.voiceReady,
    super.key,
  });

  final ValueChanged<String> onPrompt;
  final VoidCallback onAttach;
  final bool voiceReady;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return LayoutBuilder(
      builder: (context, bounds) {
        return SingleChildScrollView(
          padding: const EdgeInsets.symmetric(
            horizontal: DiyaSpacing.s6,
            vertical: DiyaSpacing.s5,
          ),
          child: ConstrainedBox(
            constraints: BoxConstraints(minHeight: bounds.maxHeight - 48),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 700),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const VoiceMarkGlow(size: 52),
                    const SizedBox(height: DiyaSpacing.s4),
                    Text(
                      'YOUR PRIVATE ASSISTANT',
                      style: theme.textTheme.labelSmall?.copyWith(
                        color: DiyaTokens.of(context).teal,
                        letterSpacing: 1.4,
                      ),
                    ),
                    const SizedBox(height: DiyaSpacing.s2),
                    Text(
                      "What's on your mind?",
                      style: theme.textTheme.displaySmall,
                    ),
                    const SizedBox(height: DiyaSpacing.s3),
                    Text(
                      'Ask a question, think something through, or bring in a '
                      'document. Diya answers from your own files, right here '
                      'on this computer.',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: DiyaTokens.of(context).muted,
                      ),
                    ),
                    const SizedBox(height: DiyaSpacing.s5),
                    LayoutBuilder(
                      builder: (context, suggestionsBounds) {
                        final wide = suggestionsBounds.maxWidth >= 620;
                        final width = wide
                            ? (suggestionsBounds.maxWidth - DiyaSpacing.s2 * 2) / 3
                            : suggestionsBounds.maxWidth;
                        return Wrap(
                          spacing: DiyaSpacing.s2,
                          runSpacing: DiyaSpacing.s2,
                          children: [
                            SizedBox(
                              width: width,
                              child: _Suggestion(
                                icon: Icons.calendar_month_outlined,
                                title: 'Plan my study time',
                                subtitle: 'A plan that fits my day',
                                onTap: () => onPrompt(
                                  'Help me make a simple study plan.',
                                ),
                              ),
                            ),
                            SizedBox(
                              width: width,
                              child: _Suggestion(
                                icon: Icons.auto_awesome_outlined,
                                title: 'Explain a topic',
                                subtitle: 'Make a tricky idea feel simple',
                                onTap: () => onPrompt(
                                  'Explain photosynthesis in simple words.',
                                ),
                              ),
                            ),
                            SizedBox(
                              width: width,
                              child: _Suggestion(
                                icon: Icons.description_outlined,
                                title: 'Read a document',
                                subtitle: 'Ask about a PDF or image',
                                onTap: onAttach,
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                    if (voiceReady) ...[
                      const SizedBox(height: DiyaSpacing.s4),
                      Row(
                        children: [
                          Icon(
                            Icons.mic_none_rounded,
                            size: 14,
                            color: DiyaTokens.of(context).faint,
                          ),
                          const SizedBox(width: DiyaSpacing.s2),
                          Text(
                            'Voice is ready — tap the mic in the composer to talk to Diya.',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: DiyaTokens.of(context).faint,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _Suggestion extends StatelessWidget {
  const _Suggestion({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Material(
      color: c.surface,
      borderRadius: BorderRadius.circular(DiyaSpacing.rMd),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(DiyaSpacing.rMd),
        child: Container(
          constraints: const BoxConstraints(minHeight: 72),
          padding: const EdgeInsets.symmetric(
            horizontal: DiyaSpacing.s3,
            vertical: DiyaSpacing.s3,
          ),
          decoration: BoxDecoration(
            border: Border.all(color: c.border),
            borderRadius: BorderRadius.circular(DiyaSpacing.rMd),
          ),
          child: Row(
            children: [
              Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  color: c.tealSoft,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: c.teal, size: 18),
              ),
              const SizedBox(width: DiyaSpacing.s3),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      title,
                      style: TextStyle(
                        color: c.text,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      subtitle,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(color: c.muted, fontSize: 12),
                    ),
                  ],
                ),
              ),
              Icon(
                Icons.arrow_outward_rounded,
                color: c.faint,
                size: 14,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
