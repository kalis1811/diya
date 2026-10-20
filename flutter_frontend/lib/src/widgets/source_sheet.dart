import 'package:flutter/material.dart';

import '../theme/diya_tokens.dart';

/// Chip row of source citations under an answer.
class SourceChips extends StatelessWidget {
  const SourceChips({required this.sources, super.key});

  final List<Map<String, dynamic>> sources;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return Wrap(
      spacing: DiyaSpacing.s2,
      runSpacing: DiyaSpacing.s2,
      children: sources.map((source) {
        final label = source['file']?.toString() ?? 'Source';
        return ActionChip(
          avatar: Icon(Icons.link_rounded, color: c.teal, size: 13),
          label: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 220),
            child: Text(
              label,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: c.muted, fontSize: 10.5),
            ),
          ),
          backgroundColor: c.surface,
          side: BorderSide(color: c.border),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(999),
          ),
          onPressed: () => showModalBottomSheet<void>(
            context: context,
            showDragHandle: true,
            builder: (context) => SourceSheet(sources: sources),
          ),
        );
      }).toList(),
    );
  }
}

/// Bottom sheet listing full source snippets behind an answer.
class SourceSheet extends StatelessWidget {
  const SourceSheet({required this.sources, super.key});

  final List<Map<String, dynamic>> sources;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return SafeArea(
      child: ListView.separated(
        shrinkWrap: true,
        padding: const EdgeInsets.fromLTRB(
          DiyaSpacing.s5,
          0,
          DiyaSpacing.s5,
          DiyaSpacing.s5,
        ),
        itemCount: sources.length,
        separatorBuilder: (_, _) => Divider(color: c.border, height: DiyaSpacing.s5),
        itemBuilder: (context, index) {
          final source = sources[index];
          final score = (source['score'] as num?)?.toDouble();
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(Icons.description_outlined, color: c.teal, size: 15),
                  const SizedBox(width: DiyaSpacing.s2),
                  Expanded(
                    child: Text(
                      source['file']?.toString() ?? 'Source',
                      style: TextStyle(
                        color: c.text,
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                  ),
                  if (score != null)
                    Text(
                      '${(score * 100).round()}% match',
                      style: TextStyle(color: c.faint, fontSize: 11),
                    ),
                ],
              ),
              const SizedBox(height: DiyaSpacing.s2),
              Text(
                source['snippet']?.toString() ?? '',
                style: TextStyle(color: c.muted, fontSize: 12.5, height: 1.55),
              ),
            ],
          );
        },
      ),
    );
  }
}
