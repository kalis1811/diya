import 'package:flutter/material.dart';

import '../../api_client.dart';
import '../theme/diya_tokens.dart';

/// Owner dashboard: model, knowledge, memories, storage, voice status.
class AdminDialog extends StatefulWidget {
  const AdminDialog({required this.api, super.key});

  final DiyaApiClient api;

  @override
  State<AdminDialog> createState() => _AdminDialogState();
}

class _AdminDialogState extends State<AdminDialog> {
  Map<String, dynamic>? _data;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final result = await widget.api.get('/admin/overview');
      if (mounted) setState(() => _data = result);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    final screenWidth = MediaQuery.sizeOf(context).width;
    return AlertDialog(
      title: Text('Diya owner status', style: Theme.of(context).textTheme.titleMedium),
      content: SizedBox(
        width: screenWidth < 560 ? screenWidth - 80 : 480,
        child: _data == null && _error == null
            ? const Padding(
                padding: EdgeInsets.all(DiyaSpacing.s5),
                child: Center(child: CircularProgressIndicator()),
              )
            : _error != null
            ? Text(_error!, style: TextStyle(color: c.danger))
            : Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Signed in as ${_data!['owner'] ?? 'owner'}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                  const SizedBox(height: DiyaSpacing.s4),
                  Wrap(
                    spacing: DiyaSpacing.s2,
                    runSpacing: DiyaSpacing.s2,
                    children: [
                      _StatusCard(label: 'Model', value: _data!['model']),
                      _StatusCard(
                        label: 'Knowledge',
                        value: '${_data!['indexed_chunks'] ?? 0} pieces',
                      ),
                      _StatusCard(
                        label: 'Memories',
                        value: '${_data!['memories'] ?? 0} saved',
                      ),
                      _StatusCard(
                        label: 'Conversations',
                        value: '${_data!['conversations'] ?? 0}',
                      ),
                      _StatusCard(label: 'Storage', value: _data!['storage']),
                      _StatusCard(
                        label: 'Voice output',
                        value: _data!['voice_output'],
                      ),
                      _StatusCard(
                        label: 'Voice input',
                        value: _data!['voice_input'],
                      ),
                    ],
                  ),
                ],
              ),
      ),
      actions: [
        TextButton(onPressed: _load, child: const Text('Refresh status')),
        FilledButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Done'),
        ),
      ],
    );
  }
}

class _StatusCard extends StatelessWidget {
  const _StatusCard({required this.label, required this.value});

  final String label;
  final dynamic value;

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return SizedBox(
      width: 205,
      child: Container(
        padding: const EdgeInsets.all(DiyaSpacing.s3),
      decoration: BoxDecoration(
          color: c.surfaceHigh,
        border: Border.all(color: c.border),
        borderRadius: BorderRadius.circular(DiyaSpacing.rSm + 1),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                fontSize: 11,
              ),
            ),
            const SizedBox(height: 5),
            Text(
              value?.toString() ?? 'Unknown',
              style: TextStyle(
                color: c.text,
                fontWeight: FontWeight.w600,
                fontSize: 13,
              ),
              overflow: TextOverflow.ellipsis,
              maxLines: 2,
            ),
          ],
        ),
      ),
    );
  }
}
