import 'package:flutter/material.dart';

import '../../api_client.dart';
import '../theme/diya_tokens.dart';
import 'error_note.dart';

/// Bottom sheet for reviewing and curating what Diya remembers.
class MemorySheet extends StatefulWidget {
  const MemorySheet({required this.api, super.key});

  final DiyaApiClient api;

  @override
  State<MemorySheet> createState() => _MemorySheetState();
}

class _MemorySheetState extends State<MemorySheet> {
  final _text = TextEditingController();
  List<Map<String, dynamic>> _memories = [];
  bool _loading = true;
  bool _saving = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    try {
      final result = await widget.api.get('/memory');
      if (mounted) {
        setState(() {
          _memories = (result['memories'] as List? ?? [])
              .cast<Map<String, dynamic>>();
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = e.toString();
        });
      }
    }
  }

  Future<void> _save() async {
    final value = _text.text.trim();
    if (value.isEmpty || _saving) return;
    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      await widget.api.post('/memory', {'text': value, 'kind': 'fact'});
      _text.clear();
      await _load();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _forget(String id) async {
    try {
      await widget.api.delete('/memory/${Uri.encodeComponent(id)}');
      await _load();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return SafeArea(
      child: DraggableScrollableSheet(
        initialChildSize: .75,
        minChildSize: .45,
        maxChildSize: .94,
        expand: false,
        builder: (context, scrollController) => Column(
          children: [
            const SizedBox(height: DiyaSpacing.s2),
            Container(
              width: 38,
              height: 4,
              decoration: BoxDecoration(
                color: c.border,
                borderRadius: BorderRadius.circular(5),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(
                DiyaSpacing.s5,
                DiyaSpacing.s4,
                DiyaSpacing.s4,
                DiyaSpacing.s3,
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Things Diya remembers',
                          style: Theme.of(context).textTheme.headlineMedium,
                        ),
                        const SizedBox(height: DiyaSpacing.s1),
                        Text(
                          'Review or remove facts you asked Diya to save.',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.close_rounded),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: DiyaSpacing.s5),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _text,
                      maxLength: 2000,
                      decoration: const InputDecoration(
                        hintText: 'Add a fact or preference',
                        counterText: '',
                      ),
                    ),
                  ),
                  const SizedBox(width: DiyaSpacing.s2),
                  IconButton.filled(
                    onPressed: _saving ? null : _save,
                    tooltip: 'Save memory',
                    icon: _saving
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.add_rounded),
                  ),
                ],
              ),
            ),
            if (_error != null)
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: DiyaSpacing.s5,
                ),
                child: ErrorNote(message: _error!),
              ),
            const SizedBox(height: DiyaSpacing.s3),
            Expanded(
              child: _loading
                  ? const Center(child: CircularProgressIndicator())
                  : _memories.isEmpty
                  ? Center(
                      child: Padding(
                        padding: const EdgeInsets.all(DiyaSpacing.s6),
                        child: Text(
                          'Nothing saved yet. Tell Diya "Remember that…" '
                          'or add a fact above.',
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ),
                    )
                  : ListView.separated(
                      controller: scrollController,
                      padding: const EdgeInsets.fromLTRB(
                        DiyaSpacing.s5,
                        0,
                        DiyaSpacing.s5,
                        DiyaSpacing.s6,
                      ),
                      itemCount: _memories.length,
                      separatorBuilder: (_, _) =>
                          Divider(height: 1, color: c.border),
                      itemBuilder: (context, index) {
                        final item = _memories[index];
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 10),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment:
                                      CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      item['text']?.toString() ?? '',
                                      style: TextStyle(
                                        color: c.text,
                                        fontSize: 13,
                                        height: 1.45,
                                      ),
                                    ),
                                    const SizedBox(height: 5),
                                    Text(
                                      item['kind']?.toString() ?? 'fact',
                                      style: TextStyle(
                                        color: c.teal,
                                        fontSize: 11,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              TextButton(
                                onPressed: () => _forget(
                                  item['id']?.toString() ?? '',
                                ),
                                child: Text(
                                  'Forget',
                                  style: TextStyle(color: c.danger),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
