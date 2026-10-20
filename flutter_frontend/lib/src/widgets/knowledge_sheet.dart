import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../../api_client.dart';
import '../theme/diya_tokens.dart';
import 'error_note.dart';

/// Manage local documents that are retained and indexed for future answers.
class KnowledgeSheet extends StatefulWidget {
  const KnowledgeSheet({
    required this.api,
    required this.onChanged,
    super.key,
  });

  final DiyaApiClient api;
  final VoidCallback onChanged;

  @override
  State<KnowledgeSheet> createState() => _KnowledgeSheetState();
}

class _KnowledgeSheetState extends State<KnowledgeSheet> {
  List<Map<String, dynamic>> _files = [];
  bool _loading = true;
  bool _busy = false;
  int _chunks = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final result = await widget.api.listKnowledge();
      if (!mounted) return;
      setState(() {
        _files = (result['files'] as List? ?? [])
            .whereType<Map>()
            .map((file) => Map<String, dynamic>.from(file))
            .toList();
        _chunks = result['indexed_chunks'] as int? ?? 0;
        _loading = false;
        _error = null;
      });
    } catch (e) {
      if (mounted) setState(() { _loading = false; _error = e.toString(); });
    }
  }

  Future<void> _add() async {
    if (_busy) return;
    final picked = await FilePicker.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'txt', 'md', 'csv', 'docx', 'html', 'htm', 'json'],
    );
    if (picked.isEmpty || !mounted) return;
    setState(() { _busy = true; _error = null; });
    try {
      await widget.api.uploadKnowledge(picked.single);
      widget.onChanged();
      await _load();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _remove(String filename) async {
    if (_busy) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Remove this document?'),
        content: Text('“$filename” will be removed from Diya’s local knowledge base.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Remove')),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    setState(() { _busy = true; _error = null; });
    try {
      await widget.api.removeKnowledge(filename);
      widget.onChanged();
      await _load();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = DiyaTokens.of(context);
    return SafeArea(
      child: DraggableScrollableSheet(
        initialChildSize: .78,
        minChildSize: .48,
        maxChildSize: .94,
        expand: false,
        builder: (context, scrollController) => Column(
          children: [
            const SizedBox(height: DiyaSpacing.s2),
            Container(width: 38, height: 4,
              decoration: BoxDecoration(color: c.border, borderRadius: BorderRadius.circular(5))),
            Padding(
              padding: const EdgeInsets.fromLTRB(DiyaSpacing.s5, DiyaSpacing.s4, DiyaSpacing.s4, DiyaSpacing.s3),
              child: Row(children: [
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Knowledge base', style: Theme.of(context).textTheme.headlineMedium),
                  const SizedBox(height: DiyaSpacing.s1),
                  Text('Local documents Diya can use in future answers.', style: Theme.of(context).textTheme.bodySmall),
                ])),
                IconButton(onPressed: () => Navigator.pop(context), icon: const Icon(Icons.close_rounded)),
              ]),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: DiyaSpacing.s5),
              child: Row(children: [
                Expanded(child: Text('${_files.length} saved files · $_chunks indexed pieces',
                  style: Theme.of(context).textTheme.bodySmall)),
                FilledButton.icon(
                  onPressed: _busy ? null : _add,
                  icon: _busy
                      ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.add_rounded),
                  label: Text(_busy ? 'Indexing…' : 'Add document'),
                ),
              ]),
            ),
            if (_error != null)
              Padding(padding: const EdgeInsets.all(DiyaSpacing.s4), child: ErrorNote(message: _error!)),
            const SizedBox(height: DiyaSpacing.s2),
            Expanded(
              child: _loading
                  ? const Center(child: CircularProgressIndicator())
                  : _files.isEmpty
                      ? Center(child: Padding(
                          padding: const EdgeInsets.all(DiyaSpacing.s6),
                          child: Text('No documents yet. Add a PDF, text, Markdown, CSV, DOCX, HTML, or JSON file to start.',
                            textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodyMedium),
                        ))
                      : ListView.separated(
                          controller: scrollController,
                          padding: const EdgeInsets.fromLTRB(DiyaSpacing.s5, DiyaSpacing.s2, DiyaSpacing.s5, DiyaSpacing.s5),
                          itemCount: _files.length,
                          separatorBuilder: (_, _) => Divider(color: c.border, height: 1),
                          itemBuilder: (context, index) {
                            final file = _files[index];
                            final name = file['name']?.toString() ?? 'Unknown file';
                            final bytes = file['size'] as int? ?? 0;
                            return ListTile(
                              contentPadding: EdgeInsets.zero,
                              leading: Icon(Icons.description_outlined, color: c.teal),
                              title: Text(name, maxLines: 1, overflow: TextOverflow.ellipsis),
                              subtitle: Text('${(bytes / 1024).ceil()} KB'),
                              trailing: IconButton(
                                tooltip: 'Remove from knowledge base',
                                onPressed: _busy ? null : () => _remove(name),
                                icon: Icon(Icons.delete_outline_rounded, color: c.danger),
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
