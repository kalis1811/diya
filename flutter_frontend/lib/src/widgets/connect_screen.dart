import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../theme/diya_tokens.dart';
import 'error_note.dart';
import 'voice_mark.dart';

/// First screen: point the app at a local Diya server and sign in.
class ConnectScreen extends StatefulWidget {
  const ConnectScreen({
    required this.connecting,
    required this.error,
    required this.onConnect,
    super.key,
  });

  final bool connecting;
  final String? error;
  final void Function(String baseUrl, String username, String password)
  onConnect;

  @override
  State<ConnectScreen> createState() => _ConnectScreenState();
}

class _ConnectScreenState extends State<ConnectScreen> {
  final _url = TextEditingController();
  final _user = TextEditingController();
  final _password = TextEditingController();
  bool _obscure = true;

  @override
  void initState() {
    super.initState();
    _url.text = _defaultUrl();
    _user.text = 'admin';
  }

  @override
  void dispose() {
    _url.dispose();
    _user.dispose();
    _password.dispose();
    super.dispose();
  }

  String _defaultUrl() {
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8400';
    }
    return 'http://127.0.0.1:8400';
  }

  void _submit() =>
      widget.onConnect(_url.text, _user.text, _password.text);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          final compact = constraints.maxWidth < 860;
          final form = _buildForm(context, compact);
          final hero = _buildHero(context, compact);
          return compact
              ? SingleChildScrollView(child: form)
              : Row(
                  children: [
                    SizedBox(width: constraints.maxWidth * .44, child: hero),
                    Expanded(child: SingleChildScrollView(child: form)),
                  ],
                );
        },
      ),
    );
  }

  Widget _buildHero(BuildContext context, bool compact) {
    return Container(
      color: DiyaTokens.of(context).navBg,
      child: SafeArea(
        child: Padding(
          padding: EdgeInsets.all(compact ? DiyaSpacing.s5 : DiyaSpacing.s7),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const VoiceMark(size: 54),
              SizedBox(height: compact ? DiyaSpacing.s6 : 64),
              Text(
                'A little help,\nright here.',
                style: Theme.of(context).textTheme.displaySmall?.copyWith(
                  color: Colors.white,
                  fontSize: compact ? 30 : 40,
                ),
              ),
              const SizedBox(height: DiyaSpacing.s4),
              const Text(
                'Diya answers from your own files and chats, on your own '
                'machine. Nothing is sent anywhere else.',
                style: TextStyle(
                  color: Color(0xFFBAC8D5),
                  fontSize: 14,
                  height: 1.65,
                ),
              ),
              const Spacer(),
              const Row(
                children: [
                  Icon(
                    Icons.lock_outline_rounded,
                    color: Color(0xFF9ED3C9),
                    size: 15,
                  ),
                  SizedBox(width: DiyaSpacing.s2),
                  Text(
                    'Private by design',
                    style: TextStyle(color: Color(0xFF9ED3C9), fontSize: 12),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildForm(BuildContext context, bool compact) {
    final c = DiyaTokens.of(context);
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 380),
        child: Padding(
          padding: EdgeInsets.symmetric(
            horizontal: DiyaSpacing.s5,
            vertical: compact ? DiyaSpacing.s5 : DiyaSpacing.s7,
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (compact) ...[
                const VoiceMark(size: 44),
                const SizedBox(height: DiyaSpacing.s5),
              ],
              Text('Connect to Diya', style: Theme.of(context).textTheme.headlineMedium),
              const SizedBox(height: DiyaSpacing.s2),
              Text(
                'The Diya server should be running on this computer.',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: DiyaSpacing.s5),
              TextField(
                controller: _url,
                enabled: !widget.connecting,
                decoration: const InputDecoration(
                  labelText: 'Server address',
                  hintText: 'http://127.0.0.1:8400',
                ),
              ),
              const SizedBox(height: DiyaSpacing.s3),
              TextField(
                controller: _user,
                enabled: !widget.connecting,
                decoration: const InputDecoration(labelText: 'Username'),
              ),
              const SizedBox(height: DiyaSpacing.s3),
              TextField(
                controller: _password,
                enabled: !widget.connecting,
                obscureText: _obscure,
                onSubmitted: (_) => _submit(),
                decoration: InputDecoration(
                  labelText: 'Password',
                  suffixIcon: IconButton(
                    tooltip: _obscure ? 'Show password' : 'Hide password',
                    onPressed: () => setState(() => _obscure = !_obscure),
                    icon: Icon(
                      _obscure
                          ? Icons.visibility_outlined
                          : Icons.visibility_off_outlined,
                      size: 18,
                    ),
                  ),
                ),
              ),
              if (widget.error != null) ...[
                const SizedBox(height: DiyaSpacing.s4),
                ErrorNote(message: widget.error!),
              ],
              const SizedBox(height: DiyaSpacing.s5),
              FilledButton(
                onPressed: widget.connecting ? null : _submit,
                child: widget.connecting
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Connect'),
              ),
              const SizedBox(height: DiyaSpacing.s4),
              Text(
                'Credentials stay in memory for this session.',
                textAlign: TextAlign.center,
                style: TextStyle(color: c.faint, fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
