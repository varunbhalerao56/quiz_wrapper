import 'package:flutter/material.dart';
import 'package:quiz_wrapper/src/isolates/llama_isolate_parent.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma Chat',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple), useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}

enum GenerationMode { json, precise, creative, balanced }

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _llama = LlamaIsolateParent();
  final _promptController = TextEditingController();
  final _scrollController = ScrollController();

  // Initialization state
  bool _isInitialized = false;
  bool _showStepByStep = false;
  bool _isolateStarted = false;
  bool _backendInit = false;
  bool _modelLoaded = false;
  bool _contextCreated = false;
  bool _isProcessing = false;

  // Generation settings
  GenerationMode _selectedMode = GenerationMode.json;
  bool _useStreaming = true;

  // Output
  String _response = '';
  bool _isGenerating = false;

  @override
  void dispose() {
    _llama.dispose();
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ==================== Initialization Methods ====================

  Future<void> _initializeAll() async {
    setState(() => _isProcessing = true);

    try {
      // Start isolate
      await _llama.start();
      setState(() => _isolateStarted = true);

      // Backend
      await _llama.initBackend();
      setState(() => _backendInit = true);

      // Model
      final loaded = await _llama.loadModel(
        '/Users/skywar56/Documents/Flutter/quiz_wrapper/assets/gemma-3-4b-it-Q4_K_M.gguf',
        config: const ModelConfig(nGpuLayers: 0, useMmap: true),
      );

      if (!loaded) {
        throw Exception('Failed to load model');
      }
      setState(() => _modelLoaded = true);

      // Context
      final contextCreated = await _llama.createContext(config: const ContextConfig(nCtx: 18192, nThreads: 4));

      if (!contextCreated) {
        throw Exception('Failed to create context');
      }

      setState(() {
        _contextCreated = true;
        _isInitialized = true;
        _isProcessing = false;
        _response = '✓ Model ready! Select a mode and enter your prompt.';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _response = '✗ Initialization failed: $e';
      });
    }
  }

  void _startStepByStep() {
    setState(() => _showStepByStep = true);
  }

  Future<void> _startIsolate() async {
    setState(() => _isProcessing = true);

    try {
      await _llama.start();
      setState(() {
        _isolateStarted = true;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _response = '✗ Isolate start failed: $e';
      });
    }
  }

  Future<void> _initBackend() async {
    setState(() => _isProcessing = true);

    try {
      await _llama.initBackend();
      setState(() {
        _backendInit = true;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _response = '✗ Backend init failed: $e';
      });
    }
  }

  Future<void> _loadModel() async {
    setState(() => _isProcessing = true);

    try {
      final loaded = await _llama.loadModel(
        '/Users/skywar56/Documents/Flutter/quiz_wrapper/assets/olmo-2-1124-13B-instruct-Q4_K_M.gguf',
        config: const ModelConfig(nGpuLayers: 0, useMmap: true),
      );

      if (!loaded) {
        throw Exception('Failed to load model');
      }

      setState(() {
        _modelLoaded = true;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _response = '✗ Model loading failed: $e';
      });
    }
  }

  Future<void> _createContext() async {
    setState(() => _isProcessing = true);

    try {
      final contextCreated = await _llama.createContext(config: const ContextConfig(nCtx: 4096, nThreads: 4));

      if (!contextCreated) {
        throw Exception('Failed to create context');
      }

      setState(() {
        _contextCreated = true;
        _isInitialized = true;
        _isProcessing = false;
        _response = '✓ All steps complete! Model ready for generation.';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _response = '✗ Context creation failed: $e';
      });
    }
  }

  // ==================== Generation Method ====================

  Future<void> _generate() async {
    if (_promptController.text.isEmpty) {
      _showSnackBar('Please enter a prompt');
      return;
    }

    setState(() {
      _isGenerating = true;
      _response = '';
    });

    final systemPrompt = '''
''';

    try {
      // Get sampler config based on mode
      final config = _getSamplerConfig();

      if (_useStreaming) {
        // Streaming - UI stays smooth!
        await for (final token in _llama.generateStream(
          _promptController.text,
          config: config,
          systemPrompt: _selectedMode == GenerationMode.json ? systemPrompt : null,
        )) {
          setState(() => _response += token);
          _scrollToBottom();
        }
        setState(() => _isGenerating = false);
      } else {
        // Non-streaming
        final result = await _llama.generate(_promptController.text, config: config);
        setState(() {
          _response = result ?? 'Generation failed';
          _isGenerating = false;
        });
      }

      _scrollToBottom();
    } catch (e) {
      setState(() {
        _response = 'Error: $e';
        _isGenerating = false;
      });
    }
  }

  SamplerConfig _getSamplerConfig() {
    final baseConfig = switch (_selectedMode) {
      GenerationMode.creative => SamplerConfig.creative,
      GenerationMode.balanced => SamplerConfig.balanced,
      GenerationMode.precise => SamplerConfig.precise,
      GenerationMode.json => SamplerConfig.deterministic,
    };

    return baseConfig.copyWith(
      maxTokens: 3072,
      stopStrings: [
        "<|user|>",
        "<|system|>",
        "<|assistant|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|system|>",
        "<|end|>",
        "</s>",
      ],
    );
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 100),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message), duration: const Duration(seconds: 2)));
  }

  // ==================== UI ====================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Initialization buttons
            if (!_isInitialized) ...[
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessing ? null : _startStepByStep,
                      icon: const Icon(Icons.stairs),
                      label: const Text('Init Step by Step'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.all(16),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessing ? null : _initializeAll,
                      icon: const Icon(Icons.flash_on),
                      label: const Text('Init All'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.all(16),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
            ],

            // Step-by-step UI
            if (_showStepByStep && !_isInitialized) ...[
              Card(
                elevation: 2,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text(
                        'Manual Initialization',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 16),

                      // Step 0: Start Isolate
                      _buildStepTile(
                        stepNumber: 0,
                        title: 'Start Isolate',
                        completed: _isolateStarted,
                        onPressed: !_isolateStarted && !_isProcessing ? _startIsolate : null,
                      ),
                      const SizedBox(height: 8),

                      // Step 1: Backend
                      _buildStepTile(
                        stepNumber: 1,
                        title: 'Initialize Backend',
                        completed: _backendInit,
                        enabled: _isolateStarted,
                        onPressed: _isolateStarted && !_backendInit && !_isProcessing ? _initBackend : null,
                      ),
                      const SizedBox(height: 8),

                      // Step 2: Model
                      _buildStepTile(
                        stepNumber: 2,
                        title: 'Load Model',
                        completed: _modelLoaded,
                        enabled: _backendInit,
                        onPressed: _backendInit && !_modelLoaded && !_isProcessing ? _loadModel : null,
                      ),
                      const SizedBox(height: 8),

                      // Step 3: Context
                      _buildStepTile(
                        stepNumber: 3,
                        title: 'Create Context',
                        completed: _contextCreated,
                        enabled: _modelLoaded,
                        onPressed: _modelLoaded && !_contextCreated && !_isProcessing ? _createContext : null,
                      ),

                      if (_isProcessing) ...[
                        const SizedBox(height: 16),
                        const Center(child: CircularProgressIndicator()),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],

            const SizedBox(height: 16),

            // Generation controls
            if (_isInitialized) ...[
              // Mode selection
              const Text('Generation Mode', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              SegmentedButton<GenerationMode>(
                segments: const [
                  ButtonSegment(value: GenerationMode.json, label: Text('JSON'), icon: Icon(Icons.code)),
                  ButtonSegment(value: GenerationMode.precise, label: Text('Precise'), icon: Icon(Icons.gps_fixed)),
                  ButtonSegment(
                    value: GenerationMode.creative,
                    label: Text('Creative'),
                    icon: Icon(Icons.auto_awesome),
                  ),
                  ButtonSegment(value: GenerationMode.balanced, label: Text('Balanced'), icon: Icon(Icons.balance)),
                ],
                selected: {_selectedMode},
                onSelectionChanged: (Set<GenerationMode> newSelection) {
                  setState(() => _selectedMode = newSelection.first);
                },
              ),
              const SizedBox(height: 16),

              // Streaming toggle
              SwitchListTile(
                title: const Text('Use Streaming'),
                subtitle: const Text('Generate tokens in real-time'),
                value: _useStreaming,
                onChanged: (value) => setState(() => _useStreaming = value),
              ),
              const SizedBox(height: 16),

              // Prompt input
              TextField(
                controller: _promptController,
                decoration: const InputDecoration(
                  labelText: 'Prompt',
                  hintText: 'Enter your prompt here...',
                  border: OutlineInputBorder(),
                ),
                maxLines: 8,
                enabled: !_isGenerating,
              ),
              const SizedBox(height: 16),

              // Submit button
              ElevatedButton.icon(
                onPressed: _isGenerating ? null : _generate,
                icon: _isGenerating
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                      )
                    : const Icon(Icons.send),
                label: Text(_isGenerating ? 'Generating...' : 'Generate'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurple,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.all(16),
                ),
              ),
              const SizedBox(height: 24),

              const Divider(thickness: 2),
              const SizedBox(height: 16),

              // Response area
              const Text('Output', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Container(
                width: double.infinity,
                constraints: const BoxConstraints(minHeight: 200),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey[100],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey[300]!),
                ),
                child: SelectableText(_response, style: const TextStyle(fontSize: 14, fontFamily: 'monospace')),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildStepTile({
    required int stepNumber,
    required String title,
    required bool completed,
    bool enabled = true,
    VoidCallback? onPressed,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: completed ? Colors.green[50] : (enabled ? Colors.white : Colors.grey[100]),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: completed ? Colors.green : (enabled ? Colors.blue : Colors.grey[300]!), width: 2),
      ),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: completed ? Colors.green : (enabled ? Colors.blue : Colors.grey),
          child: completed
              ? const Icon(Icons.check, color: Colors.white)
              : Text('$stepNumber', style: const TextStyle(color: Colors.white)),
        ),
        title: Text(
          title,
          style: TextStyle(fontWeight: FontWeight.bold, color: enabled ? Colors.black : Colors.grey),
        ),
        trailing: onPressed != null
            ? ElevatedButton(
                onPressed: onPressed,
                style: ElevatedButton.styleFrom(backgroundColor: Colors.blue, foregroundColor: Colors.white),
                child: const Text('Run'),
              )
            : completed
            ? const Icon(Icons.check_circle, color: Colors.green)
            : null,
      ),
    );
  }
}
