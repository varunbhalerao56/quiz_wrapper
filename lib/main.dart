import 'package:flutter/material.dart';
import 'dart:ffi';

import 'package:quiz_wrapper/src/llama_service_enhanced.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';
import 'views/chat_page.dart';
import 'views/embedding_page.dart';
import 'views/json_prompt_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LLama.cpp Test',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple), useMaterial3: true),
      home: const MyHomePage(title: 'LLama.cpp Tests'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _libraryStatus = 'Not tested';
  String _modelStatus = 'Not tested';
  String _contextStatus = 'Not tested';
  String _tokenStatus = 'Not tested';
  String _generateStatus = 'Not tested';
  String _streamingText = '';

  bool _testingLibraries = false;
  bool _testingModel = false;
  bool _testingContext = false;
  bool _testingTokens = false;
  bool _testingGenerate = false;

  LlamaServiceEnhanced? _llama;

  late final String _modelPath;

  @override
  void initState() {
    super.initState();
    _modelPath = '/Users/skywar56/Documents/Flutter/quiz_wrapper/assets/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';
    print('Model path: $_modelPath');
  }

  @override
  void dispose() {
    _llama?.dispose();
    super.dispose();
  }

  Future<void> _testLibraries() async {
    setState(() {
      _testingLibraries = true;
      _libraryStatus = 'Testing libraries...';
    });

    try {
      final librariesInOrder = [
        'ggml-base.framework/ggml-base',
        'ggml-blas.framework/ggml-blas',
        'ggml-cpu.framework/ggml-cpu',
        'ggml-metal.framework/ggml-metal',
        'ggml.framework/ggml',
        'llama.framework/llama',
        'mtmd.framework/mtmd',
      ];

      final results = <String>[];
      final loadedLibs = <String, DynamicLibrary>{};

      for (final libPath in librariesInOrder) {
        final libName = libPath.split('/').last;
        print('Attempting to load: $libPath');
        try {
          final dylib = DynamicLibrary.open(libPath);
          loadedLibs[libName] = dylib;
          results.add('$libName: ✓');
          print('✓ Successfully loaded: $libName');
        } catch (e) {
          print('✗ Failed to load $libName: $e');
          results.add('$libName: ✗');
        }
      }

      setState(() {
        _libraryStatus = 'Library Load Test:\n\n${results.join('\n')}';
        _testingLibraries = false;
      });
    } catch (e) {
      print('Overall error: $e');
      setState(() {
        _libraryStatus = 'Error: $e';
        _testingLibraries = false;
      });
    }
  }

  Future<void> _testModelLoading() async {
    setState(() {
      _testingModel = true;
      _modelStatus = 'Initializing llama.cpp...';
    });

    try {
      print('Creating LlamaServiceEnhanced instance...');
      _llama = LlamaServiceEnhanced();

      print('Initializing backend...');
      _llama!.init();

      setState(() {
        _modelStatus = 'Loading model with mmap...\nPlease wait...';
      });

      await Future.delayed(Duration(milliseconds: 100));

      print('Loading model: $_modelPath');
      final success = _llama!.loadModel(_modelPath);

      setState(() {
        if (success) {
          _modelStatus =
              '✓ Model Loaded!\n\n'
              'Memory-mapped successfully\n'
              'Model: tinyllama-1.1b\n'
              'Size: 638MB\n'
              'Quantization: Q4_K_M';
        } else {
          _modelStatus =
              '✗ Failed to load model\n\n'
              'Check console for details';
        }
        _testingModel = false;
      });
    } catch (e, stack) {
      print('Error: $e');
      print('Stack: $stack');
      setState(() {
        _modelStatus = '✗ Error:\n\n$e';
        _testingModel = false;
      });
    }
  }

  Future<void> _testContextCreation() async {
    if (_llama == null) {
      setState(() {
        _contextStatus = '✗ Load model first!';
      });
      return;
    }

    setState(() {
      _testingContext = true;
      _contextStatus = 'Creating context...';
    });

    await Future.delayed(Duration(milliseconds: 100));

    try {
      print('Creating context with n_ctx=2048, n_threads=4...');
      final success = _llama!.createContext(config: const ContextConfig(nCtx: 2048, nThreads: 4));

      setState(() {
        if (success) {
          _contextStatus =
              '✓ Context Created!\n\n'
              'Context size: 2048 tokens\n'
              'Threads: 4\n'
              'Ready for inference';
        } else {
          _contextStatus =
              '✗ Failed to create context\n\n'
              'Check console for details';
        }
        _testingContext = false;
      });
    } catch (e, stack) {
      print('Error: $e');
      print('Stack: $stack');
      setState(() {
        _contextStatus = '✗ Error:\n\n$e';
        _testingContext = false;
      });
    }
  }

  Future<void> _testTokenization() async {
    if (_llama == null) {
      setState(() {
        _tokenStatus = '✗ Load model first!';
      });
      return;
    }

    setState(() {
      _testingTokens = true;
      _tokenStatus = 'Testing tokenization...';
    });

    await Future.delayed(Duration(milliseconds: 100));

    try {
      // Test simple text
      final testText = 'Hello, world!';
      print('Testing tokenization with: "$testText"');

      final tokens = _llama!.tokenizeText(testText, addBos: true);

      // Get special tokens
      final bosToken = _llama!.getBosToken();
      final eosToken = _llama!.getEosToken();

      // Detokenize each token to see what they represent
      final detokenized = <String>[];
      for (final token in tokens) {
        final text = _llama!.detokenize(token);
        detokenized.add(text);
      }

      setState(() {
        _tokenStatus =
            '✓ Tokenization Test!\n\n'
            'Input: "$testText"\n'
            'Tokens (${tokens.length}): $tokens\n\n'
            'Detokenized:\n${detokenized.map((t) => '"$t"').join(' + ')}\n\n'
            'BOS token: $bosToken\n'
            'EOS token: $eosToken';
        _testingTokens = false;
      });
    } catch (e, stack) {
      print('Error: $e');
      print('Stack: $stack');
      setState(() {
        _tokenStatus = '✗ Error:\n\n$e';
        _testingTokens = false;
      });
    }
  }

  Future<void> _testGeneration() async {
    if (_llama == null) {
      setState(() {
        _generateStatus = '✗ Load model and create context first!';
      });
      return;
    }

    setState(() {
      _testingGenerate = true;
      _generateStatus = 'Testing advanced generation...';
    });

    try {
      // Test 1: Regular generation with creative sampling
      // print('\n=== Test 1: Creative Generation ===');
      // setState(() {
      //   _generateStatus = 'Running Test 1: Creative Generation...';
      // });
      // await Future.delayed(const Duration(milliseconds: 100)); // Let UI update

      // const prompt1 = 'Once upon a time, in a magical forest,';
      // final result1 = await _llama!.generateEnhanced(prompt1, config: SamplerConfig.creative.copyWith(maxTokens: 50));

      // // Test 2: Precise generation with stop strings
      // print('\n=== Test 2: Precise Generation with Stop ===');
      // setState(() {
      //   _generateStatus = 'Running Test 2: Precise Generation...';
      // });
      // await Future.delayed(const Duration(milliseconds: 100)); // Let UI update

      // const prompt2 = 'Q: What is 2+2?\nA:';
      // final result2 = await _llama!.generateEnhanced(
      //   prompt2,
      //   config: const SamplerConfig(temperature: 0.3, stopStrings: ['\n', 'Q:', '?'], maxTokens: 20),
      // );

      // Test 3: JSON generation
      print('\n=== Test 3: JSON Generation ===');
      setState(() {
        _generateStatus = 'Running Test 3: JSON Generation...';
      });
      await Future.delayed(const Duration(milliseconds: 100)); // Let UI update

      const prompt3 = 'Generate a JSON object for a person with name, age, and city:';
      final result3 = await _llama!.generateJson(
        prompt3,
        jsonConfig: const JsonConfig(strictMode: false, prettyPrint: true),
        samplerConfig: const SamplerConfig(temperature: 0.5, maxTokens: 100),
      );

      setState(() {
        _generateStatus =
            '''✓ Advanced Generation Tests Complete!


📋 JSON Generation:
Prompt: "$prompt3"
Result: ${result3 ?? "Failed"}''';
        _testingGenerate = false;
      });
    } catch (e, stack) {
      print('Error: $e');
      print('Stack: $stack');
      setState(() {
        _generateStatus = '✗ Error:\n\n$e';
        _testingGenerate = false;
      });
    }
  }

  Future<void> _testStreamingGeneration() async {
    if (_llama == null) {
      setState(() {
        _generateStatus = '✗ Load model and create context first!';
      });
      return;
    }

    setState(() {
      _testingGenerate = true;
      _generateStatus = 'Starting streaming generation...';
      _streamingText = '';
    });

    try {
      const prompt = 'Write a short poem about artificial intelligence:';
      print('\n=== Streaming Generation Test ===');

      await for (final token in _llama!.generateStream(
        prompt,
        config: SamplerConfig.creative.copyWith(maxTokens: 50),
      )) {
        setState(() {
          _streamingText += token;
          _generateStatus =
              '''🌊 Streaming Generation...

Prompt: "$prompt"

Generated so far:
$_streamingText

Progress: ${_streamingText.length} characters''';
        });

        // Small delay to see the streaming effect
        await Future.delayed(const Duration(milliseconds: 50));
      }

      setState(() {
        _generateStatus =
            '''✓ Streaming Generation Complete!

🌊 Real-time Streaming:
Prompt: "$prompt"

Generated Text:
$_streamingText''';
        _testingGenerate = false;
      });
    } catch (e, stack) {
      print('Error: $e');
      print('Stack: $stack');
      setState(() {
        _generateStatus = '✗ Streaming Error:\n\n$e';
        _testingGenerate = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
        actions: [
          PopupMenuButton<String>(
            icon: const Icon(Icons.apps),
            tooltip: 'Demo Pages',
            onSelected: (value) {
              switch (value) {
                case 'chat':
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const ChatPage()));
                  break;
                case 'embeddings':
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const EmbeddingPage()));
                  break;
                case 'json':
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const JsonPromptPage()));
                  break;
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'chat',
                child: ListTile(
                  leading: Icon(Icons.chat),
                  title: Text('AI Chat'),
                  subtitle: Text('Auto-loading chat interface'),
                ),
              ),
              const PopupMenuItem(
                value: 'embeddings',
                child: ListTile(
                  leading: Icon(Icons.search),
                  title: Text('Embeddings Demo'),
                  subtitle: Text('Similarity search & RAG'),
                ),
              ),
              const PopupMenuItem(
                value: 'json',
                child: ListTile(
                  leading: Icon(Icons.code),
                  title: Text('JSON Generator'),
                  subtitle: Text('Custom JSON prompts'),
                ),
              ),
            ],
          ),
        ],
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                // Step 1: Library Test
                const Icon(Icons.library_books, size: 48, color: Colors.blue),
                const SizedBox(height: 12),
                const Text('Step 1: Library Loading', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 16),
                _buildTestCard(_libraryStatus, Colors.blue),
                const SizedBox(height: 12),
                _buildTestButton(
                  'Test Library Loading',
                  Icons.check_circle,
                  _testLibraries,
                  _testingLibraries,
                  Colors.blue,
                ),

                const SizedBox(height: 32),
                const Divider(),
                const SizedBox(height: 32),

                // Step 2: Model Test
                const Icon(Icons.memory, size: 48, color: Colors.deepPurple),
                const SizedBox(height: 12),
                const Text('Step 2: Model Loading', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 16),
                _buildTestCard(_modelStatus, Colors.purple),
                const SizedBox(height: 12),
                _buildTestButton('Load Model', Icons.play_arrow, _testModelLoading, _testingModel, Colors.deepPurple),

                const SizedBox(height: 32),
                const Divider(),
                const SizedBox(height: 32),

                // Step 3: Context Test
                const Icon(Icons.settings, size: 48, color: Colors.green),
                const SizedBox(height: 12),
                const Text('Step 3: Context Creation', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 16),
                _buildTestCard(_contextStatus, Colors.green),
                const SizedBox(height: 12),
                _buildTestButton(
                  'Create Context',
                  Icons.add_circle,
                  _testContextCreation,
                  _testingContext,
                  Colors.green,
                ),

                const SizedBox(height: 32),
                const Divider(),
                const SizedBox(height: 32),

                // Step 4: Tokenization Test
                const Icon(Icons.text_fields, size: 48, color: Colors.orange),
                const SizedBox(height: 12),
                const Text('Step 4: Tokenization', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 16),
                _buildTestCard(_tokenStatus, Colors.orange),
                const SizedBox(height: 12),
                _buildTestButton(
                  'Test Tokenization',
                  Icons.text_snippet,
                  _testTokenization,
                  _testingTokens,
                  Colors.orange,
                ),

                const SizedBox(height: 32),
                const Divider(),
                const SizedBox(height: 32),

                // Step 5: Text Generation Test
                const Icon(Icons.auto_awesome, size: 48, color: Colors.red),
                const SizedBox(height: 12),
                const Text('Step 5: Text Generation', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 16),
                _buildTestCard(_generateStatus, Colors.red),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: _buildTestButton(
                        'Generate Text',
                        Icons.bolt,
                        _testGeneration,
                        _testingGenerate,
                        Colors.red,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildTestButton(
                        'Stream Text',
                        Icons.stream,
                        _testStreamingGeneration,
                        _testingGenerate,
                        Colors.orange,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          showModalBottomSheet(
            context: context,
            builder: (context) => Container(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('🚀 Demo Pages', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 16),
                  ListTile(
                    leading: const Icon(Icons.chat, color: Colors.blue),
                    title: const Text('AI Chat'),
                    subtitle: const Text('Auto-loading chat with streaming responses'),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.push(context, MaterialPageRoute(builder: (_) => const ChatPage()));
                    },
                  ),
                  ListTile(
                    leading: const Icon(Icons.search, color: Colors.green),
                    title: const Text('Embeddings Demo'),
                    subtitle: const Text('Similarity search and RAG functionality'),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.push(context, MaterialPageRoute(builder: (_) => const EmbeddingPage()));
                    },
                  ),
                  ListTile(
                    leading: const Icon(Icons.code, color: Colors.purple),
                    title: const Text('JSON Generator'),
                    subtitle: const Text('Custom prompts with guaranteed JSON output'),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.push(context, MaterialPageRoute(builder: (_) => const JsonPromptPage()));
                    },
                  ),
                ],
              ),
            ),
          );
        },
        icon: const Icon(Icons.rocket_launch),
        label: const Text('Demo Pages'),
      ),
    );
  }

  Widget _buildTestCard(String status, MaterialColor color) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color[50],
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color[200]!),
      ),
      child: Text(
        status,
        textAlign: TextAlign.left,
        style: const TextStyle(fontSize: 14, fontFamily: 'monospace'),
      ),
    );
  }

  Widget _buildTestButton(String label, IconData icon, VoidCallback onPressed, bool isTesting, Color color) {
    return ElevatedButton.icon(
      onPressed: isTesting ? null : onPressed,
      icon: Icon(icon),
      label: isTesting ? SizedBox(height: 20, width: 20, child: const CircularProgressIndicator()) : Text(label),
      style: ElevatedButton.styleFrom(backgroundColor: color, foregroundColor: Colors.white),
    );
  }
}
