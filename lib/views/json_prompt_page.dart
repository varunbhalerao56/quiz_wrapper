/// Custom JSON prompt page
///
/// Allows users to enter custom prompts and request specific JSON formats
/// with configurable sampling parameters and validation.

import 'package:flutter/material.dart';
import '../src/llama_service_enhanced.dart';
import '../src/utils/llama_config.dart';

class JsonPromptPage extends StatefulWidget {
  const JsonPromptPage({super.key});

  @override
  State<JsonPromptPage> createState() => _JsonPromptPageState();
}

class _JsonPromptPageState extends State<JsonPromptPage> {
  LlamaServiceEnhanced? _llama;
  final TextEditingController _promptController = TextEditingController();
  final TextEditingController _schemaController = TextEditingController();

  bool _isLoading = true;
  bool _isGenerating = false;
  String _loadingStatus = 'Initializing...';
  String _jsonResult = '';

  // Sampling configuration
  SamplerConfig _samplerConfig = SamplerConfig.balanced;
  bool _strictMode = true;
  bool _prettyPrint = true;

  // Preset prompts
  final Map<String, String> _presetPrompts = {
    'User Profile': 'Generate a user profile with name, age, email, and preferences',
    'Product Info': 'Create a product listing with name, price, description, and category',
    'Weather Data': 'Generate weather information with temperature, humidity, conditions, and forecast',
    'Recipe': 'Create a recipe with ingredients, instructions, prep time, and difficulty',
    'Book Info': 'Generate book information with title, author, genre, year, and summary',
    'Company Data': 'Create company information with name, industry, employees, and revenue',
  };

  @override
  void initState() {
    super.initState();
    _autoLoadModel();
  }

  @override
  void dispose() {
    _llama?.dispose();
    _promptController.dispose();
    _schemaController.dispose();
    super.dispose();
  }

  /// Auto-load the model
  Future<void> _autoLoadModel() async {
    try {
      setState(() {
        _loadingStatus = 'Creating service...';
      });

      _llama = LlamaServiceEnhanced();

      setState(() {
        _loadingStatus = 'Initializing backend...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      _llama!.init();

      setState(() {
        _loadingStatus = 'Loading model...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      final modelLoaded = _llama!.loadModel(
        '/Users/skywar56/Documents/Flutter/quiz_wrapper/assets/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        config: const ModelConfig(useMmap: true),
      );

      if (!modelLoaded) {
        throw Exception('Failed to load model');
      }

      setState(() {
        _loadingStatus = 'Creating context...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      final contextCreated = _llama!.createContext(config: const ContextConfig(nCtx: 2048, nThreads: 4));

      if (!contextCreated) {
        throw Exception('Failed to create context');
      }

      setState(() {
        _isLoading = false;
        _loadingStatus = 'Ready for JSON generation!';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _loadingStatus = 'Error: $e';
      });
    }
  }

  /// Generate JSON from prompt
  Future<void> _generateJson() async {
    if (_promptController.text.trim().isEmpty) return;

    setState(() {
      _isGenerating = true;
      _jsonResult = 'Generating JSON...';
    });

    try {
      final prompt = _promptController.text.trim();

      final result = await _llama!.generateJson(
        prompt,
        jsonConfig: JsonConfig(strictMode: _strictMode, prettyPrint: _prettyPrint),
        samplerConfig: _samplerConfig.copyWith(maxTokens: 200),
      );

      setState(() {
        if (result != null) {
          _jsonResult =
              '''✅ JSON Generated Successfully!

📝 Prompt: "$prompt"

🎛️ Sampling Config:
• Temperature: ${_samplerConfig.temperature}
• Top-P: ${_samplerConfig.topP}
• Top-K: ${_samplerConfig.topK}
• Repeat Penalty: ${_samplerConfig.repeatPenalty}

⚙️ JSON Config:
• Strict Mode: $_strictMode
• Pretty Print: $_prettyPrint

📋 Generated JSON:
$result''';
        } else {
          _jsonResult = '''❌ JSON Generation Failed

The model could not generate valid JSON for this prompt.
Try:
• Simplifying the prompt
• Using a lower temperature
• Enabling non-strict mode''';
        }
      });
    } catch (e) {
      setState(() {
        _jsonResult = 'Error: $e';
      });
    } finally {
      setState(() {
        _isGenerating = false;
      });
    }
  }

  /// Generate simple JSON (fallback method)
  Future<void> _generateSimpleJson() async {
    if (_promptController.text.trim().isEmpty) return;

    setState(() {
      _isGenerating = true;
      _jsonResult = 'Generating simple JSON...';
    });

    try {
      final prompt = _promptController.text.trim();

      final result = await _llama!.generateSimpleJson(
        prompt,
        jsonConfig: JsonConfig(strictMode: _strictMode, prettyPrint: _prettyPrint),
      );

      setState(() {
        if (result != null) {
          _jsonResult =
              '''✅ Simple JSON Generated!

📝 Prompt: "$prompt"

🎛️ Method: Template-based fallback
⚙️ Always produces valid JSON

📋 Generated JSON:
$result''';
        } else {
          _jsonResult = '''❌ Simple JSON Generation Failed

This shouldn't happen with the fallback method.''';
        }
      });
    } catch (e) {
      setState(() {
        _jsonResult = 'Error: $e';
      });
    } finally {
      setState(() {
        _isGenerating = false;
      });
    }
  }

  /// Load a preset prompt
  void _loadPreset(String preset) {
    _promptController.text = _presetPrompts[preset] ?? '';
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('JSON Generator'),
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 24),
              Text(_loadingStatus, style: const TextStyle(fontSize: 18), textAlign: TextAlign.center),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('JSON Generator'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Info card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('📋 JSON Generator', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    const Text(
                      'Enter a prompt describing what JSON you want, configure the sampling parameters, '
                      'and get guaranteed valid JSON output using grammar constraints.',
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Preset prompts
            const Text('Quick Presets:', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _presetPrompts.keys
                  .map(
                    (preset) => ActionChip(
                      label: Text(preset),
                      onPressed: () => _loadPreset(preset),
                      backgroundColor: Theme.of(context).colorScheme.primaryContainer,
                    ),
                  )
                  .toList(),
            ),

            const SizedBox(height: 24),

            // Prompt input
            TextField(
              controller: _promptController,
              decoration: const InputDecoration(
                labelText: 'JSON Generation Prompt',
                hintText: 'Describe what JSON structure you want...',
                border: OutlineInputBorder(),
                helperText: 'Be specific about the fields and data types you want',
              ),
              maxLines: 3,
            ),

            const SizedBox(height: 24),

            // Sampling configuration
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '🎛️ Sampling Configuration',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),

                    // Preset selector
                    const Text('Preset:'),
                    const SizedBox(height: 8),
                    SegmentedButton<String>(
                      segments: const [
                        ButtonSegment(value: 'deterministic', label: Text('Deterministic')),
                        ButtonSegment(value: 'precise', label: Text('Precise')),
                        ButtonSegment(value: 'balanced', label: Text('Balanced')),
                        ButtonSegment(value: 'creative', label: Text('Creative')),
                      ],
                      selected: {_getPresetName()},
                      onSelectionChanged: (Set<String> selection) {
                        setState(() {
                          switch (selection.first) {
                            case 'deterministic':
                              _samplerConfig = SamplerConfig.deterministic;
                              break;
                            case 'precise':
                              _samplerConfig = SamplerConfig.precise;
                              break;
                            case 'balanced':
                              _samplerConfig = SamplerConfig.balanced;
                              break;
                            case 'creative':
                              _samplerConfig = SamplerConfig.creative;
                              break;
                          }
                        });
                      },
                    ),

                    const SizedBox(height: 16),

                    // Temperature slider
                    Text('Temperature: ${_samplerConfig.temperature.toStringAsFixed(2)}'),
                    Slider(
                      value: _samplerConfig.temperature,
                      min: 0.0,
                      max: 2.0,
                      divisions: 20,
                      onChanged: (value) {
                        setState(() {
                          _samplerConfig = _samplerConfig.copyWith(temperature: value);
                        });
                      },
                    ),

                    // Top-P slider
                    Text('Top-P: ${_samplerConfig.topP.toStringAsFixed(2)}'),
                    Slider(
                      value: _samplerConfig.topP,
                      min: 0.1,
                      max: 1.0,
                      divisions: 18,
                      onChanged: (value) {
                        setState(() {
                          _samplerConfig = _samplerConfig.copyWith(topP: value);
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // JSON configuration
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('⚙️ JSON Configuration', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 12),

                    SwitchListTile(
                      title: const Text('Strict Mode'),
                      subtitle: const Text('Reject invalid JSON'),
                      value: _strictMode,
                      onChanged: (value) {
                        setState(() {
                          _strictMode = value;
                        });
                      },
                    ),

                    SwitchListTile(
                      title: const Text('Pretty Print'),
                      subtitle: const Text('Format JSON with indentation'),
                      value: _prettyPrint,
                      onChanged: (value) {
                        setState(() {
                          _prettyPrint = value;
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Generate buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isGenerating ? null : _generateJson,
                    icon: _isGenerating
                        ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.auto_fix_high),
                    label: Text(_isGenerating ? 'Generating...' : 'Generate JSON'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.purple,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.all(16),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isGenerating ? null : _generateSimpleJson,
                    icon: const Icon(Icons.auto_fix_normal),
                    label: const Text('Simple JSON'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.orange,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.all(16),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 24),

            // Results
            if (_jsonResult.isNotEmpty) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Text('📋 Generated JSON', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                          const Spacer(),
                          if (_jsonResult.contains('✅')) ...[
                            IconButton(icon: const Icon(Icons.copy), onPressed: _copyToClipboard, tooltip: 'Copy JSON'),
                          ],
                        ],
                      ),
                      const SizedBox(height: 8),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.grey[100],
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.grey[300]!),
                        ),
                        child: SelectableText(
                          _jsonResult,
                          style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],

            const SizedBox(height: 24),

            // Tips card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '💡 JSON Generation Methods',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 12),
                    const Text('🔥 Generate JSON: Advanced method with grammar constraints'),
                    const Text('🛡️ Simple JSON: Reliable fallback with template approach'),
                    const SizedBox(height: 8),
                    const Text('Tips for better results:'),
                    const Text('• Be specific about field names and types'),
                    const Text('• Use lower temperature for consistent structure'),
                    const Text('• Try Simple JSON if regular method fails'),
                    const SizedBox(height: 12),
                    const Text(
                      'Example: "Create a user object with string name, integer age, boolean isActive, and array of string hobbies"',
                      style: TextStyle(fontStyle: FontStyle.italic, color: Colors.grey),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Get current preset name
  String _getPresetName() {
    if (_samplerConfig.temperature == 0.0) return 'deterministic';
    if (_samplerConfig.temperature == 0.3) return 'precise';
    if (_samplerConfig.temperature == 0.7) return 'balanced';
    if (_samplerConfig.temperature == 0.9) return 'creative';
    return 'custom';
  }

  /// Copy JSON result to clipboard
  void _copyToClipboard() {
    if (_jsonResult.contains('Generated JSON:')) {
      // Copy to clipboard (you'd need to add clipboard package)
      // Clipboard.setData(ClipboardData(text: jsonPart));

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('JSON copied to clipboard! (Feature needs clipboard package)'),
          duration: Duration(seconds: 2),
        ),
      );
    }
  }
}
