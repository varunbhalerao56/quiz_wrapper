import 'dart:ffi';
import 'src/llama_service_enhanced.dart';
import 'src/utils/llama_config.dart';

/// Service that handles all LLM test operations
class DemoService {
  final String modelPath;
  LlamaServiceEnhanced? _llama;

  DemoService({required this.modelPath});

  bool get isModelLoaded => _llama != null;

  // ==================== Library Tests ====================

  /// Test if all required native libraries can be loaded
  Future<Map<String, bool>> testLibraries() async {
    final libraries = [
      'ggml-base.framework/ggml-base',
      'ggml-blas.framework/ggml-blas',
      'ggml-cpu.framework/ggml-cpu',
      'ggml-metal.framework/ggml-metal',
      'ggml.framework/ggml',
      'llama.framework/llama',
      'mtmd.framework/mtmd',
    ];

    final results = <String, bool>{};

    for (final libPath in libraries) {
      final libName = libPath.split('/').last;
      print('Testing library: $libPath');

      try {
        DynamicLibrary.open(libPath);
        results[libName] = true;
        print('✓ $libName loaded successfully');
      } catch (e) {
        results[libName] = false;
        print('✗ $libName failed: $e');
      }
    }

    return results;
  }

  // ==================== Model Operations ====================

  /// Initialize and load the model
  Future<bool> loadModel() async {
    try {
      print('Creating LlamaServiceEnhanced...');
      _llama = LlamaServiceEnhanced();

      print('Initializing backend...');
      _llama!.init();

      print('Loading model: $modelPath');
      final success = _llama!.loadModel(modelPath);

      if (success) {
        print('✓ Model loaded successfully');
      } else {
        print('✗ Model failed to load');
      }

      return success;
    } catch (e, stack) {
      print('Error loading model: $e');
      print('Stack trace: $stack');
      return false;
    }
  }

  /// Create inference context
  Future<bool> createContext({int nCtx = 2048, int nThreads = 4}) async {
    if (_llama == null) {
      print('Cannot create context: model not loaded');
      return false;
    }

    try {
      print('Creating context with n_ctx=$nCtx, n_threads=$nThreads');
      final success = _llama!.createContext(
        config: ContextConfig(nCtx: nCtx, nThreads: nThreads),
      );

      if (success) {
        print('✓ Context created successfully');
      } else {
        print('✗ Context creation failed');
      }

      return success;
    } catch (e, stack) {
      print('Error creating context: $e');
      print('Stack trace: $stack');
      return false;
    }
  }

  // ==================== Tokenization ====================

  /// Test tokenization with a sample text
  Future<Map<String, dynamic>> testTokenization(String text) async {
    if (_llama == null) {
      throw StateError('Model not loaded');
    }

    print('Tokenizing: "$text"');

    final tokens = _llama!.tokenizeText(text, addBos: true);
    final bosToken = _llama!.getBosToken();
    final eosToken = _llama!.getEosToken();

    final detokenized = <String>[];
    for (final token in tokens) {
      detokenized.add(_llama!.detokenize(token));
    }

    print('✓ Tokenization complete: ${tokens.length} tokens');

    return {'tokens': tokens, 'detokenized': detokenized, 'bosToken': bosToken, 'eosToken': eosToken};
  }

  // ==================== Text Generation ====================

  /// Test creative text generation
  Future<String?> testCreativeGeneration(String prompt, {int maxTokens = 50}) async {
    if (_llama == null) {
      throw StateError('Model not loaded');
    }

    print('\n=== Creative Generation ===');
    print('Prompt: "$prompt"');

    return await _llama!.generateEnhanced(prompt, config: SamplerConfig.creative.copyWith(maxTokens: maxTokens));
  }

  /// Test precise generation with stop strings
  Future<String?> testPreciseGeneration(String prompt, {int maxTokens = 20}) async {
    if (_llama == null) {
      throw StateError('Model not loaded');
    }

    print('\n=== Precise Generation ===');
    print('Prompt: "$prompt"');

    return await _llama!.generateEnhanced(
      prompt,
      config: const SamplerConfig(temperature: 0.3, stopStrings: ['\n', 'Q:', '?'], maxTokens: 20),
    );
  }

  /// Test JSON generation
  Future<String?> testJsonGeneration(String prompt) async {
    if (_llama == null) {
      throw StateError('Model not loaded');
    }

    print('\n=== JSON Generation ===');
    print('Prompt: "$prompt"');

    return await _llama!.generateJson(
      prompt,
      jsonConfig: const JsonConfig(strictMode: true, prettyPrint: true),
      samplerConfig: const SamplerConfig(temperature: 0.5, maxTokens: 100),
    );
  }

  /// Test streaming generation
  Stream<String> testStreamingGeneration(String prompt, {int maxTokens = 50}) async* {
    if (_llama == null) {
      throw StateError('Model not loaded');
    }

    print('\n=== Streaming Generation ===');
    print('Prompt: "$prompt"');

    yield* _llama!.generateStream(prompt, config: SamplerConfig.creative.copyWith(maxTokens: maxTokens));
  }

  // ==================== Cleanup ====================

  /// Clean up resources
  void dispose() {
    print('Disposing DemoService...');
    _llama?.dispose();
    _llama = null;
  }
}
