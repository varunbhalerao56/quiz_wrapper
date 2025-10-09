import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'ffi/llama_ffi.dart';

/// Configuration for sampling parameters
class SamplerConfig {
  final double temperature;
  final double topP;
  final int topK;
  final double minP;
  final double repeatPenalty;
  final double frequencyPenalty;
  final double presencePenalty;
  final List<String> stopStrings;
  final int maxTokens;
  final int seed;

  const SamplerConfig({
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK = 40,
    this.minP = 0.05,
    this.repeatPenalty = 1.1,
    this.frequencyPenalty = 0.0,
    this.presencePenalty = 0.0,
    this.stopStrings = const [],
    this.maxTokens = 100,
    this.seed = -1, // -1 = random seed
  });

  // Preset configurations
  static const SamplerConfig creative = SamplerConfig(temperature: 0.9, topP: 0.95, topK: 50, repeatPenalty: 1.05);

  static const SamplerConfig balanced = SamplerConfig(temperature: 0.7, topP: 0.9, topK: 40, repeatPenalty: 1.1);

  static const SamplerConfig precise = SamplerConfig(temperature: 0.3, topP: 0.8, topK: 20, repeatPenalty: 1.15);

  static const SamplerConfig deterministic = SamplerConfig(temperature: 0.0, topP: 1.0, topK: 1, repeatPenalty: 1.0);
}

/// Extension for copying SamplerConfig with modifications
extension SamplerConfigExtension on SamplerConfig {
  SamplerConfig copyWith({
    double? temperature,
    double? topP,
    int? topK,
    double? minP,
    double? repeatPenalty,
    double? frequencyPenalty,
    double? presencePenalty,
    List<String>? stopStrings,
    int? maxTokens,
    int? seed,
  }) {
    return SamplerConfig(
      temperature: temperature ?? this.temperature,
      topP: topP ?? this.topP,
      topK: topK ?? this.topK,
      minP: minP ?? this.minP,
      repeatPenalty: repeatPenalty ?? this.repeatPenalty,
      frequencyPenalty: frequencyPenalty ?? this.frequencyPenalty,
      presencePenalty: presencePenalty ?? this.presencePenalty,
      stopStrings: stopStrings ?? this.stopStrings,
      maxTokens: maxTokens ?? this.maxTokens,
      seed: seed ?? this.seed,
    );
  }
}

/// Configuration for JSON output
class JsonConfig {
  final Map<String, dynamic>? schema;
  final bool strictMode;
  final bool prettyPrint;

  const JsonConfig({this.schema, this.strictMode = true, this.prettyPrint = false});
}

/// Service class that wraps the generated FFI bindings for llama.cpp
/// Provides the same API as LlamaSimple but uses the generated bindings
class LlamaService {
  late final DynamicLibrary _lib;
  late final llama_cpp _llamaCpp;

  Pointer<llama_model>? _model;
  Pointer<llama_context>? _context;
  Pointer<llama_vocab>? _vocab;
  Pointer<llama_sampler>? _sampler;
  bool _initialized = false;

  LlamaService() {
    _lib = DynamicLibrary.open('llama.framework/llama');
    _llamaCpp = llama_cpp(_lib);
  }

  /// Initialize the llama.cpp backend
  void init() {
    if (_initialized) return;
    _llamaCpp.llama_backend_init();
    _initialized = true;
    print('✓ LlamaService initialized with generated FFI bindings');
  }

  /// Load a model from file
  bool loadModel(String modelPath) {
    if (!_initialized) {
      throw StateError('Must call init() first');
    }

    if (!File(modelPath).existsSync()) {
      print('Model file not found: $modelPath');
      return false;
    }

    // Get default model parameters
    final modelParams = _llamaCpp.llama_model_default_params();

    // Configure model parameters
    modelParams.use_mmap = true;
    modelParams.use_mlock = false;
    modelParams.n_gpu_layers = 0; // CPU only for now
    modelParams.vocab_only = false;
    modelParams.check_tensors = true;

    print('Model params:');
    print('  use_mmap: ${modelParams.use_mmap}');
    print('  use_mlock: ${modelParams.use_mlock}');
    print('  n_gpu_layers: ${modelParams.n_gpu_layers}');

    // Load model
    final pathPtr = modelPath.toNativeUtf8();
    try {
      _model = _llamaCpp.llama_load_model_from_file(pathPtr.cast<Char>(), modelParams);

      if (_model == nullptr || _model!.address == 0) {
        print('Failed to load model');
        return false;
      }

      print('✓ Model loaded successfully with memory mapping!');

      // Get vocab handle for tokenization
      _vocab = _llamaCpp.llama_model_get_vocab(_model!);
      if (_vocab == nullptr || _vocab!.address == 0) {
        print('Warning: Failed to get vocab handle');
      }

      return true;
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Create a context for inference
  bool createContext({int nCtx = 2048, int nThreads = 4}) {
    if (_model == null || _model!.address == 0) {
      print('Must load model first');
      return false;
    }

    if (_context != null && _context!.address != 0) {
      print('Context already exists');
      return false;
    }

    // Get default context parameters
    final ctxParams = _llamaCpp.llama_context_default_params();

    // Set our custom parameters
    ctxParams.n_ctx = nCtx;
    ctxParams.n_threads = nThreads;
    ctxParams.n_threads_batch = nThreads;
    ctxParams.embeddings = false;
    ctxParams.offload_kqv = false;
    ctxParams.no_perf = false;

    print('Context params:');
    print('  n_ctx: ${ctxParams.n_ctx}');
    print('  n_threads: ${ctxParams.n_threads}');

    // Create context
    _context = _llamaCpp.llama_new_context_with_model(_model!, ctxParams);

    if (_context == nullptr || _context!.address == 0) {
      print('Failed to create context');
      return false;
    }

    final actualCtx = _llamaCpp.llama_n_ctx(_context!);
    print('✓ Context created successfully! Actual n_ctx: $actualCtx');

    // Initialize greedy sampler
    final samplerParams = _llamaCpp.llama_sampler_chain_default_params();
    samplerParams.no_perf = false;
    _sampler = _llamaCpp.llama_sampler_chain_init(samplerParams);

    if (_sampler == nullptr || _sampler!.address == 0) {
      print('Failed to create sampler');
      return false;
    }

    final greedySampler = _llamaCpp.llama_sampler_init_greedy();
    _llamaCpp.llama_sampler_chain_add(_sampler!, greedySampler);
    print('✓ Greedy sampler initialized');

    return true;
  }

  /// Tokenize text into a list of token IDs
  List<int> tokenizeText(String text, {bool addBos = true, bool special = true}) {
    if (_model == null || _model!.address == 0) {
      throw StateError('Must load model first');
    }

    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Vocab not available');
    }

    final textPtr = text.toNativeUtf8();
    try {
      // First pass: get token count (negative return means needed size)
      final tokenCount = -_llamaCpp.llama_tokenize(
        _vocab!,
        textPtr.cast<Char>(),
        text.length,
        nullptr,
        0,
        addBos,
        special,
      );

      if (tokenCount <= 0) {
        print('Tokenization returned invalid count: $tokenCount');
        return [];
      }

      print('Need $tokenCount tokens for "$text"');

      // Allocate buffer for tokens
      final tokensPtr = malloc<llama_token>(tokenCount);
      try {
        // Second pass: get actual tokens
        final actualCount = _llamaCpp.llama_tokenize(
          _vocab!,
          textPtr.cast<Char>(),
          text.length,
          tokensPtr,
          tokenCount,
          addBos,
          special,
        );

        if (actualCount < 0) {
          print('Failed to get tokens, error: $actualCount');
          return [];
        }

        // Convert to Dart list
        final tokens = <int>[];
        for (int i = 0; i < actualCount; i++) {
          tokens.add(tokensPtr[i]);
        }

        print('✓ Tokenized "$text" into ${tokens.length} tokens: $tokens');
        return tokens;
      } finally {
        malloc.free(tokensPtr);
      }
    } finally {
      malloc.free(textPtr);
    }
  }

  /// Convert a token ID back to text
  String detokenize(int token) {
    if (_model == null || _model!.address == 0) {
      throw StateError('Must load model first');
    }

    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Vocab not available');
    }

    // Allocate buffer for the token string (128 bytes should be enough)
    final bufferPtr = malloc<Int8>(128);
    try {
      final length = _llamaCpp.llama_token_to_piece(
        _vocab!,
        token,
        bufferPtr.cast<Char>(),
        128,
        0, // lstrip (0 = no strip)
        true, // special (render special tokens)
      );

      if (length < 0) {
        print('Failed to detokenize token $token, error: $length');
        return '';
      }

      if (length == 0) {
        return '';
      }

      // Convert to Dart string
      final bytes = bufferPtr.cast<Uint8>().asTypedList(length);
      return String.fromCharCodes(bytes);
    } finally {
      malloc.free(bufferPtr);
    }
  }

  /// Get the beginning-of-sequence token ID
  int getBosToken() {
    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Must load model first');
    }
    return _llamaCpp.llama_token_bos(_vocab!);
  }

  /// Get the end-of-sequence token ID
  int getEosToken() {
    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Must load model first');
    }
    return _llamaCpp.llama_token_eos(_vocab!);
  }

  /// Create a configurable sampler chain based on the provided config
  Pointer<llama_sampler> _createConfigurableSampler(SamplerConfig config) {
    final samplerParams = _llamaCpp.llama_sampler_chain_default_params();
    samplerParams.no_perf = false;
    final chain = _llamaCpp.llama_sampler_chain_init(samplerParams);

    print('Creating sampler with config:');
    print('  temperature: ${config.temperature}');
    print('  top_p: ${config.topP}');
    print('  top_k: ${config.topK}');
    print('  min_p: ${config.minP}');
    print('  repeat_penalty: ${config.repeatPenalty}');

    // Add samplers in the correct order (order matters!)

    // 1. Penalties first (they modify logits based on history)
    if (config.repeatPenalty != 1.0 || config.frequencyPenalty != 0.0 || config.presencePenalty != 0.0) {
      final penalties = _llamaCpp.llama_sampler_init_penalties(
        64, // penalty_last_n (look back 64 tokens for repetition)
        config.repeatPenalty,
        config.frequencyPenalty,
        config.presencePenalty,
      );
      _llamaCpp.llama_sampler_chain_add(chain, penalties);
      print('  ✓ Added penalties sampler');
    }

    // 2. Top-K filtering (keep only K most likely tokens)
    if (config.topK > 0 && config.topK < 1000) {
      final topK = _llamaCpp.llama_sampler_init_top_k(config.topK);
      _llamaCpp.llama_sampler_chain_add(chain, topK);
      print('  ✓ Added top-k sampler');
    }

    // 3. Min-P filtering (remove tokens below minimum probability)
    if (config.minP > 0.0 && config.minP < 1.0) {
      final minP = _llamaCpp.llama_sampler_init_min_p(config.minP, 1);
      _llamaCpp.llama_sampler_chain_add(chain, minP);
      print('  ✓ Added min-p sampler');
    }

    // 4. Top-P (nucleus) filtering (keep tokens that sum to P probability mass)
    if (config.topP < 1.0 && config.topP > 0.0) {
      final topP = _llamaCpp.llama_sampler_init_top_p(config.topP, 1);
      _llamaCpp.llama_sampler_chain_add(chain, topP);
      print('  ✓ Added top-p sampler');
    }

    // 5. Temperature scaling (adjust randomness)
    if (config.temperature != 1.0) {
      final temp = _llamaCpp.llama_sampler_init_temp(config.temperature);
      _llamaCpp.llama_sampler_chain_add(chain, temp);
      print('  ✓ Added temperature sampler');
    }

    // 6. Final sampling method
    final finalSampler = config.temperature == 0.0
        ? _llamaCpp.llama_sampler_init_greedy()
        : _llamaCpp.llama_sampler_init_dist(config.seed == -1 ? DateTime.now().millisecondsSinceEpoch : config.seed);

    _llamaCpp.llama_sampler_chain_add(chain, finalSampler);
    print('  ✓ Added final sampler (${config.temperature == 0.0 ? "greedy" : "distribution"})');

    return chain;
  }

  /// Check if generated text contains any stop strings
  bool _containsStopString(String text, List<String> stopStrings) {
    for (final stopString in stopStrings) {
      if (text.contains(stopString)) {
        return true;
      }
    }
    return false;
  }

  /// Extract text up to the first stop string
  String _truncateAtStopString(String text, List<String> stopStrings) {
    int earliestIndex = text.length;

    for (final stopString in stopStrings) {
      final index = text.indexOf(stopString);
      if (index != -1 && index < earliestIndex) {
        earliestIndex = index;
      }
    }

    return text.substring(0, earliestIndex);
  }

  /// Generate text from a prompt with configurable sampling
  String? generate(String prompt, {SamplerConfig? config}) {
    config ??= const SamplerConfig();

    if (_context == null || _context!.address == 0) {
      print('Must create context first');
      return null;
    }

    print('\n=== Starting generation with configurable sampling ===');
    print('Prompt: "$prompt"');

    // Tokenize the prompt
    final tokens = tokenizeText(prompt);
    if (tokens.isEmpty) {
      print('Failed to tokenize prompt');
      return null;
    }

    final nPrompt = tokens.length;
    print('Prompt tokens: $nPrompt');

    // Create configurable sampler instead of hardcoded greedy
    if (_sampler != null && _sampler!.address != 0) {
      _llamaCpp.llama_sampler_free(_sampler!);
    }
    _sampler = _createConfigurableSampler(config);

    if (_sampler == null || _sampler!.address == 0) {
      print('Failed to create configurable sampler');
      return null;
    }

    // Create batch for prompt
    final tokensPtr = malloc<llama_token>(nPrompt);
    for (int i = 0; i < nPrompt; i++) {
      tokensPtr[i] = tokens[i];
    }

    try {
      var batch = _llamaCpp.llama_batch_get_one(tokensPtr, nPrompt);

      print('Decoding prompt batch...');
      if (_llamaCpp.llama_decode(_context!, batch) != 0) {
        print('Failed to decode prompt');
        return null;
      }
      print('✓ Prompt decoded');

      // Generate tokens
      final result = StringBuffer();
      int nDecoded = 0;
      int nPos = nPrompt;

      final singleTokenPtr = malloc<llama_token>();
      try {
        while (nDecoded < config.maxTokens && nPos < _llamaCpp.llama_n_ctx(_context!)) {
          // Sample next token using configured sampler
          final newToken = _llamaCpp.llama_sampler_sample(_sampler!, _context!, -1);

          // Check if end of generation
          if (_llamaCpp.llama_token_is_eog(_vocab!, newToken)) {
            print('\n✓ Hit end-of-generation token');
            break;
          }

          // Detokenize and add to result
          final tokenText = detokenize(newToken);
          result.write(tokenText);
          print(tokenText); // Print as we generate

          // Check for stop strings
          final currentText = result.toString();
          if (config.stopStrings.isNotEmpty && _containsStopString(currentText, config.stopStrings)) {
            print('\n✓ Hit stop string');
            final truncated = _truncateAtStopString(currentText, config.stopStrings);
            return truncated;
          }

          // Prepare batch with single new token
          singleTokenPtr[0] = newToken;
          batch = _llamaCpp.llama_batch_get_one(singleTokenPtr, 1);

          // Decode the new token
          if (_llamaCpp.llama_decode(_context!, batch) != 0) {
            print('\nFailed to decode token $nDecoded');
            break;
          }

          nDecoded++;
          nPos++;
        }
      } finally {
        malloc.free(singleTokenPtr);
      }

      print('\n\n=== Generation complete ===');
      print('Generated $nDecoded tokens');

      return result.toString();
    } finally {
      malloc.free(tokensPtr);
    }
  }

  /// Generate JSON output with grammar constraints
  String? generateJson(String prompt, {JsonConfig? jsonConfig, SamplerConfig? samplerConfig}) {
    jsonConfig ??= const JsonConfig();
    samplerConfig ??= const SamplerConfig();

    if (_context == null || _context!.address == 0) {
      print('Must create context first');
      return null;
    }

    if (_vocab == null || _vocab!.address == 0) {
      print('Vocab not available');
      return null;
    }

    print('\n=== Starting JSON generation ===');
    print('Prompt: "$prompt"');

    // JSON Grammar (simplified but functional)
    final jsonGrammar = '''
root ::= object
object ::= "{" ws ( string ":" ws value ( "," ws string ":" ws value )* )? ws "}"
array ::= "[" ws ( value ( "," ws value )* )? ws "]"
value ::= object | array | string | number | boolean | null
string ::= "\\"" ( [^"\\\\] | "\\\\" ["\\\\/bfnrt] | "\\\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* "\\""
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
boolean ::= "true" | "false"
null ::= "null"
ws ::= [ \\t\\n\\r]*
''';

    final grammarPtr = jsonGrammar.toNativeUtf8();
    final rootPtr = "root".toNativeUtf8();

    try {
      // Create grammar sampler
      final grammarSampler = _llamaCpp.llama_sampler_init_grammar(
        _vocab!,
        grammarPtr.cast<Char>(),
        rootPtr.cast<Char>(),
      );

      if (grammarSampler == nullptr || grammarSampler.address == 0) {
        print('Failed to create grammar sampler');
        return null;
      }

      // Create configurable sampler chain
      if (_sampler != null && _sampler!.address != 0) {
        _llamaCpp.llama_sampler_free(_sampler!);
      }
      _sampler = _createConfigurableSampler(samplerConfig);

      // Add grammar sampler to the chain (grammar should be last before final sampling)
      _llamaCpp.llama_sampler_chain_add(_sampler!, grammarSampler);
      print('✓ Added JSON grammar constraints');

      // Generate with grammar constraints
      final result = _generateWithSampler(prompt, samplerConfig);

      if (result != null && jsonConfig.strictMode) {
        // Validate JSON
        try {
          final decoded = jsonDecode(result);
          print('✓ Generated valid JSON');

          if (jsonConfig.prettyPrint) {
            return const JsonEncoder.withIndent('  ').convert(decoded);
          }
          return result;
        } catch (e) {
          print('✗ Generated invalid JSON: $e');
          return jsonConfig.strictMode ? null : result;
        }
      }

      return result;
    } finally {
      malloc.free(grammarPtr);
      malloc.free(rootPtr);
    }
  }

  /// Internal generation method that uses the current sampler
  String? _generateWithSampler(String prompt, SamplerConfig config) {
    // Tokenize the prompt
    final tokens = tokenizeText(prompt);
    if (tokens.isEmpty) {
      print('Failed to tokenize prompt');
      return null;
    }

    final nPrompt = tokens.length;
    print('Prompt tokens: $nPrompt');

    // Create batch for prompt
    final tokensPtr = malloc<llama_token>(nPrompt);
    for (int i = 0; i < nPrompt; i++) {
      tokensPtr[i] = tokens[i];
    }

    try {
      var batch = _llamaCpp.llama_batch_get_one(tokensPtr, nPrompt);

      print('Decoding prompt batch...');
      if (_llamaCpp.llama_decode(_context!, batch) != 0) {
        print('Failed to decode prompt');
        return null;
      }
      print('✓ Prompt decoded');

      // Generate tokens
      final result = StringBuffer();
      int nDecoded = 0;
      int nPos = nPrompt;

      final singleTokenPtr = malloc<llama_token>();
      try {
        while (nDecoded < config.maxTokens && nPos < _llamaCpp.llama_n_ctx(_context!)) {
          // Sample next token (grammar will constrain the choices)
          final newToken = _llamaCpp.llama_sampler_sample(_sampler!, _context!, -1);

          // Check if end of generation
          if (_llamaCpp.llama_token_is_eog(_vocab!, newToken)) {
            print('\n✓ Hit end-of-generation token');
            break;
          }

          // Detokenize and add to result
          final tokenText = detokenize(newToken);
          result.write(tokenText);
          print(tokenText); // Print as we generate

          // Check for stop strings
          final currentText = result.toString();
          if (config.stopStrings.isNotEmpty && _containsStopString(currentText, config.stopStrings)) {
            print('\n✓ Hit stop string');
            final truncated = _truncateAtStopString(currentText, config.stopStrings);
            return truncated;
          }

          // Prepare batch with single new token
          singleTokenPtr[0] = newToken;
          batch = _llamaCpp.llama_batch_get_one(singleTokenPtr, 1);

          // Decode the new token
          if (_llamaCpp.llama_decode(_context!, batch) != 0) {
            print('\nFailed to decode token $nDecoded');
            break;
          }

          nDecoded++;
          nPos++;
        }
      } finally {
        malloc.free(singleTokenPtr);
      }

      print('\n\n=== Generation complete ===');
      print('Generated $nDecoded tokens');

      return result.toString();
    } finally {
      malloc.free(tokensPtr);
    }
  }

  /// Clean up resources
  void dispose() {
    if (_sampler != null && _sampler!.address != 0) {
      _llamaCpp.llama_sampler_free(_sampler!);
      _sampler = null;
    }

    if (_context != null && _context!.address != 0) {
      _llamaCpp.llama_free(_context!);
      _context = null;
    }

    if (_model != null && _model!.address != 0) {
      _llamaCpp.llama_model_free(_model!);
      _model = null;
    }

    if (_initialized) {
      _llamaCpp.llama_backend_free();
      _initialized = false;
    }

    print('✓ LlamaService disposed');
  }
}
