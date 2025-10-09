/// Enhanced LlamaService that extends the working version with new features
///
/// This provides immediate access to the new sampling and JSON features
/// while the full modular architecture is being refined.

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'package:ffi/ffi.dart';
import 'ffi/llama_ffi.dart';

// Import configuration from the new architecture
import 'utils/llama_config.dart';

/// Enhanced service that extends your working LlamaService
class LlamaServiceEnhanced {
  late final DynamicLibrary _lib;
  late final llama_cpp _llamaCpp;

  Pointer<llama_model>? _model;
  Pointer<llama_context>? _context;
  Pointer<llama_vocab>? _vocab;
  Pointer<llama_sampler>? _sampler;
  bool _initialized = false;

  LlamaServiceEnhanced() {
    _lib = DynamicLibrary.open('llama.framework/llama');
    _llamaCpp = llama_cpp(_lib);
  }

  /// Initialize the llama.cpp backend
  void init() {
    if (_initialized) return;
    _llamaCpp.llama_backend_init();
    _initialized = true;
    print('✓ LlamaServiceEnhanced initialized');
  }

  /// Load a model from file
  bool loadModel(String modelPath, {ModelConfig? config}) {
    config ??= const ModelConfig();

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
    modelParams.use_mmap = config.useMmap;
    modelParams.use_mlock = config.useMlock;
    modelParams.n_gpu_layers = config.nGpuLayers;
    modelParams.vocab_only = config.vocabOnly;
    modelParams.check_tensors = config.checkTensors;

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

      print('✓ Model loaded successfully!');

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
  bool createContext({ContextConfig? config}) {
    config ??= const ContextConfig();

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
    ctxParams.n_ctx = config.nCtx;
    ctxParams.n_threads = config.nThreads;
    ctxParams.n_threads_batch = config.nThreadsBatch;
    ctxParams.embeddings = config.embeddings;
    ctxParams.offload_kqv = config.offloadKqv;
    ctxParams.no_perf = config.noPerf;

    print('Context params:');
    print('  n_ctx: ${ctxParams.n_ctx}');
    print('  n_threads: ${ctxParams.n_threads}');
    print('  embeddings: ${ctxParams.embeddings}');

    // Create context
    _context = _llamaCpp.llama_new_context_with_model(_model!, ctxParams);

    if (_context == nullptr || _context!.address == 0) {
      print('Failed to create context');
      return false;
    }

    final actualCtx = _llamaCpp.llama_n_ctx(_context!);
    print('✓ Context created successfully! Actual n_ctx: $actualCtx');

    return true;
  }

  /// Create a configurable sampler chain
  Pointer<llama_sampler> _createConfigurableSampler(SamplerConfig config) {
    final samplerParams = _llamaCpp.llama_sampler_chain_default_params();
    samplerParams.no_perf = false;
    final chain = _llamaCpp.llama_sampler_chain_init(samplerParams);

    print('Creating sampler with config: $config');

    // Add samplers in the correct order

    // 1. Penalties first
    if (config.repeatPenalty != 1.0 || config.frequencyPenalty != 0.0 || config.presencePenalty != 0.0) {
      final penalties = _llamaCpp.llama_sampler_init_penalties(
        64, // penalty_last_n
        config.repeatPenalty,
        config.frequencyPenalty,
        config.presencePenalty,
      );
      _llamaCpp.llama_sampler_chain_add(chain, penalties);
      print('  ✓ Added penalties sampler');
    }

    // 2. Top-K filtering
    if (config.topK > 0 && config.topK < 1000) {
      final topK = _llamaCpp.llama_sampler_init_top_k(config.topK);
      _llamaCpp.llama_sampler_chain_add(chain, topK);
      print('  ✓ Added top-k sampler');
    }

    // 3. Min-P filtering
    if (config.minP > 0.0 && config.minP < 1.0) {
      final minP = _llamaCpp.llama_sampler_init_min_p(config.minP, 1);
      _llamaCpp.llama_sampler_chain_add(chain, minP);
      print('  ✓ Added min-p sampler');
    }

    // 4. Top-P filtering
    if (config.topP < 1.0 && config.topP > 0.0) {
      final topP = _llamaCpp.llama_sampler_init_top_p(config.topP, 1);
      _llamaCpp.llama_sampler_chain_add(chain, topP);
      print('  ✓ Added top-p sampler');
    }

    // 5. Temperature scaling
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
    print('  ✓ Added final sampler');

    return chain;
  }

  /// Enhanced generate method with full configuration support
  Future<String?> generateEnhanced(String prompt, {SamplerConfig? config}) async {
    config ??= const SamplerConfig();

    if (_context == null || _context!.address == 0) {
      print('Must create context first');
      return null;
    }

    print('\n=== Enhanced Generation ===');
    print('Prompt: "$prompt"');
    print('Config: $config');

    // Yield to UI thread
    await Future.delayed(const Duration(milliseconds: 10));

    // Tokenize the prompt
    final tokens = tokenizeText(prompt);
    if (tokens.isEmpty) {
      print('Failed to tokenize prompt');
      return null;
    }

    // Yield to UI thread
    await Future.delayed(const Duration(milliseconds: 10));

    // Create configurable sampler
    if (_sampler != null && _sampler!.address != 0) {
      _llamaCpp.llama_sampler_free(_sampler!);
    }
    _sampler = _createConfigurableSampler(config);

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
          // Yield to UI thread every few tokens to prevent blocking
          if (nDecoded % 5 == 0) {
            await Future.delayed(const Duration(milliseconds: 1));
          }

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
          print(tokenText);

          // Check for stop strings
          final currentText = result.toString();
          if (config.stopStrings.isNotEmpty) {
            for (final stopString in config.stopStrings) {
              if (currentText.contains(stopString)) {
                print('\n✓ Hit stop string: "$stopString"');
                // Truncate at stop string
                final index = currentText.indexOf(stopString);
                return currentText.substring(0, index);
              }
            }
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

  /// Simple tokenization (reuse from parent)
  List<int> tokenizeText(String text, {bool addBos = true, bool special = true}) {
    if (_model == null || _model!.address == 0) {
      throw StateError('Must load model first');
    }

    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Vocab not available');
    }

    final textPtr = text.toNativeUtf8();
    try {
      // First pass: get token count
      final tokenCount = -_llamaCpp.llama_tokenize(
        _vocab!,
        textPtr.cast<Char>(),
        text.length,
        nullptr,
        0,
        addBos,
        special,
      );

      if (tokenCount <= 0) return [];

      // Second pass: get tokens
      final tokensPtr = malloc<llama_token>(tokenCount);
      try {
        final actualCount = _llamaCpp.llama_tokenize(
          _vocab!,
          textPtr.cast<Char>(),
          text.length,
          tokensPtr,
          tokenCount,
          addBos,
          special,
        );

        if (actualCount < 0) return [];

        final tokens = <int>[];
        for (int i = 0; i < actualCount; i++) {
          tokens.add(tokensPtr[i]);
        }

        return tokens;
      } finally {
        malloc.free(tokensPtr);
      }
    } finally {
      malloc.free(textPtr);
    }
  }

  /// Simple detokenization
  String detokenize(int token) {
    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Vocab not available');
    }

    final bufferPtr = malloc<Int8>(128);
    try {
      final length = _llamaCpp.llama_token_to_piece(_vocab!, token, bufferPtr.cast<Char>(), 128, 0, true);

      if (length <= 0) return '';

      final bytes = bufferPtr.cast<Uint8>().asTypedList(length);
      return String.fromCharCodes(bytes);
    } finally {
      malloc.free(bufferPtr);
    }
  }

  /// Get special tokens
  int getBosToken() {
    if (_vocab == null) throw StateError('Vocab not available');
    return _llamaCpp.llama_token_bos(_vocab!);
  }

  int getEosToken() {
    if (_vocab == null) throw StateError('Vocab not available');
    return _llamaCpp.llama_token_eos(_vocab!);
  }

  /// Generate JSON with optimized approach (without grammar for now)
  Future<String?> generateJson(String prompt, {JsonConfig? jsonConfig, SamplerConfig? samplerConfig}) async {
    jsonConfig ??= const JsonConfig();
    samplerConfig ??= const SamplerConfig();

    if (_context == null || _vocab == null) {
      print('Must create context first');
      return null;
    }

    print('\n=== JSON Generation (Optimized) ===');

    // Use a more effective prompt format that's clearer about what we want
    final formattedPrompt = '''Create a JSON object based on the following prompt: $prompt
          Only return the JSON object, no other text.
{''';

    print('Formatted prompt: "$formattedPrompt"');

    try {
      // Use very conservative sampling for JSON generation
      final jsonSamplerConfig = SamplerConfig(
        temperature: 0.01, // Almost deterministic
        topP: 0.5,
        topK: 5, // Very focused
        repeatPenalty: 2.0, // Very strong repeat penalty
        frequencyPenalty: 0.5, // Strong frequency penalty
        presencePenalty: 0.3,
        maxTokens: math.min(samplerConfig.maxTokens, 150), // Limit tokens for JSON
        stopStrings: ['\n\n', 'User:', 'Respond', '```', '//', '/*'],
      );

      print('Using optimized JSON sampling: $jsonSamplerConfig');

      // Generate without grammar constraints (they seem to be causing issues)
      final result = await generateEnhanced(formattedPrompt, config: jsonSamplerConfig);

      if (result != null) {
        // Since we started with '{', we need to reconstruct the JSON
        String cleanedResult = '{${result.trim()}';

        // Remove comments and fix common issues
        cleanedResult = _cleanJsonText(cleanedResult);

        // Try to extract a complete JSON object
        cleanedResult = _extractCompleteJson(cleanedResult);

        if (jsonConfig.strictMode) {
          // Validate JSON
          try {
            final decoded = jsonDecode(cleanedResult);
            print('✓ Generated valid JSON');

            if (jsonConfig.prettyPrint) {
              return const JsonEncoder.withIndent('  ').convert(decoded);
            }
            return cleanedResult;
          } catch (e) {
            print('✗ Generated invalid JSON: $e');
            print('Raw result: "$result"');
            print('Cleaned result: "$cleanedResult"');

            // Try to fix common JSON issues
            String fixedJson = _tryFixJson(cleanedResult);
            try {
              final decoded = jsonDecode(fixedJson);
              print('✓ Fixed and validated JSON');
              return jsonConfig.prettyPrint ? const JsonEncoder.withIndent('  ').convert(decoded) : fixedJson;
            } catch (e2) {
              print('✗ Could not fix JSON: $e2');
              return jsonConfig.strictMode ? null : cleanedResult;
            }
          }
        }

        return cleanedResult;
      }

      return null;
    } catch (e) {
      print('JSON generation error: $e');
      return null;
    }
  }

  /// Clean JSON text by removing comments and fixing common issues
  String _cleanJsonText(String json) {
    String cleaned = json;

    // Remove comments (// style) - this was the main issue!
    cleaned = cleaned.replaceAll(RegExp(r'//.*$', multiLine: true), '');

    // Remove /* */ style comments
    cleaned = cleaned.replaceAll(RegExp(r'/\*.*?\*/', multiLine: true, dotAll: true), '');

    // Remove template placeholders like <NAME>, <|user's name here>
    cleaned = cleaned.replaceAll(RegExp(r'<\|[^>]*\|>'), 'placeholder');
    cleaned = cleaned.replaceAll(RegExp(r'<[^>]*>'), 'placeholder');

    // Fix malformed field names (missing quotes, colons)
    cleaned = cleaned.replaceAll(RegExp(r'"price_incl:([^,}]+)'), r'"price": $1');
    cleaned = cleaned.replaceAll(RegExp(r'(\w+)_incl:'), r'"$1":');

    // Remove any explanatory text in parentheses
    cleaned = cleaned.replaceAll(RegExp(r'\([^)]*\)'), '');

    // Clean up whitespace
    cleaned = cleaned.replaceAll(RegExp(r'\s+'), ' ');

    return cleaned;
  }

  /// Extract a complete JSON object from text
  String _extractCompleteJson(String text) {
    String result = text.trim();

    // Find the first complete JSON object
    int braceCount = 0;
    int startIndex = -1;
    int endIndex = -1;
    bool inString = false;
    bool escaped = false;

    for (int i = 0; i < result.length; i++) {
      final char = result[i];

      if (escaped) {
        escaped = false;
        continue;
      }

      if (char == '\\' && inString) {
        escaped = true;
        continue;
      }

      if (char == '"') {
        inString = !inString;
        continue;
      }

      if (!inString) {
        if (char == '{') {
          if (startIndex == -1) startIndex = i;
          braceCount++;
        } else if (char == '}') {
          braceCount--;
          if (braceCount == 0 && startIndex != -1) {
            endIndex = i + 1;
            break;
          }
        }
      }
    }

    if (startIndex != -1 && endIndex != -1) {
      result = result.substring(startIndex, endIndex);
    }

    return result;
  }

  /// Try to fix common JSON formatting issues
  String _tryFixJson(String json) {
    String fixed = json.trim();

    // Remove any trailing commas before closing braces/brackets
    fixed = fixed.replaceAll(RegExp(r',(\s*[}\]])'), r'$1');

    // Fix double quotes issue
    fixed = fixed.replaceAll('""', '"');

    // Fix missing quotes around field names
    fixed = fixed.replaceAll(RegExp(r'(\w+)(\s*:)'), r'"\1"$2');

    // Fix single quotes to double quotes
    fixed = fixed.replaceAll("'", '"');

    // Fix trailing commas
    fixed = fixed.replaceAll(RegExp(r',(\s*})'), r'$1');

    // Ensure proper closing
    if (!fixed.endsWith('}') && !fixed.endsWith(']')) {
      if (fixed.contains('{')) {
        fixed += '}';
      } else if (fixed.contains('[')) {
        fixed += ']';
      }
    }

    return fixed;
  }

  /// Stream generation with real-time updates (better UX)
  Stream<String> generateStream(String prompt, {SamplerConfig? config}) async* {
    config ??= const SamplerConfig();

    if (_context == null || _context!.address == 0) {
      yield 'Error: Must create context first';
      return;
    }

    print('\n=== Streaming Generation ===');
    print('Prompt: "$prompt"');

    try {
      // Tokenize the prompt
      await Future.delayed(const Duration(milliseconds: 10));
      final tokens = tokenizeText(prompt);
      if (tokens.isEmpty) {
        yield 'Error: Failed to tokenize prompt';
        return;
      }

      // Create configurable sampler
      if (_sampler != null && _sampler!.address != 0) {
        _llamaCpp.llama_sampler_free(_sampler!);
      }
      _sampler = _createConfigurableSampler(config);

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
          yield 'Error: Failed to decode prompt';
          return;
        }
        print('✓ Prompt decoded');

        // Generate tokens one by one
        int nDecoded = 0;
        int nPos = nPrompt;
        final result = StringBuffer();

        final singleTokenPtr = malloc<llama_token>();
        try {
          while (nDecoded < config.maxTokens && nPos < _llamaCpp.llama_n_ctx(_context!)) {
            // Sample next token
            final newToken = _llamaCpp.llama_sampler_sample(_sampler!, _context!, -1);

            // Check if end of generation
            if (_llamaCpp.llama_token_is_eog(_vocab!, newToken)) {
              print('\n✓ Hit end-of-generation token');
              break;
            }

            // Detokenize and yield the token
            final tokenText = detokenize(newToken);
            result.write(tokenText);
            yield tokenText; // Stream the token immediately

            // Check for stop strings
            final currentText = result.toString();
            if (config.stopStrings.isNotEmpty) {
              for (final stopString in config.stopStrings) {
                if (currentText.contains(stopString)) {
                  print('\n✓ Hit stop string: "$stopString"');
                  return; // Stop streaming
                }
              }
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

            // Small delay to prevent UI blocking
            await Future.delayed(const Duration(milliseconds: 1));
          }
        } finally {
          malloc.free(singleTokenPtr);
        }

        print('\n\n=== Streaming complete ===');
        print('Generated $nDecoded tokens');
      } finally {
        malloc.free(tokensPtr);
      }
    } catch (e) {
      yield 'Error: $e';
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

    print('✓ LlamaServiceEnhanced disposed');
  }
}
