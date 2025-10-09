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
import 'utils/llama_config.dart';

// Toggle debug printing
const bool _kDebugMode = true;

void _debugPrint(String message) {
  if (_kDebugMode) {
    print('[LlamaService] $message');
  }
}

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
    _debugPrint('Instance created');
  }

  /// Initialize the llama.cpp backend
  void init() {
    if (_initialized) {
      _debugPrint('Already initialized, skipping');
      return;
    }
    _llamaCpp.llama_backend_init();
    _initialized = true;
    _debugPrint('✓ Backend initialized');
  }

  /// Load a model from file
  bool loadModel(String modelPath, {ModelConfig? config}) {
    config ??= const ModelConfig();

    if (!_initialized) {
      throw StateError('Must call init() first');
    }

    if (!File(modelPath).existsSync()) {
      _debugPrint('✗ Model file not found: $modelPath');
      return false;
    }

    _debugPrint('Loading model from: $modelPath');

    // Get default model parameters
    final modelParams = _llamaCpp.llama_model_default_params();

    // Configure model parameters
    modelParams.use_mmap = config.useMmap;
    modelParams.use_mlock = config.useMlock;
    modelParams.n_gpu_layers = config.nGpuLayers;
    modelParams.vocab_only = config.vocabOnly;
    modelParams.check_tensors = config.checkTensors;

    _debugPrint(
      'Model config: mmap=${modelParams.use_mmap}, mlock=${modelParams.use_mlock}, gpu_layers=${modelParams.n_gpu_layers}',
    );

    // Load model
    final pathPtr = modelPath.toNativeUtf8();
    try {
      _model = _llamaCpp.llama_load_model_from_file(pathPtr.cast<Char>(), modelParams);

      if (_model == nullptr || _model!.address == 0) {
        _debugPrint('✗ Failed to load model');
        return false;
      }

      _debugPrint('✓ Model loaded successfully');

      // Get vocab handle for tokenization
      _vocab = _llamaCpp.llama_model_get_vocab(_model!);
      if (_vocab == nullptr || _vocab!.address == 0) {
        _debugPrint('⚠ Warning: Failed to get vocab handle');
      } else {
        _debugPrint('✓ Vocab handle acquired');
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
      _debugPrint('✗ Cannot create context: model not loaded');
      return false;
    }

    if (_context != null && _context!.address != 0) {
      _debugPrint('⚠ Context already exists');
      return false;
    }

    _debugPrint('Creating context with n_ctx=${config.nCtx}, n_threads=${config.nThreads}');

    // Get default context parameters
    final ctxParams = _llamaCpp.llama_context_default_params();

    // Set our custom parameters
    ctxParams.n_ctx = config.nCtx;
    ctxParams.n_threads = config.nThreads;
    ctxParams.n_threads_batch = config.nThreadsBatch;
    ctxParams.embeddings = config.embeddings;
    ctxParams.offload_kqv = config.offloadKqv;
    ctxParams.no_perf = config.noPerf;

    // Create context
    _context = _llamaCpp.llama_new_context_with_model(_model!, ctxParams);

    if (_context == nullptr || _context!.address == 0) {
      _debugPrint('✗ Failed to create context');
      return false;
    }

    final actualCtx = _llamaCpp.llama_n_ctx(_context!);
    _debugPrint('✓ Context created! Actual n_ctx: $actualCtx');

    return true;
  }

  /// Create a configurable sampler chain
  Pointer<llama_sampler> _createConfigurableSampler(SamplerConfig config) {
    _debugPrint('Creating sampler chain with config: $config');

    final samplerParams = _llamaCpp.llama_sampler_chain_default_params();
    samplerParams.no_perf = false;
    final chain = _llamaCpp.llama_sampler_chain_init(samplerParams);

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
      _debugPrint('  ✓ Added penalties sampler');
    }

    // 2. Top-K filtering
    if (config.topK > 0 && config.topK < 1000) {
      final topK = _llamaCpp.llama_sampler_init_top_k(config.topK);
      _llamaCpp.llama_sampler_chain_add(chain, topK);
      _debugPrint('  ✓ Added top-k=${config.topK} sampler');
    }

    // 3. Min-P filtering
    if (config.minP > 0.0 && config.minP < 1.0) {
      final minP = _llamaCpp.llama_sampler_init_min_p(config.minP, 1);
      _llamaCpp.llama_sampler_chain_add(chain, minP);
      _debugPrint('  ✓ Added min-p=${config.minP} sampler');
    }

    // 4. Top-P filtering
    if (config.topP < 1.0 && config.topP > 0.0) {
      final topP = _llamaCpp.llama_sampler_init_top_p(config.topP, 1);
      _llamaCpp.llama_sampler_chain_add(chain, topP);
      _debugPrint('  ✓ Added top-p=${config.topP} sampler');
    }

    // 5. Temperature scaling
    if (config.temperature != 1.0) {
      final temp = _llamaCpp.llama_sampler_init_temp(config.temperature);
      _llamaCpp.llama_sampler_chain_add(chain, temp);
      _debugPrint('  ✓ Added temperature=${config.temperature} sampler');
    }

    // 6. Final sampling method
    final finalSampler = config.temperature == 0.0
        ? _llamaCpp.llama_sampler_init_greedy()
        : _llamaCpp.llama_sampler_init_dist(config.seed == -1 ? DateTime.now().millisecondsSinceEpoch : config.seed);

    _llamaCpp.llama_sampler_chain_add(chain, finalSampler);
    _debugPrint('  ✓ Added final sampler (${config.temperature == 0.0 ? "greedy" : "dist"})');

    return chain;
  }

  /// Enhanced generate method with full configuration support
  Future<String?> generateEnhanced(String prompt, {SamplerConfig? config}) async {
    config ??= const SamplerConfig();

    if (_context == null || _context!.address == 0) {
      _debugPrint('✗ Cannot generate: context not created');
      return null;
    }

    _debugPrint('\n=== Enhanced Generation ===');
    _debugPrint('Prompt: "$prompt"');
    _debugPrint('Max tokens: ${config.maxTokens}, Temperature: ${config.temperature}');

    // Yield to UI thread
    await Future.delayed(const Duration(milliseconds: 10));

    // Tokenize the prompt
    final tokens = tokenizeText(prompt);
    if (tokens.isEmpty) {
      _debugPrint('✗ Failed to tokenize prompt');
      return null;
    }

    _debugPrint('✓ Tokenized into ${tokens.length} tokens');

    // Yield to UI thread
    await Future.delayed(const Duration(milliseconds: 10));

    // Create configurable sampler
    if (_sampler != null && _sampler!.address != 0) {
      _llamaCpp.llama_sampler_free(_sampler!);
    }
    _sampler = _createConfigurableSampler(config);

    final nPrompt = tokens.length;

    // Create batch for prompt
    final tokensPtr = malloc<llama_token>(nPrompt);
    for (int i = 0; i < nPrompt; i++) {
      tokensPtr[i] = tokens[i];
    }

    try {
      var batch = _llamaCpp.llama_batch_get_one(tokensPtr, nPrompt);

      _debugPrint('Decoding prompt batch...');
      if (_llamaCpp.llama_decode(_context!, batch) != 0) {
        _debugPrint('✗ Failed to decode prompt');
        return null;
      }
      _debugPrint('✓ Prompt decoded');

      // Generate tokens
      final result = StringBuffer();
      int nDecoded = 0;
      int nPos = nPrompt;

      final singleTokenPtr = malloc<llama_token>();
      try {
        while (nDecoded < config.maxTokens && nPos < _llamaCpp.llama_n_ctx(_context!)) {
          // Yield to UI thread every few tokens
          if (nDecoded % 5 == 0) {
            await Future.delayed(const Duration(milliseconds: 1));
          }

          // Sample next token
          final newToken = _llamaCpp.llama_sampler_sample(_sampler!, _context!, -1);

          // Check if end of generation
          if (_llamaCpp.llama_token_is_eog(_vocab!, newToken)) {
            _debugPrint('\n✓ Hit end-of-generation token at position $nDecoded');
            break;
          }

          // Detokenize and add to result
          final tokenText = detokenize(newToken);
          result.write(tokenText);

          // Check for stop strings
          if (config.stopStrings.isNotEmpty) {
            final currentText = result.toString();
            for (final stopString in config.stopStrings) {
              if (currentText.contains(stopString)) {
                _debugPrint('✓ Hit stop string: "$stopString" at position $nDecoded');
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
            _debugPrint('✗ Failed to decode token at position $nDecoded');
            break;
          }

          nDecoded++;
          nPos++;
        }
      } finally {
        malloc.free(singleTokenPtr);
      }

      _debugPrint('=== Generation complete: $nDecoded tokens ===\n');

      return result.toString();
    } finally {
      malloc.free(tokensPtr);
    }
  }

  /// Tokenization
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

      if (tokenCount <= 0) {
        _debugPrint('⚠ Tokenization returned count: $tokenCount');
        return [];
      }

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

        if (actualCount < 0) {
          _debugPrint('⚠ Tokenization failed with code: $actualCount');
          return [];
        }

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

  /// Detokenization
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

  /// JSON generation with explicit rules - let model generate complete JSON
  Future<String?> generateJson(String prompt, {JsonConfig? jsonConfig, SamplerConfig? samplerConfig}) async {
    jsonConfig ??= const JsonConfig();
    samplerConfig ??= const SamplerConfig();

    if (_context == null || _vocab == null) {
      _debugPrint('✗ Cannot generate JSON: context/vocab not available');
      return null;
    }

    _debugPrint('\n=== JSON Generation ===');
    _debugPrint('User task: "$prompt"');

    // GEMMA CHAT TEMPLATE with explicit JSON rules
    final formattedPrompt =
        '''<start_of_turn>user
Task: $prompt

Generate a valid JSON object following these rules:

JSON STRUCTURE:
- Start with { and end with }
- Use key-value pairs separated by commas
- Keys must be strings in double quotes
- No trailing comma after last item

DATA TYPES:
- String: "text in double quotes"
- Number: 42 or 3.14 (no quotes)
- Boolean: true or false (no quotes)
- Null: null (no quotes)
- Object: {"nested": "data"}
- Array: [1, 2, 3]

CRITICAL:
- Use ONLY double quotes, never single quotes
- No comments allowed
- All keys must be quoted
- No trailing commas

Output only the JSON, nothing else.
<end_of_turn>
<start_of_turn>model
''';

    _debugPrint('Using Gemma template with explicit JSON rules');

    try {
      // Balanced sampling
      final jsonSamplerConfig = SamplerConfig(
        temperature: 0.3,
        topP: 0.9,
        topK: 40,
        repeatPenalty: 1.1,
        frequencyPenalty: 0.0,
        presencePenalty: 0.0,
        maxTokens: 8000, // Increased for rules + JSON
        stopStrings: ['<end_of_turn>', '<start_of_turn>', '\n\n\n'],
      );

      _debugPrint('Sampler: temp=${jsonSamplerConfig.temperature}, maxTokens=${jsonSamplerConfig.maxTokens}');

      final result = await generateEnhanced(formattedPrompt, config: jsonSamplerConfig);

      if (result == null) {
        _debugPrint('✗ Generation returned null');
        return null;
      }

      _debugPrint('Raw output (${result.length} chars):');
      _debugPrint(result);

      // Minimal cleanup - just trim whitespace
      final cleaned = result.trim();

      // Try to parse
      try {
        final decoded = jsonDecode(cleaned);
        _debugPrint('✓ Valid JSON!');

        if (jsonConfig.prettyPrint) {
          return const JsonEncoder.withIndent('  ').convert(decoded);
        }
        return cleaned;
      } catch (e) {
        // _debugPrint('✗ Invalid JSON: $e');
        // _debugPrint('Cleaned output: "$cleaned"');

        // if (jsonConfig.strictMode) {
        //   _debugPrint('Strict mode: returning null');
        //   return null;
        // }

        // _debugPrint('Non-strict mode: returning raw output');
        return cleaned;
      }
    } catch (e) {
      _debugPrint('✗ Error: $e');
      return null;
    }
  }

  /// Stream generation with real-time updates
  Stream<String> generateStream(String prompt, {SamplerConfig? config}) async* {
    config ??= const SamplerConfig();

    if (_context == null || _context!.address == 0) {
      _debugPrint('✗ Cannot stream: context not created');
      yield 'Error: Must create context first';
      return;
    }

    _debugPrint('\n=== Streaming Generation ===');
    _debugPrint('Prompt: "$prompt"');

    try {
      await Future.delayed(const Duration(milliseconds: 10));
      final tokens = tokenizeText(prompt);
      if (tokens.isEmpty) {
        _debugPrint('✗ Failed to tokenize prompt');
        yield 'Error: Failed to tokenize';
        return;
      }

      _debugPrint('✓ Tokenized into ${tokens.length} tokens');

      // Create sampler
      if (_sampler != null && _sampler!.address != 0) {
        _llamaCpp.llama_sampler_free(_sampler!);
      }
      _sampler = _createConfigurableSampler(config);

      final nPrompt = tokens.length;

      // Create batch
      final tokensPtr = malloc<llama_token>(nPrompt);
      for (int i = 0; i < nPrompt; i++) {
        tokensPtr[i] = tokens[i];
      }

      try {
        var batch = _llamaCpp.llama_batch_get_one(tokensPtr, nPrompt);

        _debugPrint('Decoding prompt...');
        if (_llamaCpp.llama_decode(_context!, batch) != 0) {
          _debugPrint('✗ Failed to decode prompt');
          yield 'Error: Decode failed';
          return;
        }
        _debugPrint('✓ Prompt decoded, starting generation...');

        int nDecoded = 0;
        int nPos = nPrompt;
        final result = StringBuffer();

        final singleTokenPtr = malloc<llama_token>();
        try {
          while (nDecoded < config.maxTokens && nPos < _llamaCpp.llama_n_ctx(_context!)) {
            final newToken = _llamaCpp.llama_sampler_sample(_sampler!, _context!, -1);

            if (_llamaCpp.llama_token_is_eog(_vocab!, newToken)) {
              _debugPrint('✓ Streaming complete: EOG token at $nDecoded');
              break;
            }

            final tokenText = detokenize(newToken);
            result.write(tokenText);
            yield tokenText; // Stream immediately

            // Check stop strings
            if (config.stopStrings.isNotEmpty) {
              final currentText = result.toString();
              for (final stopString in config.stopStrings) {
                if (currentText.contains(stopString)) {
                  _debugPrint('✓ Streaming complete: stop string "$stopString" at $nDecoded');
                  return;
                }
              }
            }

            singleTokenPtr[0] = newToken;
            batch = _llamaCpp.llama_batch_get_one(singleTokenPtr, 1);

            if (_llamaCpp.llama_decode(_context!, batch) != 0) {
              _debugPrint('✗ Decode failed at token $nDecoded');
              break;
            }

            nDecoded++;
            nPos++;

            await Future.delayed(const Duration(milliseconds: 1));
          }

          _debugPrint('=== Streaming complete: $nDecoded tokens ===');
        } finally {
          malloc.free(singleTokenPtr);
        }
      } finally {
        malloc.free(tokensPtr);
      }
    } catch (e) {
      _debugPrint('✗ Streaming error: $e');
      yield 'Error: $e';
    }
  }

  /// Clean up resources
  void dispose() {
    _debugPrint('Disposing resources...');

    if (_sampler != null && _sampler!.address != 0) {
      _llamaCpp.llama_sampler_free(_sampler!);
      _sampler = null;
      _debugPrint('  ✓ Sampler freed');
    }

    if (_context != null && _context!.address != 0) {
      _llamaCpp.llama_free(_context!);
      _context = null;
      _debugPrint('  ✓ Context freed');
    }

    if (_model != null && _model!.address != 0) {
      _llamaCpp.llama_model_free(_model!);
      _model = null;
      _debugPrint('  ✓ Model freed');
    }

    if (_initialized) {
      _llamaCpp.llama_backend_free();
      _initialized = false;
      _debugPrint('  ✓ Backend freed');
    }

    _debugPrint('✓ Disposal complete');
  }
}
