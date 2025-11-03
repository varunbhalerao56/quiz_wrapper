/// Enhanced LlamaService that extends the working version with new features
///
/// This provides immediate access to the new sampling and JSON features
/// while the full modular architecture is being refined.

// ignore_for_file: dangling_library_doc_comments

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:quiz_wrapper/src/generation/llama_sampler.dart';
import 'package:quiz_wrapper/src/utils/llama_helpers.dart';
import 'ffi/llama_ffi.dart';
import 'utils/llama_config.dart';

// Toggle debug printing
const bool _kDebugMode = true;

void _debugPrint(String message) {
  if (_kDebugMode) {
    // ignore: avoid_print
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
  LlamaSampler? _samplerManager;
  PerformanceMonitor? _perfMonitor;
  bool _initialized = false;

  LlamaServiceEnhanced() {
    _lib = DynamicLibrary.open('libllama.dylib');
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

    _samplerManager = LlamaSampler(_llamaCpp);
    _debugPrint('✓ Sampler manager initialized');

    _perfMonitor = PerformanceMonitor(_llamaCpp);
    _debugPrint('✓ Performance monitor initialized');

    final actualCtx = _llamaCpp.llama_n_ctx(_context!);
    _debugPrint('✓ Context created! Actual n_ctx: $actualCtx');

    return true;
  }

  /// Get model metadata information
  /// Get model metadata information
  LlamaModelInfo getModelInfo() {
    if (_model == null || _model!.address == 0) {
      throw StateError('Model not loaded. Call loadModel() first.');
    }

    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Vocab not available. Call loadModel() first.');
    }

    try {
      _debugPrint('Extracting model info...');

      // Extract metadata using llama.cpp FFI functions
      // Use _vocab for vocab size (not _model!)
      final vocabSize = _llamaCpp.llama_vocab_n_tokens(_vocab!);
      final contextSize = _llamaCpp.llama_model_n_ctx_train(_model!);
      final embeddingSize = _llamaCpp.llama_n_embd(_model!);
      final numLayers = _llamaCpp.llama_n_layer(_model!);

      // Get model architecture name
      final archBuffer = malloc<Char>(256);
      try {
        _llamaCpp.llama_model_desc(_model!, archBuffer, 256);
        final architecture = archBuffer.cast<Utf8>().toDartString();

        // Estimate parameter count (rough approximation)
        final numParams = embeddingSize * embeddingSize * numLayers * 12;

        final info = LlamaModelInfo(
          vocabSize: vocabSize,
          contextSize: contextSize,
          embeddingSize: embeddingSize,
          numLayers: numLayers,
          architecture: architecture,
          numParams: numParams,
        );

        _debugPrint('✓ Model info extracted: $info');
        return info;
      } finally {
        malloc.free(archBuffer);
      }
    } catch (e) {
      _debugPrint('✗ Failed to extract model info: $e');
      rethrow;
    }
  }

  /// Create a configurable sampler chain
  /// Create sampler chain using new LlamaSampler (with optional grammar)
  /// Create sampler chain using new LlamaSampler (with optional grammar)
  Pointer<llama_sampler> _createConfigurableSampler(SamplerConfig config) {
    _debugPrint('Creating sampler chain with config: $config');

    if (_samplerManager == null) {
      throw StateError('Sampler manager not initialized');
    }

    // Convert SamplerConfig to SamplerParams
    final params = SamplerParams(
      seed: config.seed == -1 ? DateTime.now().millisecondsSinceEpoch : config.seed,
      temp: config.temperature,
      topK: config.topK,
      topP: config.topP,
      minP: config.minP,
      penaltyRepeat: config.repeatPenalty,
      penaltyFreq: config.frequencyPenalty,
      penaltyPresent: config.presencePenalty,
    );

    // Get context size for DRY sampler
    final nCtxTrain = _llamaCpp.llama_model_n_ctx_train(_model!);

    // Create sampler with optional grammar and context size
    return _samplerManager!.createSamplerChain(
      vocab: _vocab!,
      params: params,
      grammar: config.grammar,
      nCtxTrain: nCtxTrain, // ✅ Pass context size for DRY sampler
    );
  }

  /// Enhanced generate method with full configuration support
  Future<GenerationResult?> generateEnhanced(
    String prompt, {
    SamplerConfig? config,
    String? systemPrompt = 'You are a helpful, concise assistant.',
  }) async {
    config ??= const SamplerConfig();

    final stopChecker = StopStringChecker(config.stopStrings);

    if (_context == null || _context!.address == 0) {
      _debugPrint('✗ Cannot generate: context not created');
      return null;
    }

    _debugPrint('\n=== Enhanced Generation ===');
    _debugPrint('Prompt: "$prompt"');
    _debugPrint('Max tokens: ${config.maxTokens}, Temperature: ${config.temperature}');

    // Oloma
    final formattedPrompt =
        '''
<|begin_of_text|>
<|system|>
$systemPrompt
<|user|>
$prompt
<|assistant|>
''';

    // Tokenize the prompt
    final tokens = tokenizeText(formattedPrompt);
    if (tokens.isEmpty) {
      _debugPrint('✗ Failed to tokenize prompt');
      return null;
    }

    _debugPrint('✓ Tokenized into ${tokens.length} tokens');

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
          if (stopChecker.hasStopStrings) {
            final currentText = result.toString();
            final checkResult = stopChecker.checkAndClean(currentText);

            if (checkResult.stopped) {
              _debugPrint('✓ Hit stop string: "${checkResult.stopString}" at position $nDecoded');

              // Get metrics before returning
              final metrics = _perfMonitor?.getContextPerformance(_context!);
              _debugPrint('$metrics');

              return GenerationResult(checkResult.text, nDecoded, metrics);
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

      // ✅ ADD THIS: Get metrics from llama.cpp
      final metrics = _perfMonitor?.getContextPerformance(_context!);
      _debugPrint('$metrics');

      return GenerationResult(result.toString(), nDecoded, metrics);
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

      // Convert the C buffer to a Dart byte list
      final bytes = bufferPtr.cast<Uint8>().asTypedList(length);

      // ✅ Decode as UTF-8 instead of Latin-1
      return utf8.decode(bytes, allowMalformed: true);
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

  /// Stream generation with real-time updates
  Stream<StreamEvent> generateStream(
    String prompt, {
    SamplerConfig? config,
    String? systemPrompt = 'You are a helpful, concise assistant.',
  }) async* {
    config ??= const SamplerConfig();

    final stopChecker = StopStringChecker(config.stopStrings);

    if (_context == null || _context!.address == 0) {
      _debugPrint('✗ Cannot stream: context not created');
      return;
    }

    _debugPrint('\n=== Streaming Generation ===');
    _debugPrint('Prompt: "$prompt"');

    _debugPrint('systemPrompt: $systemPrompt');

    // Oloma
    final formattedPrompt =
        '''
<|begin_of_text|>
<|system|>
$systemPrompt
<|user|>
$prompt
<|assistant|>
''';

    try {
      final tokens = tokenizeText(formattedPrompt);
      if (tokens.isEmpty) {
        _debugPrint('✗ Failed to tokenize prompt');
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
            yield TokenEvent(tokenText); // Yield token event

            // Check stop strings
            if (stopChecker.hasStopStrings) {
              final currentText = result.toString();
              final checkResult = stopChecker.checkAndClean(currentText);

              if (checkResult.stopped) {
                _debugPrint('✓ Streaming complete: stop string "${checkResult.stopString}" at $nDecoded');

                // ✅ ADD THESE 4 LINES:
                final metrics = _perfMonitor?.getContextPerformance(_context!);
                if (metrics != null) {
                  yield MetricsEvent(metrics); // ✅ Yield metrics before done
                }

                yield DoneEvent(nDecoded); // Yield done event with count
                return;
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
          }

          _debugPrint('=== Streaming complete: $nDecoded tokens ===');
          final metrics = _perfMonitor?.getContextPerformance(_context!);
          if (metrics != null) {
            yield MetricsEvent(metrics); // ✅ Yield metrics before done
          }

          yield DoneEvent(nDecoded); // Yield final done event
        } finally {
          malloc.free(singleTokenPtr);
        }
      } finally {
        malloc.free(tokensPtr);
      }
    } catch (e) {
      _debugPrint('✗ Streaming error: $e');
    }
  }

  /// Clean up resources
  void dispose() {
    _debugPrint('Disposing resources...');

    if (_samplerManager != null) {
      _samplerManager!.dispose();
      _samplerManager = null;
      _debugPrint('  ✓ Sampler manager disposed');
    }

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
