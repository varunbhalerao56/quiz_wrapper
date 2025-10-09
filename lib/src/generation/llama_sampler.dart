/// lib/src/generation/llama_sampler.dart
///
/// Sampler management and grammar-constrained generation.
/// Provides advanced sampling control and JSON schema enforcement.

// ignore_for_file: dangling_library_doc_comments

import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:quiz_wrapper/src/ffi/llama_ffi.dart';
import 'package:quiz_wrapper/src/core/llama_exceptions.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';
import 'package:quiz_wrapper/src/utils/llama_helpers.dart';

/// Sampler parameters for createSamplerChain
///
/// Simplified parameters that work independently of your existing SamplerConfig.
/// Use these when creating a sampler chain with LlamaSampler.
class SamplerParams {
  final int seed;
  final int topK;
  final double topP;
  final double minP;
  final double temp;
  final double penaltyRepeat;
  final double penaltyFreq;
  final double penaltyPresent;
  final int penaltyLastN;

  const SamplerParams({
    this.seed = 0,
    this.topK = 40,
    this.topP = 0.95,
    this.minP = 0.05,
    this.temp = 0.8,
    this.penaltyRepeat = 1.0,
    this.penaltyFreq = 0.0,
    this.penaltyPresent = 0.0,
    this.penaltyLastN = 64,
  });

  /// Deterministic sampling - always picks highest probability
  static const deterministic = SamplerParams(temp: 0.0, topK: 1, topP: 1.0, minP: 0.0);

  /// Creative sampling - high randomness
  static const creative = SamplerParams(temp: 0.9, topP: 0.95, topK: 40, minP: 0.05);

  /// Balanced sampling - moderate randomness
  static const balanced = SamplerParams(temp: 0.7, topP: 0.9, topK: 40, minP: 0.05);

  /// Precise sampling - low randomness
  static const precise = SamplerParams(temp: 0.3, topP: 0.85, topK: 20, minP: 0.1);

  /// Copy with modifications
  SamplerParams copyWith({
    int? seed,
    int? topK,
    double? topP,
    double? minP,
    double? temp,
    double? penaltyRepeat,
    double? penaltyFreq,
    double? penaltyPresent,
    int? penaltyLastN,
  }) {
    return SamplerParams(
      seed: seed ?? this.seed,
      topK: topK ?? this.topK,
      topP: topP ?? this.topP,
      minP: minP ?? this.minP,
      temp: temp ?? this.temp,
      penaltyRepeat: penaltyRepeat ?? this.penaltyRepeat,
      penaltyFreq: penaltyFreq ?? this.penaltyFreq,
      penaltyPresent: penaltyPresent ?? this.penaltyPresent,
      penaltyLastN: penaltyLastN ?? this.penaltyLastN,
    );
  }
}

/// Sampler manager for creating and managing llama.cpp sampler chains
///
/// Usage:
/// ```dart
/// final sampler = LlamaSampler(llamaCpp);
/// final chain = sampler.createSamplerChain(
///   vocab: vocab,
///   params: SamplerParams.precise,
///   grammar: GrammarConfig.quizJson,  // Force valid quiz JSON!
/// );
/// ```
class LlamaSampler with DisposableMixin {
  final llama_cpp _llamaCpp;
  Pointer<llama_sampler>? _sampler;

  LlamaSampler(this._llamaCpp);

  /// Create a sampler chain with optional grammar constraint
  ///
  /// The sampler chain processes tokens in order:
  /// 1. Distribution sampling (seed-based randomness)
  /// 2. Top-K filtering (keep only top K tokens)
  /// 3. Top-P filtering (nucleus sampling)
  /// 4. Min-P filtering (minimum probability threshold)
  /// 5. Temperature scaling (randomness control)
  /// 6. Grammar constraint (if provided - FORCES valid output)
  /// 7. Repetition penalties (prevent repetition)
  ///
  /// [vocab] - The vocabulary from the model
  /// [params] - Sampling parameters (use presets or custom)
  /// [grammar] - Optional grammar to constrain output (100% valid JSON)
  Pointer<llama_sampler> createSamplerChain({
    required Pointer<llama_vocab> vocab,
    SamplerParams params = const SamplerParams(),
    GrammarConfig? grammar,
  }) {
    checkNotDisposed('createSamplerChain');

    try {
      // Initialize sampler chain
      final sparams = _llamaCpp.llama_sampler_chain_default_params();
      sparams.no_perf = false;
      final chain = _llamaCpp.llama_sampler_chain_init(sparams);

      // 1. Distribution sampling (provides base randomness)
      _llamaCpp.llama_sampler_chain_add(chain, _llamaCpp.llama_sampler_init_dist(params.seed));

      // 2. Top-K sampling (keep only K most likely tokens)
      _llamaCpp.llama_sampler_chain_add(chain, _llamaCpp.llama_sampler_init_top_k(params.topK));

      // 3. Top-P / Nucleus sampling (cumulative probability threshold)
      _llamaCpp.llama_sampler_chain_add(chain, _llamaCpp.llama_sampler_init_top_p(params.topP, 1));

      // 4. Min-P sampling (minimum probability threshold)
      _llamaCpp.llama_sampler_chain_add(chain, _llamaCpp.llama_sampler_init_min_p(params.minP, 1));

      // 5. Temperature sampling (scale randomness)
      _llamaCpp.llama_sampler_chain_add(chain, _llamaCpp.llama_sampler_init_temp(params.temp));

      // 6. Grammar constraint (if provided - this FORCES valid structure)
      if (grammar != null && grammar.grammarStr.isNotEmpty) {
        final grammarSampler = _createGrammarSampler(vocab, grammar);
        if (grammarSampler != nullptr) {
          _llamaCpp.llama_sampler_chain_add(chain, grammarSampler);
          LlamaLogger.info('✓ Grammar constraint applied: ${grammar.grammarRoot}');
        }
      }

      // 7. Repetition penalties (prevent repetitive output)
      _llamaCpp.llama_sampler_chain_add(
        chain,
        _llamaCpp.llama_sampler_init_penalties(
          params.penaltyLastN,
          params.penaltyRepeat,
          params.penaltyFreq,
          params.penaltyPresent,
        ),
      );

      _sampler = chain;
      LlamaLogger.info('✓ Sampler chain created successfully');
      return chain;
    } catch (e) {
      throw LlamaException('Failed to create sampler chain: $e', operation: 'createSamplerChain');
    }
  }

  /// Internal: Create grammar sampler (standard or lazy)
  Pointer<llama_sampler> _createGrammarSampler(Pointer<llama_vocab> vocab, GrammarConfig grammar) {
    final grammarStrPtr = grammar.grammarStr.toNativeUtf8().cast<Char>();
    final grammarRootPtr = grammar.grammarRoot.toNativeUtf8().cast<Char>();

    try {
      // Lazy grammar (with trigger words)
      if (grammar.lazy && grammar.triggerWords != null && grammar.triggerWords!.isNotEmpty) {
        return _createLazyGrammar(vocab, grammarStrPtr, grammarRootPtr, grammar.triggerWords!);
      }

      // Standard grammar (always active)
      final grammarSampler = _llamaCpp.llama_sampler_init_grammar(vocab, grammarStrPtr, grammarRootPtr);

      if (grammarSampler == nullptr) {
        throw LlamaException('Failed to initialize grammar sampler');
      }

      return grammarSampler;
    } finally {
      malloc.free(grammarStrPtr);
      malloc.free(grammarRootPtr);
    }
  }

  /// Internal: Create lazy grammar sampler (activates on trigger words)
  Pointer<llama_sampler> _createLazyGrammar(
    Pointer<llama_vocab> vocab,
    Pointer<Char> grammarStr,
    Pointer<Char> grammarRoot,
    List<String> triggerWords,
  ) {
    final numWords = triggerWords.length;
    final triggerWordsPtr = malloc<Pointer<Char>>(numWords);

    try {
      // Convert trigger words to native pointers
      for (int i = 0; i < numWords; i++) {
        triggerWordsPtr[i] = triggerWords[i].toNativeUtf8().cast<Char>();
      }

      final grammarSampler = _llamaCpp.llama_sampler_init_grammar_lazy(
        vocab,
        grammarStr,
        grammarRoot,
        triggerWordsPtr,
        numWords,
        nullptr, // trigger_tokens (not used)
        0,
      );

      if (grammarSampler == nullptr) {
        throw LlamaException('Failed to initialize lazy grammar sampler');
      }

      return grammarSampler;
    } finally {
      // Free trigger word strings
      for (int i = 0; i < numWords; i++) {
        malloc.free(triggerWordsPtr[i]);
      }
      malloc.free(triggerWordsPtr);
    }
  }

  /// Get the current sampler chain (if created)
  Pointer<llama_sampler>? get sampler => _sampler;

  /// Check if sampler has been created
  bool get hasSampler => _sampler != null && _sampler != nullptr;

  void dispose() {
    if (_sampler != null && _sampler != nullptr) {
      _llamaCpp.llama_sampler_free(_sampler!);
      _sampler = null;
      LlamaLogger.info('✓ Sampler disposed');
    }
    markDisposed();
  }
}

/// Helper extension for creating samplers from existing presets
extension SamplerParamsExtension on SamplerParams {
  /// Convert to a human-readable description
  String describe() {
    return 'SamplerParams(temp: $temp, topK: $topK, topP: $topP, minP: $minP)';
  }

  /// Check if this is deterministic sampling
  bool get isDeterministic => temp <= 0.0 || topK == 1;

  /// Check if this is high-temperature (creative) sampling
  bool get isCreative => temp >= 0.85;
}
