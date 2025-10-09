/// Advanced sampling methods for LlamaService
///
/// Provides Mirostat 1.0/2.0, DRY repetition control, Typical sampling,
/// XTC sampling, and other advanced sampling techniques.

// ignore_for_file: dangling_library_doc_comments

import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:quiz_wrapper/src/ffi/llama_ffi.dart';
import 'package:quiz_wrapper/src/core/llama_exceptions.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';
import 'package:flutter/foundation.dart';

/// Advanced sampling configuration
class AdvancedSamplerConfig {
  // Mirostat parameters
  final bool useMirostat;
  final int mirostatVersion; // 1 or 2
  final double mirostatTau;
  final double mirostatEta;
  final int mirostatM; // Only for v1

  // DRY (Don't Repeat Yourself) parameters
  final bool useDry;
  final double dryMultiplier;
  final double dryBase;
  final int dryAllowedLength;
  final int dryPenaltyLastN;
  final List<String> drySequenceBreakers;

  // Typical sampling
  final bool useTypical;
  final double typicalP;
  final int typicalMinKeep;

  // XTC sampling
  final bool useXtc;
  final double xtcProbability;
  final double xtcThreshold;
  final int xtcMinKeep;

  const AdvancedSamplerConfig({
    // Mirostat
    this.useMirostat = false,
    this.mirostatVersion = 2,
    this.mirostatTau = 5.0,
    this.mirostatEta = 0.1,
    this.mirostatM = 100,

    // DRY
    this.useDry = false,
    this.dryMultiplier = 0.8,
    this.dryBase = 1.75,
    this.dryAllowedLength = 2,
    this.dryPenaltyLastN = 256,
    this.drySequenceBreakers = const ['\n', '.', '!', '?', ':', ';'],

    // Typical
    this.useTypical = false,
    this.typicalP = 0.95,
    this.typicalMinKeep = 1,

    // XTC
    this.useXtc = false,
    this.xtcProbability = 0.1,
    this.xtcThreshold = 0.1,
    this.xtcMinKeep = 1,
  });

  /// Preset for high-quality creative writing
  static const AdvancedSamplerConfig creative = AdvancedSamplerConfig(
    useDry: true,
    dryMultiplier: 0.8,
    dryBase: 1.75,
    useTypical: true,
    typicalP: 0.95,
  );

  /// Preset for consistent, coherent output
  static const AdvancedSamplerConfig coherent = AdvancedSamplerConfig(
    useMirostat: true,
    mirostatVersion: 2,
    mirostatTau: 5.0,
    mirostatEta: 0.1,
    useDry: true,
  );

  /// Preset for exploration and variety
  static const AdvancedSamplerConfig exploratory = AdvancedSamplerConfig(
    useXtc: true,
    xtcProbability: 0.1,
    xtcThreshold: 0.1,
    useTypical: true,
    typicalP: 0.9,
  );
}

/// Advanced sampler builder that creates complex sampler chains
class AdvancedSamplerBuilder {
  final llama_cpp _llamaCpp;
  final Pointer<llama_vocab> _vocab;

  AdvancedSamplerBuilder(this._llamaCpp, this._vocab);

  /// Create an advanced sampler chain
  Pointer<llama_sampler> createAdvancedSampler({
    required SamplerConfig basicConfig,
    AdvancedSamplerConfig? advancedConfig,
  }) {
    advancedConfig ??= const AdvancedSamplerConfig();

    // Validate parameters
    LlamaSafety.validateSamplingParams(
      temperature: basicConfig.temperature,
      topP: basicConfig.topP,
      topK: basicConfig.topK,
      minP: basicConfig.minP,
      operation: 'createAdvancedSampler',
    );

    final samplerParams = _llamaCpp.llama_sampler_chain_default_params();
    samplerParams.no_perf = false;
    final chain = _llamaCpp.llama_sampler_chain_init(samplerParams);

    debugPrint('Creating advanced sampler chain:');
    _logSamplerConfig(basicConfig, advancedConfig);

    // Build sampler chain in correct order
    _addPenaltySamplers(chain, basicConfig);
    _addDrySampler(chain, advancedConfig);
    _addFilteringSamplers(chain, basicConfig);
    _addAdvancedSamplers(chain, advancedConfig);
    _addTemperatureSampler(chain, basicConfig);
    _addFinalSampler(chain, basicConfig, advancedConfig);

    return chain;
  }

  /// Add penalty-based samplers (repetition, frequency, presence)
  void _addPenaltySamplers(Pointer<llama_sampler> chain, SamplerConfig config) {
    if (config.repeatPenalty != 1.0 || config.frequencyPenalty != 0.0 || config.presencePenalty != 0.0) {
      final penalties = _llamaCpp.llama_sampler_init_penalties(
        64, // penalty_last_n
        config.repeatPenalty,
        config.frequencyPenalty,
        config.presencePenalty,
      );
      _llamaCpp.llama_sampler_chain_add(chain, penalties);
      debugPrint('  ✓ Added penalties sampler');
    }
  }

  /// Add DRY (Don't Repeat Yourself) sampler
  void _addDrySampler(Pointer<llama_sampler> chain, AdvancedSamplerConfig config) {
    if (!config.useDry) return;

    try {
      // Convert sequence breakers to C string array
      final breakersPtr = malloc<Pointer<Char>>(config.drySequenceBreakers.length);
      final breakerPtrs = <Pointer<Utf8>>[];

      try {
        for (int i = 0; i < config.drySequenceBreakers.length; i++) {
          final breakerPtr = config.drySequenceBreakers[i].toNativeUtf8();
          breakerPtrs.add(breakerPtr);
          breakersPtr[i] = breakerPtr.cast<Char>();
        }

        final drySampler = _llamaCpp.llama_sampler_init_dry(
          _vocab,
          2048, // n_ctx_train (would get from model)
          config.dryMultiplier,
          config.dryBase,
          config.dryAllowedLength,
          config.dryPenaltyLastN,
          breakersPtr,
          config.drySequenceBreakers.length,
        );

        _llamaCpp.llama_sampler_chain_add(chain, drySampler);
        debugPrint('  ✓ Added DRY sampler');
      } finally {
        for (final ptr in breakerPtrs) {
          malloc.free(ptr);
        }
        malloc.free(breakersPtr);
      }
    } catch (e) {
      debugPrint('  ✗ Failed to add DRY sampler: $e');
    }
  }

  /// Add filtering samplers (Top-K, Top-P, Min-P)
  void _addFilteringSamplers(Pointer<llama_sampler> chain, SamplerConfig config) {
    // Top-K filtering
    if (config.topK > 0 && config.topK < 1000) {
      final topK = _llamaCpp.llama_sampler_init_top_k(config.topK);
      _llamaCpp.llama_sampler_chain_add(chain, topK);
      debugPrint('  ✓ Added top-k sampler');
    }

    // Min-P filtering
    if (config.minP > 0.0 && config.minP < 1.0) {
      final minP = _llamaCpp.llama_sampler_init_min_p(config.minP, 1);
      _llamaCpp.llama_sampler_chain_add(chain, minP);
      debugPrint('  ✓ Added min-p sampler');
    }

    // Top-P (nucleus) filtering
    if (config.topP < 1.0 && config.topP > 0.0) {
      final topP = _llamaCpp.llama_sampler_init_top_p(config.topP, 1);
      _llamaCpp.llama_sampler_chain_add(chain, topP);
      debugPrint('  ✓ Added top-p sampler');
    }
  }

  /// Add advanced sampling methods
  void _addAdvancedSamplers(Pointer<llama_sampler> chain, AdvancedSamplerConfig config) {
    // Typical sampling
    if (config.useTypical) {
      final typical = _llamaCpp.llama_sampler_init_typical(config.typicalP, config.typicalMinKeep);
      _llamaCpp.llama_sampler_chain_add(chain, typical);
      debugPrint('  ✓ Added typical sampler');
    }

    // XTC sampling
    if (config.useXtc) {
      final xtc = _llamaCpp.llama_sampler_init_xtc(
        config.xtcProbability,
        config.xtcThreshold,
        config.xtcMinKeep,
        config.useXtc ? DateTime.now().millisecondsSinceEpoch : 0,
      );
      _llamaCpp.llama_sampler_chain_add(chain, xtc);
      debugPrint('  ✓ Added XTC sampler');
    }
  }

  /// Add temperature scaling
  void _addTemperatureSampler(Pointer<llama_sampler> chain, SamplerConfig config) {
    if (config.temperature != 1.0) {
      final temp = _llamaCpp.llama_sampler_init_temp(config.temperature);
      _llamaCpp.llama_sampler_chain_add(chain, temp);
      debugPrint('  ✓ Added temperature sampler');
    }
  }

  /// Add final sampling method (Mirostat or distribution/greedy)
  void _addFinalSampler(Pointer<llama_sampler> chain, SamplerConfig basicConfig, AdvancedSamplerConfig advancedConfig) {
    if (advancedConfig.useMirostat) {
      // Use Mirostat instead of regular sampling
      final vocabSize = _llamaCpp.llama_n_vocab(_vocab);

      if (advancedConfig.mirostatVersion == 1) {
        final mirostat = _llamaCpp.llama_sampler_init_mirostat(
          vocabSize,
          basicConfig.seed == -1 ? DateTime.now().millisecondsSinceEpoch : basicConfig.seed,
          advancedConfig.mirostatTau,
          advancedConfig.mirostatEta,
          advancedConfig.mirostatM,
        );
        _llamaCpp.llama_sampler_chain_add(chain, mirostat);
        debugPrint('  ✓ Added Mirostat v1 sampler');
      } else {
        final mirostat = _llamaCpp.llama_sampler_init_mirostat_v2(
          basicConfig.seed == -1 ? DateTime.now().millisecondsSinceEpoch : basicConfig.seed,
          advancedConfig.mirostatTau,
          advancedConfig.mirostatEta,
        );
        _llamaCpp.llama_sampler_chain_add(chain, mirostat);
        debugPrint('  ✓ Added Mirostat v2 sampler');
      }
    } else {
      // Regular distribution or greedy sampling
      final finalSampler = basicConfig.temperature == 0.0
          ? _llamaCpp.llama_sampler_init_greedy()
          : _llamaCpp.llama_sampler_init_dist(
              basicConfig.seed == -1 ? DateTime.now().millisecondsSinceEpoch : basicConfig.seed,
            );

      _llamaCpp.llama_sampler_chain_add(chain, finalSampler);
      debugPrint('  ✓ Added final sampler (${basicConfig.temperature == 0.0 ? "greedy" : "distribution"})');
    }
  }

  /// Log sampler configuration for debugging
  void _logSamplerConfig(SamplerConfig basic, AdvancedSamplerConfig advanced) {
    debugPrint('  Basic: temp=${basic.temperature}, top_p=${basic.topP}, top_k=${basic.topK}');
    debugPrint('  Penalties: repeat=${basic.repeatPenalty}, freq=${basic.frequencyPenalty}');

    if (advanced.useMirostat) {
      debugPrint('  Mirostat v${advanced.mirostatVersion}: tau=${advanced.mirostatTau}, eta=${advanced.mirostatEta}');
    }

    if (advanced.useDry) {
      debugPrint('  DRY: mult=${advanced.dryMultiplier}, base=${advanced.dryBase}');
    }

    if (advanced.useTypical) {
      debugPrint('  Typical: p=${advanced.typicalP}');
    }

    if (advanced.useXtc) {
      debugPrint('  XTC: prob=${advanced.xtcProbability}, thresh=${advanced.xtcThreshold}');
    }
  }
}

/// Sampler presets for different use cases
class SamplerPresets {
  /// High-quality creative writing with DRY and typical sampling
  static const creative = (
    basic: SamplerConfig(temperature: 0.8, topP: 0.95, topK: 50, repeatPenalty: 1.05),
    advanced: AdvancedSamplerConfig(useDry: true, dryMultiplier: 0.8, dryBase: 1.75, useTypical: true, typicalP: 0.95),
  );

  /// Coherent, consistent output with Mirostat
  static const coherent = (
    basic: SamplerConfig(temperature: 0.7, topP: 0.9, repeatPenalty: 1.1),
    advanced: AdvancedSamplerConfig(
      useMirostat: true,
      mirostatVersion: 2,
      mirostatTau: 5.0,
      mirostatEta: 0.1,
      useDry: true,
    ),
  );

  /// Exploratory output with XTC and variety
  static const exploratory = (
    basic: SamplerConfig(temperature: 0.9, topP: 0.95, topK: 60, repeatPenalty: 1.05),
    advanced: AdvancedSamplerConfig(
      useXtc: true,
      xtcProbability: 0.1,
      xtcThreshold: 0.1,
      useTypical: true,
      typicalP: 0.9,
    ),
  );

  /// Precise, focused output for Q&A
  static const precise = (
    basic: SamplerConfig(temperature: 0.3, topP: 0.8, topK: 20, repeatPenalty: 1.15),
    advanced: AdvancedSamplerConfig(useMirostat: true, mirostatVersion: 2, mirostatTau: 3.0, mirostatEta: 0.05),
  );

  /// Deterministic output for testing
  static const deterministic = (
    basic: SamplerConfig(
      temperature: 0.0,
      topK: 1,
      repeatPenalty: 1.0,
      seed: 42, // Fixed seed
    ),
    advanced: AdvancedSamplerConfig(),
  );
}

/// Utility for analyzing and optimizing sampling parameters
class SamplerAnalyzer {
  /// Analyze text for repetition patterns
  static RepetitionAnalysis analyzeRepetition(String text) {
    final words = text.toLowerCase().split(RegExp(r'\W+'));
    final wordCounts = <String, int>{};

    for (final word in words) {
      if (word.isNotEmpty) {
        wordCounts[word] = (wordCounts[word] ?? 0) + 1;
      }
    }

    // Find repeated words
    final repeated = wordCounts.entries.where((e) => e.value > 1).toList()..sort((a, b) => b.value.compareTo(a.value));

    // Calculate repetition score
    final totalWords = words.length;
    final uniqueWords = wordCounts.length;
    final repetitionScore = totalWords > 0 ? 1.0 - (uniqueWords / totalWords) : 0.0;

    return RepetitionAnalysis(
      totalWords: totalWords,
      uniqueWords: uniqueWords,
      repetitionScore: repetitionScore,
      mostRepeated: repeated.take(10).toList(),
    );
  }

  /// Suggest optimal sampling parameters based on text analysis
  static SamplerConfig suggestParameters(String sampleText) {
    final analysis = analyzeRepetition(sampleText);

    // High repetition -> increase penalties and use DRY
    if (analysis.repetitionScore > 0.3) {
      return const SamplerConfig(
        temperature: 0.7,
        topP: 0.9,
        repeatPenalty: 1.2,
        frequencyPenalty: 0.1,
        presencePenalty: 0.1,
      );
    }

    // Low repetition -> can use more creative settings
    if (analysis.repetitionScore < 0.1) {
      return const SamplerConfig(temperature: 0.8, topP: 0.95, repeatPenalty: 1.05);
    }

    // Balanced
    return SamplerConfig.balanced;
  }

  /// Estimate optimal context size based on text patterns
  static int suggestContextSize(List<String> sampleTexts) {
    if (sampleTexts.isEmpty) return 2048;

    final avgLength = sampleTexts.map((t) => t.length).reduce((a, b) => a + b) / sampleTexts.length;

    // Rough estimate: 4 characters per token, with 2x buffer
    final estimatedTokens = (avgLength / 4 * 2).round();

    // Round up to common context sizes
    if (estimatedTokens <= 1024) return 1024;
    if (estimatedTokens <= 2048) return 2048;
    if (estimatedTokens <= 4096) return 4096;
    if (estimatedTokens <= 8192) return 8192;
    return 16384;
  }
}

/// Analysis result for text repetition patterns
class RepetitionAnalysis {
  final int totalWords;
  final int uniqueWords;
  final double repetitionScore; // 0.0 = no repetition, 1.0 = all repeated
  final List<MapEntry<String, int>> mostRepeated;

  const RepetitionAnalysis({
    required this.totalWords,
    required this.uniqueWords,
    required this.repetitionScore,
    required this.mostRepeated,
  });

  /// Whether text has high repetition
  bool get hasHighRepetition => repetitionScore > 0.3;

  /// Whether text has low repetition (very diverse)
  bool get hasLowRepetition => repetitionScore < 0.1;

  @override
  String toString() {
    return 'RepetitionAnalysis(total: $totalWords, unique: $uniqueWords, '
        'score: ${repetitionScore.toStringAsFixed(3)})';
  }
}
