/// Configuration management and serialization for LlamaService
///
/// Provides JSON serializable configuration classes for all llama.cpp
/// parameters and settings.

// ignore_for_file: dangling_library_doc_comments

import 'dart:convert';

import 'package:quiz_wrapper/src/utils/llama_helpers.dart';

sealed class StreamEvent {}

class TokenEvent extends StreamEvent {
  final String token;
  TokenEvent(this.token);
}

class DoneEvent extends StreamEvent {
  final int tokensGenerated;
  DoneEvent(this.tokensGenerated);
}

class MetricsEvent extends StreamEvent {
  final PerformanceMetrics metrics;
  MetricsEvent(this.metrics);
}

class GenerationResult {
  final String text;
  final int tokensGenerated;
  final PerformanceMetrics? metrics;

  GenerationResult(this.text, this.tokensGenerated, this.metrics);
}

/// Configuration for sampling parameters with JSON serialization
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
  final GrammarConfig? grammar;

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
    this.grammar,
  });

  // Preset configurations
  static const SamplerConfig creative = SamplerConfig(temperature: 0.9, topP: 0.95, topK: 50, repeatPenalty: 1.05);
  static const SamplerConfig balanced = SamplerConfig(temperature: 0.7, topP: 0.9, topK: 40, repeatPenalty: 1.1);
  static const SamplerConfig precise = SamplerConfig(temperature: 0.25, topP: 0.8, topK: 30, repeatPenalty: 1.15);
  static const SamplerConfig deterministic = SamplerConfig(temperature: 0.0, topP: 1.0, topK: 1, repeatPenalty: 1.0);

  /// Create a copy with modified parameters
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
    GrammarConfig? grammar,
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
      grammar: grammar ?? this.grammar,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'temperature': temperature,
      'topP': topP,
      'topK': topK,
      'minP': minP,
      'repeatPenalty': repeatPenalty,
      'frequencyPenalty': frequencyPenalty,
      'presencePenalty': presencePenalty,
      'stopStrings': stopStrings,
      'maxTokens': maxTokens,
      'seed': seed,
      'grammar': grammar?.toJson(),
    };
  }

  /// Create from JSON
  factory SamplerConfig.fromJson(Map<String, dynamic> json) {
    return SamplerConfig(
      temperature: (json['temperature'] as num?)?.toDouble() ?? 0.7,
      topP: (json['topP'] as num?)?.toDouble() ?? 0.9,
      topK: json['topK'] as int? ?? 40,
      minP: (json['minP'] as num?)?.toDouble() ?? 0.05,
      repeatPenalty: (json['repeatPenalty'] as num?)?.toDouble() ?? 1.1,
      frequencyPenalty: (json['frequencyPenalty'] as num?)?.toDouble() ?? 0.0,
      presencePenalty: (json['presencePenalty'] as num?)?.toDouble() ?? 0.0,
      stopStrings: (json['stopStrings'] as List?)?.cast<String>() ?? const [],
      maxTokens: json['maxTokens'] as int? ?? 100,
      seed: json['seed'] as int? ?? -1,
      grammar: json['grammar'] != null ? GrammarConfig.fromJson(json['grammar']) : null,
    );
  }

  /// Convert to JSON string
  String toJsonString() => jsonEncode(toJson());

  /// Create from JSON string
  factory SamplerConfig.fromJsonString(String jsonString) {
    return SamplerConfig.fromJson(jsonDecode(jsonString));
  }

  @override
  String toString() {
    return 'SamplerConfig(temp: $temperature, top_p: $topP, top_k: $topK, '
        'min_p: $minP, repeat: $repeatPenalty, max: $maxTokens, grammar: $grammar)';
  }
}

/// Configuration for JSON output
class JsonConfig {
  final Map<String, dynamic>? schema;
  final bool strictMode;
  final bool prettyPrint;

  const JsonConfig({this.schema, this.strictMode = true, this.prettyPrint = false});

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {'schema': schema, 'strictMode': strictMode, 'prettyPrint': prettyPrint};
  }

  /// Create from JSON
  factory JsonConfig.fromJson(Map<String, dynamic> json) {
    return JsonConfig(
      schema: json['schema'] as Map<String, dynamic>?,
      strictMode: json['strictMode'] as bool? ?? true,
      prettyPrint: json['prettyPrint'] as bool? ?? false,
    );
  }
}

/// Configuration for model parameters
class ModelConfig {
  final int nGpuLayers;
  final int mainGpu;
  final List<double>? tensorSplit;
  final bool useMmap;
  final bool useMlock;
  final bool vocabOnly;
  final bool checkTensors;
  final bool useExtraBuffers;

  const ModelConfig({
    this.nGpuLayers = 0,
    this.mainGpu = 0,
    this.tensorSplit,
    this.useMmap = true,
    this.useMlock = false,
    this.vocabOnly = false,
    this.checkTensors = true,
    this.useExtraBuffers = false,
  });

  /// Create a copy with modified parameters
  ModelConfig copyWith({
    int? nGpuLayers,
    int? mainGpu,
    List<double>? tensorSplit,
    bool? useMmap,
    bool? useMlock,
    bool? vocabOnly,
    bool? checkTensors,
    bool? useExtraBuffers,
  }) {
    return ModelConfig(
      nGpuLayers: nGpuLayers ?? this.nGpuLayers,
      mainGpu: mainGpu ?? this.mainGpu,
      tensorSplit: tensorSplit ?? this.tensorSplit,
      useMmap: useMmap ?? this.useMmap,
      useMlock: useMlock ?? this.useMlock,
      vocabOnly: vocabOnly ?? this.vocabOnly,
      checkTensors: checkTensors ?? this.checkTensors,
      useExtraBuffers: useExtraBuffers ?? this.useExtraBuffers,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'nGpuLayers': nGpuLayers,
      'mainGpu': mainGpu,
      'tensorSplit': tensorSplit,
      'useMmap': useMmap,
      'useMlock': useMlock,
      'vocabOnly': vocabOnly,
      'checkTensors': checkTensors,
      'useExtraBuffers': useExtraBuffers,
    };
  }

  /// Create from JSON
  factory ModelConfig.fromJson(Map<String, dynamic> json) {
    return ModelConfig(
      nGpuLayers: json['nGpuLayers'] as int? ?? 0,
      mainGpu: json['mainGpu'] as int? ?? 0,
      tensorSplit: (json['tensorSplit'] as List?)?.cast<double>(),
      useMmap: json['useMmap'] as bool? ?? true,
      useMlock: json['useMlock'] as bool? ?? false,
      vocabOnly: json['vocabOnly'] as bool? ?? false,
      checkTensors: json['checkTensors'] as bool? ?? true,
      useExtraBuffers: json['useExtraBuffers'] as bool? ?? false,
    );
  }
}

/// Configuration for context parameters
class ContextConfig {
  final int nCtx;
  final int nBatch;
  final int nUbatch;
  final int nSeqMax;
  final int nThreads;
  final int nThreadsBatch;
  final int ropeScalingType;
  final int poolingType;
  final int attentionType;
  final double ropeFreqBase;
  final double ropeFreqScale;
  final double yarnExtFactor;
  final double yarnAttnFactor;
  final double yarnBetaFast;
  final double yarnBetaSlow;
  final int yarnOrigCtx;
  final double defragThold;
  final bool embeddings;
  final bool offloadKqv;
  final bool noPerf;

  const ContextConfig({
    this.nCtx = 2048,
    this.nBatch = 512,
    this.nUbatch = 512,
    this.nSeqMax = 1,
    this.nThreads = 4,
    this.nThreadsBatch = 4,
    this.ropeScalingType = 0,
    this.poolingType = 0,
    this.attentionType = 0,
    this.ropeFreqBase = 10000.0,
    this.ropeFreqScale = 1.0,
    this.yarnExtFactor = 1.0,
    this.yarnAttnFactor = 1.0,
    this.yarnBetaFast = 32.0,
    this.yarnBetaSlow = 1.0,
    this.yarnOrigCtx = 0,
    this.defragThold = -1.0,
    this.embeddings = false,
    this.offloadKqv = false,
    this.noPerf = false,
  });

  /// Create a copy with modified parameters
  ContextConfig copyWith({
    int? nCtx,
    int? nBatch,
    int? nUbatch,
    int? nSeqMax,
    int? nThreads,
    int? nThreadsBatch,
    int? ropeScalingType,
    int? poolingType,
    int? attentionType,
    double? ropeFreqBase,
    double? ropeFreqScale,
    double? yarnExtFactor,
    double? yarnAttnFactor,
    double? yarnBetaFast,
    double? yarnBetaSlow,
    int? yarnOrigCtx,
    double? defragThold,
    bool? embeddings,
    bool? offloadKqv,
    bool? noPerf,
  }) {
    return ContextConfig(
      nCtx: nCtx ?? this.nCtx,
      nBatch: nBatch ?? this.nBatch,
      nUbatch: nUbatch ?? this.nUbatch,
      nSeqMax: nSeqMax ?? this.nSeqMax,
      nThreads: nThreads ?? this.nThreads,
      nThreadsBatch: nThreadsBatch ?? this.nThreadsBatch,
      ropeScalingType: ropeScalingType ?? this.ropeScalingType,
      poolingType: poolingType ?? this.poolingType,
      attentionType: attentionType ?? this.attentionType,
      ropeFreqBase: ropeFreqBase ?? this.ropeFreqBase,
      ropeFreqScale: ropeFreqScale ?? this.ropeFreqScale,
      yarnExtFactor: yarnExtFactor ?? this.yarnExtFactor,
      yarnAttnFactor: yarnAttnFactor ?? this.yarnAttnFactor,
      yarnBetaFast: yarnBetaFast ?? this.yarnBetaFast,
      yarnBetaSlow: yarnBetaSlow ?? this.yarnBetaSlow,
      yarnOrigCtx: yarnOrigCtx ?? this.yarnOrigCtx,
      defragThold: defragThold ?? this.defragThold,
      embeddings: embeddings ?? this.embeddings,
      offloadKqv: offloadKqv ?? this.offloadKqv,
      noPerf: noPerf ?? this.noPerf,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'nCtx': nCtx,
      'nBatch': nBatch,
      'nUbatch': nUbatch,
      'nSeqMax': nSeqMax,
      'nThreads': nThreads,
      'nThreadsBatch': nThreadsBatch,
      'ropeScalingType': ropeScalingType,
      'poolingType': poolingType,
      'attentionType': attentionType,
      'ropeFreqBase': ropeFreqBase,
      'ropeFreqScale': ropeFreqScale,
      'yarnExtFactor': yarnExtFactor,
      'yarnAttnFactor': yarnAttnFactor,
      'yarnBetaFast': yarnBetaFast,
      'yarnBetaSlow': yarnBetaSlow,
      'yarnOrigCtx': yarnOrigCtx,
      'defragThold': defragThold,
      'embeddings': embeddings,
      'offloadKqv': offloadKqv,
      'noPerf': noPerf,
    };
  }

  /// Create from JSON
  factory ContextConfig.fromJson(Map<String, dynamic> json) {
    return ContextConfig(
      nCtx: json['nCtx'] as int? ?? 2048,
      nBatch: json['nBatch'] as int? ?? 512,
      nUbatch: json['nUbatch'] as int? ?? 512,
      nSeqMax: json['nSeqMax'] as int? ?? 1,
      nThreads: json['nThreads'] as int? ?? 4,
      nThreadsBatch: json['nThreadsBatch'] as int? ?? 4,
      ropeScalingType: json['ropeScalingType'] as int? ?? 0,
      poolingType: json['poolingType'] as int? ?? 0,
      attentionType: json['attentionType'] as int? ?? 0,
      ropeFreqBase: (json['ropeFreqBase'] as num?)?.toDouble() ?? 10000.0,
      ropeFreqScale: (json['ropeFreqScale'] as num?)?.toDouble() ?? 1.0,
      yarnExtFactor: (json['yarnExtFactor'] as num?)?.toDouble() ?? 1.0,
      yarnAttnFactor: (json['yarnAttnFactor'] as num?)?.toDouble() ?? 1.0,
      yarnBetaFast: (json['yarnBetaFast'] as num?)?.toDouble() ?? 32.0,
      yarnBetaSlow: (json['yarnBetaSlow'] as num?)?.toDouble() ?? 1.0,
      yarnOrigCtx: json['yarnOrigCtx'] as int? ?? 0,
      defragThold: (json['defragThold'] as num?)?.toDouble() ?? -1.0,
      embeddings: json['embeddings'] as bool? ?? false,
      offloadKqv: json['offloadKqv'] as bool? ?? false,
      noPerf: json['noPerf'] as bool? ?? false,
    );
  }
}

/// Configuration for embeddings
class EmbeddingsConfig {
  final bool normalize;
  final int poolingType;
  final bool addSpecialTokens;

  const EmbeddingsConfig({
    this.normalize = true,
    this.poolingType = 1, // MEAN pooling
    this.addSpecialTokens = false,
  });

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {'normalize': normalize, 'poolingType': poolingType, 'addSpecialTokens': addSpecialTokens};
  }

  /// Create from JSON
  factory EmbeddingsConfig.fromJson(Map<String, dynamic> json) {
    return EmbeddingsConfig(
      normalize: json['normalize'] as bool? ?? true,
      poolingType: json['poolingType'] as int? ?? 1,
      addSpecialTokens: json['addSpecialTokens'] as bool? ?? false,
    );
  }
}

/// Complete configuration for LlamaService
class LlamaServiceConfig {
  final ModelConfig model;
  final ContextConfig context;
  final SamplerConfig sampler;
  final EmbeddingsConfig embeddings;
  final bool verboseLogging;

  const LlamaServiceConfig({
    this.model = const ModelConfig(),
    this.context = const ContextConfig(),
    this.sampler = const SamplerConfig(),
    this.embeddings = const EmbeddingsConfig(),
    this.verboseLogging = false,
  });

  /// Create a copy with modified parameters
  LlamaServiceConfig copyWith({
    ModelConfig? model,
    ContextConfig? context,
    SamplerConfig? sampler,
    EmbeddingsConfig? embeddings,
    bool? verboseLogging,
  }) {
    return LlamaServiceConfig(
      model: model ?? this.model,
      context: context ?? this.context,
      sampler: sampler ?? this.sampler,
      embeddings: embeddings ?? this.embeddings,
      verboseLogging: verboseLogging ?? this.verboseLogging,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'model': model.toJson(),
      'context': context.toJson(),
      'sampler': sampler.toJson(),
      'embeddings': embeddings.toJson(),
      'verboseLogging': verboseLogging,
    };
  }

  /// Create from JSON
  factory LlamaServiceConfig.fromJson(Map<String, dynamic> json) {
    return LlamaServiceConfig(
      model: ModelConfig.fromJson(json['model'] ?? {}),
      context: ContextConfig.fromJson(json['context'] ?? {}),
      sampler: SamplerConfig.fromJson(json['sampler'] ?? {}),
      embeddings: EmbeddingsConfig.fromJson(json['embeddings'] ?? {}),
      verboseLogging: json['verboseLogging'] as bool? ?? false,
    );
  }

  /// Convert to JSON string
  String toJsonString() => jsonEncode(toJson());

  /// Create from JSON string
  factory LlamaServiceConfig.fromJsonString(String jsonString) {
    return LlamaServiceConfig.fromJson(jsonDecode(jsonString));
  }
}

/// Grammar configuration for constrained generation
///
/// Grammars use GBNF (GGML BNF) notation to define valid output structures.
/// This ensures the model can ONLY generate text matching the grammar rules.
class GrammarConfig {
  final String grammarStr;
  final String grammarRoot;
  final bool lazy;
  final List<String>? triggerWords;

  const GrammarConfig({required this.grammarStr, this.grammarRoot = 'root', this.lazy = false, this.triggerWords});

  /// Standard JSON grammar - ensures valid JSON output
  static const json = GrammarConfig(
    grammarRoot: 'root',
    grammarStr: '''
root ::= object
object ::= "{" ws members? ws "}"
members ::= pair (ws "," ws pair)*
pair ::= string ws ":" ws value
value ::= string | number | object | array | "true" | "false" | "null"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" escape
escape ::= ["\\\\/bfnrt] | "u" [0-9a-fA-F]{4}
number ::= integer fraction? exponent?
integer ::= "-"? ("0" | [1-9] [0-9]*)
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
ws ::= [ \\t\\n\\r]*
''',
  );

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {'grammarStr': grammarStr, 'grammarRoot': grammarRoot, 'lazy': lazy, 'triggerWords': triggerWords};
  }

  /// Create from JSON
  factory GrammarConfig.fromJson(Map<String, dynamic> json) {
    return GrammarConfig(
      grammarStr: json['grammarStr'],
      grammarRoot: json['grammarRoot'],
      lazy: json['lazy'],
      triggerWords: json['triggerWords'],
    );
  }

  /// Convert to JSON string
  String toJsonString() => jsonEncode(toJson());

  /// Create from JSON string
  factory GrammarConfig.fromJsonString(String jsonString) {
    return GrammarConfig.fromJson(jsonDecode(jsonString));
  }

  @override
  String toString() {
    return 'GrammarConfig(grammarStr: $grammarStr, grammarRoot: $grammarRoot, lazy: $lazy, triggerWords: $triggerWords)';
  }

  /// Create a copy with modified parameters
  GrammarConfig copyWith({String? grammarStr, String? grammarRoot, bool? lazy, List<String>? triggerWords}) {
    return GrammarConfig(
      grammarStr: grammarStr ?? this.grammarStr,
      grammarRoot: grammarRoot ?? this.grammarRoot,
      lazy: lazy ?? this.lazy,
      triggerWords: triggerWords ?? this.triggerWords,
    );
  }

  /// Create custom grammar from string
  factory GrammarConfig.custom({
    required String rules,
    String root = 'root',
    bool lazy = false,
    List<String>? triggerWords,
  }) {
    return GrammarConfig(grammarStr: rules, grammarRoot: root, lazy: lazy, triggerWords: triggerWords);
  }
}
