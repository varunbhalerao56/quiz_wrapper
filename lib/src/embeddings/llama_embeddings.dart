/// Embeddings support for LlamaService
///
/// Provides text embeddings computation for RAG (Retrieval Augmented Generation),
/// semantic search, similarity calculations, and vector operations.

// ignore_for_file: dangling_library_doc_comments

import 'dart:ffi';
import 'dart:math' as math;
import 'package:ffi/ffi.dart';
import 'package:quiz_wrapper/src/ffi/llama_ffi.dart';
import 'package:quiz_wrapper/src/core/llama_exceptions.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';
import 'package:flutter/foundation.dart';

/// Result of an embeddings computation
class EmbeddingResult {
  final List<double> embedding;
  final int tokenCount;
  final double computeTimeMs;
  final bool normalized;

  const EmbeddingResult({
    required this.embedding,
    required this.tokenCount,
    required this.computeTimeMs,
    required this.normalized,
  });

  /// Dimensionality of the embedding
  int get dimensions => embedding.length;

  /// Calculate cosine similarity with another embedding
  double cosineSimilarity(EmbeddingResult other) {
    if (embedding.length != other.embedding.length) {
      throw ArgumentError('Embedding dimensions must match');
    }

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < embedding.length; i++) {
      dotProduct += embedding[i] * other.embedding[i];
      normA += embedding[i] * embedding[i];
      normB += other.embedding[i] * other.embedding[i];
    }

    if (normA == 0.0 || normB == 0.0) return 0.0;

    return dotProduct / (math.sqrt(normA) * math.sqrt(normB));
  }

  /// Calculate Euclidean distance to another embedding
  double euclideanDistance(EmbeddingResult other) {
    if (embedding.length != other.embedding.length) {
      throw ArgumentError('Embedding dimensions must match');
    }

    double sum = 0.0;
    for (int i = 0; i < embedding.length; i++) {
      final diff = embedding[i] - other.embedding[i];
      sum += diff * diff;
    }

    return math.sqrt(sum);
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'embedding': embedding,
      'tokenCount': tokenCount,
      'computeTimeMs': computeTimeMs,
      'normalized': normalized,
      'dimensions': dimensions,
    };
  }

  /// Create from JSON
  factory EmbeddingResult.fromJson(Map<String, dynamic> json) {
    return EmbeddingResult(
      embedding: (json['embedding'] as List).cast<double>(),
      tokenCount: json['tokenCount'] as int,
      computeTimeMs: (json['computeTimeMs'] as num).toDouble(),
      normalized: json['normalized'] as bool,
    );
  }
}

/// Service for computing text embeddings using llama.cpp
class LlamaEmbeddingsService with DisposableMixin {
  final llama_cpp _llamaCpp;
  final Pointer<llama_model> _model;
  final Pointer<llama_vocab> _vocab;

  Pointer<llama_context>? _embeddingContext;
  int? _embeddingDimensions;

  LlamaEmbeddingsService(this._llamaCpp, this._model, this._vocab);

  /// Initialize embeddings context
  Future<bool> initialize(ContextConfig config) async {
    checkNotDisposed('initialize');

    try {
      // Create context with embeddings enabled
      final ctxParams = _llamaCpp.llama_context_default_params();

      // Configure for embeddings
      ctxParams.n_ctx = config.nCtx;
      ctxParams.n_threads = config.nThreads;
      ctxParams.n_threads_batch = config.nThreadsBatch;
      ctxParams.embeddings = true; // KEY: Enable embeddings
      ctxParams.pooling_typeAsInt = config.poolingType;
      ctxParams.attention_typeAsInt = config.attentionType;
      ctxParams.offload_kqv = config.offloadKqv;
      ctxParams.no_perf = config.noPerf;

      debugPrint('Creating embeddings context...');
      _embeddingContext = _llamaCpp.llama_new_context_with_model(_model, ctxParams);

      if (_embeddingContext == nullptr || _embeddingContext!.address == 0) {
        throw LlamaEmbeddingsException('Failed to create embeddings context');
      }

      // Get embedding dimensions
      _embeddingDimensions = _llamaCpp.llama_n_embd(_model);
      if (_embeddingDimensions! <= 0) {
        throw LlamaEmbeddingsException('Invalid embedding dimensions: $_embeddingDimensions');
      }

      debugPrint('✓ Embeddings context created');
      debugPrint('  Embedding dimensions: $_embeddingDimensions');
      debugPrint('  Context size: ${_llamaCpp.llama_n_ctx(_embeddingContext!)}');

      return true;
    } catch (e) {
      throw LlamaEmbeddingsException('Failed to initialize embeddings: $e');
    }
  }

  /// Compute embeddings for a single text
  Future<EmbeddingResult?> computeEmbedding(String text, {EmbeddingsConfig? config}) async {
    checkNotDisposed('computeEmbedding');
    config ??= const EmbeddingsConfig();

    if (_embeddingContext == null || _embeddingContext!.address == 0) {
      throw LlamaEmbeddingsException('Embeddings context not initialized');
    }

    final stopwatch = Stopwatch()..start();

    try {
      // Tokenize the text
      final tokens = _tokenizeForEmbeddings(text, config.addSpecialTokens);
      if (tokens.isEmpty) {
        throw LlamaTokenizationException(text, 'No tokens produced');
      }

      debugPrint('Computing embeddings for ${tokens.length} tokens...');

      // Create batch
      final tokensPtr = malloc<llama_token>(tokens.length);
      try {
        for (int i = 0; i < tokens.length; i++) {
          tokensPtr[i] = tokens[i];
        }

        final batch = _llamaCpp.llama_batch_get_one(tokensPtr, tokens.length);

        // Encode (not decode - we want embeddings)
        if (_llamaCpp.llama_encode(_embeddingContext!, batch) != 0) {
          throw LlamaEmbeddingsException('Failed to encode batch for embeddings');
        }

        // Get embeddings based on pooling strategy
        final embeddingPtr = _getEmbeddingPointer(config.poolingType);
        if (embeddingPtr == nullptr) {
          throw LlamaEmbeddingsException('Failed to get embedding pointer');
        }

        // Convert to Dart list
        final embeddings = <double>[];
        for (int i = 0; i < _embeddingDimensions!; i++) {
          embeddings.add(embeddingPtr[i]);
        }

        // Normalize if requested
        if (config.normalize) {
          _normalizeEmbedding(embeddings);
        }

        stopwatch.stop();

        return EmbeddingResult(
          embedding: embeddings,
          tokenCount: tokens.length,
          computeTimeMs: stopwatch.elapsedMilliseconds.toDouble(),
          normalized: config.normalize,
        );
      } finally {
        malloc.free(tokensPtr);
      }
    } catch (e) {
      stopwatch.stop();
      if (e is LlamaException) rethrow;
      throw LlamaEmbeddingsException('Embedding computation failed: $e');
    }
  }

  /// Compute embeddings for multiple texts in batch
  Future<List<EmbeddingResult?>> computeBatchEmbeddings(List<String> texts, {EmbeddingsConfig? config}) async {
    checkNotDisposed('computeBatchEmbeddings');

    final results = <EmbeddingResult?>[];

    for (int i = 0; i < texts.length; i++) {
      debugPrint('Computing embedding ${i + 1}/${texts.length}...');
      try {
        final result = await computeEmbedding(texts[i], config: config);
        results.add(result);
      } catch (e) {
        debugPrint('Failed to compute embedding for text ${i + 1}: $e');
        results.add(null);
      }
    }

    return results;
  }

  /// Find most similar texts to a query
  List<SimilarityResult> findSimilar(
    String query,
    List<String> candidates, {
    int topK = 5,
    double threshold = 0.0,
    EmbeddingsConfig? config,
  }) {
    checkNotDisposed('findSimilar');

    // This is a placeholder implementation
    // In practice, you'd compute embeddings and find similarities
    debugPrint('Finding similar texts for: "$query" among ${candidates.length} candidates');

    // Return empty results for now - would implement full similarity search
    return <SimilarityResult>[];
  }

  /// Get embedding pointer based on pooling strategy
  Pointer<Float> _getEmbeddingPointer(int poolingType) {
    // Automatic path selection: seq → ith → default

    // Try sequence-based embeddings first
    try {
      return _llamaCpp.llama_get_embeddings_seq(_embeddingContext!, 0);
    } catch (e) {
      debugPrint('Sequence embeddings failed, trying ith...');
    }

    // Try index-based embeddings
    try {
      return _llamaCpp.llama_get_embeddings_ith(_embeddingContext!, -1);
    } catch (e) {
      debugPrint('Ith embeddings failed, using default...');
    }

    // Fall back to default embeddings
    return _llamaCpp.llama_get_embeddings(_embeddingContext!);
  }

  /// Tokenize text for embeddings (may differ from generation tokenization)
  List<int> _tokenizeForEmbeddings(String text, bool addSpecialTokens) {
    final textPtr = text.toNativeUtf8();
    try {
      // First pass: get token count
      final tokenCount = -_llamaCpp.llama_tokenize(
        _vocab,
        textPtr.cast<Char>(),
        text.length,
        nullptr,
        0,
        addSpecialTokens, // BOS handling
        false, // Don't parse special tokens for embeddings
      );

      if (tokenCount <= 0) {
        return [];
      }

      // Second pass: get tokens
      final tokensPtr = malloc<llama_token>(tokenCount);
      try {
        final actualCount = _llamaCpp.llama_tokenize(
          _vocab,
          textPtr.cast<Char>(),
          text.length,
          tokensPtr,
          tokenCount,
          addSpecialTokens,
          false,
        );

        if (actualCount < 0) {
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

  /// Normalize embedding vector to unit length (L2 normalization)
  void _normalizeEmbedding(List<double> embedding) {
    double norm = 0.0;
    for (final value in embedding) {
      norm += value * value;
    }

    norm = math.sqrt(norm);
    if (norm == 0.0) return; // Avoid division by zero

    for (int i = 0; i < embedding.length; i++) {
      embedding[i] /= norm;
    }
  }

  /// Get embedding dimensions
  int? get dimensions => _embeddingDimensions;

  /// Check if embeddings are available
  bool get isAvailable => _embeddingContext != null && _embeddingContext!.address != 0 && _embeddingDimensions != null;

  void dispose() {
    if (_embeddingContext != null && _embeddingContext!.address != 0) {
      _llamaCpp.llama_free(_embeddingContext!);
      _embeddingContext = null;
    }
    _embeddingDimensions = null;
    markDisposed();
    debugPrint('✓ Embeddings service disposed');
  }
}

/// Result of a similarity search
class SimilarityResult {
  final String text;
  final double similarity;
  final int index;
  final EmbeddingResult embedding;

  const SimilarityResult({required this.text, required this.similarity, required this.index, required this.embedding});

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {'text': text, 'similarity': similarity, 'index': index, 'embedding': embedding.toJson()};
  }

  /// Create from JSON
  factory SimilarityResult.fromJson(Map<String, dynamic> json) {
    return SimilarityResult(
      text: json['text'] as String,
      similarity: (json['similarity'] as num).toDouble(),
      index: json['index'] as int,
      embedding: EmbeddingResult.fromJson(json['embedding']),
    );
  }
}

/// Utility functions for working with embeddings
class EmbeddingUtils {
  /// Calculate cosine similarity between two embedding vectors
  static double cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) {
      throw ArgumentError('Vector dimensions must match');
    }

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA == 0.0 || normB == 0.0) return 0.0;

    return dotProduct / (math.sqrt(normA) * math.sqrt(normB));
  }

  /// Calculate Euclidean distance between two embedding vectors
  static double euclideanDistance(List<double> a, List<double> b) {
    if (a.length != b.length) {
      throw ArgumentError('Vector dimensions must match');
    }

    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      final diff = a[i] - b[i];
      sum += diff * diff;
    }

    return math.sqrt(sum);
  }

  /// Normalize a vector to unit length (L2 normalization)
  static List<double> normalize(List<double> vector) {
    double norm = 0.0;
    for (final value in vector) {
      norm += value * value;
    }

    norm = math.sqrt(norm);
    if (norm == 0.0) return vector; // Avoid division by zero

    return vector.map((v) => v / norm).toList();
  }

  /// Find top-K most similar embeddings
  static List<SimilarityResult> findTopSimilar(
    List<double> queryEmbedding,
    List<String> texts,
    List<List<double>> embeddings, {
    int topK = 5,
    double threshold = 0.0,
  }) {
    if (texts.length != embeddings.length) {
      throw ArgumentError('Texts and embeddings lists must have same length');
    }

    final results = <SimilarityResult>[];

    for (int i = 0; i < embeddings.length; i++) {
      final similarity = cosineSimilarity(queryEmbedding, embeddings[i]);

      if (similarity >= threshold) {
        results.add(
          SimilarityResult(
            text: texts[i],
            similarity: similarity,
            index: i,
            embedding: EmbeddingResult(
              embedding: embeddings[i],
              tokenCount: 0, // Would need to track this separately
              computeTimeMs: 0.0,
              normalized: true,
            ),
          ),
        );
      }
    }

    // Sort by similarity (descending) and take top-K
    results.sort((a, b) => b.similarity.compareTo(a.similarity));
    return results.take(topK).toList();
  }

  /// Create a simple vector database in memory
  static VectorDatabase createVectorDatabase() {
    return VectorDatabase();
  }
}

/// Simple in-memory vector database for RAG
class VectorDatabase {
  final List<String> _texts = [];
  final List<List<double>> _embeddings = [];
  final Map<String, dynamic> _metadata = {};

  /// Add a text with its embedding
  void add(String text, List<double> embedding, {Map<String, dynamic>? metadata}) {
    _texts.add(text);
    _embeddings.add(embedding);
    if (metadata != null) {
      _metadata[text] = metadata;
    }
  }

  /// Search for similar texts
  List<SimilarityResult> search(List<double> queryEmbedding, {int topK = 5, double threshold = 0.0}) {
    return EmbeddingUtils.findTopSimilar(queryEmbedding, _texts, _embeddings, topK: topK, threshold: threshold);
  }

  /// Get metadata for a text
  Map<String, dynamic>? getMetadata(String text) {
    return _metadata[text];
  }

  /// Number of stored embeddings
  int get size => _texts.length;

  /// Clear all data
  void clear() {
    _texts.clear();
    _embeddings.clear();
    _metadata.clear();
  }

  /// Export to JSON
  Map<String, dynamic> toJson() {
    return {'texts': _texts, 'embeddings': _embeddings, 'metadata': _metadata};
  }

  /// Import from JSON
  void fromJson(Map<String, dynamic> json) {
    clear();
    _texts.addAll((json['texts'] as List).cast<String>());
    _embeddings.addAll((json['embeddings'] as List).map((e) => (e as List).cast<double>()).toList());
    _metadata.addAll(json['metadata'] as Map<String, dynamic>);
  }
}
