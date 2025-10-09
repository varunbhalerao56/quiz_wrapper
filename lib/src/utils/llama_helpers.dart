/// Quality-of-life helpers and utilities for LlamaService
///
/// Provides capacity management, performance monitoring, text processing,
/// and various utility functions to make working with llama.cpp easier.

// ignore_for_file: dangling_library_doc_comments
import 'package:flutter/foundation.dart';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;

import 'package:quiz_wrapper/src/ffi/llama_ffi.dart';

/// Performance metrics for llama.cpp operations
class PerformanceMetrics {
  final double tokensPerSecond;
  final double promptProcessingSpeed;
  final Duration totalTime;
  final Duration promptTime;
  final Duration generationTime;
  final int totalTokens;
  final int promptTokens;
  final int generatedTokens;
  final double memoryUsageMB;

  const PerformanceMetrics({
    required this.tokensPerSecond,
    required this.promptProcessingSpeed,
    required this.totalTime,
    required this.promptTime,
    required this.generationTime,
    required this.totalTokens,
    required this.promptTokens,
    required this.generatedTokens,
    required this.memoryUsageMB,
  });

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'tokensPerSecond': tokensPerSecond,
      'promptProcessingSpeed': promptProcessingSpeed,
      'totalTimeMs': totalTime.inMilliseconds,
      'promptTimeMs': promptTime.inMilliseconds,
      'generationTimeMs': generationTime.inMilliseconds,
      'totalTokens': totalTokens,
      'promptTokens': promptTokens,
      'generatedTokens': generatedTokens,
      'memoryUsageMB': memoryUsageMB,
    };
  }

  /// Create from JSON
  factory PerformanceMetrics.fromJson(Map<String, dynamic> json) {
    return PerformanceMetrics(
      tokensPerSecond: (json['tokensPerSecond'] as num).toDouble(),
      promptProcessingSpeed: (json['promptProcessingSpeed'] as num).toDouble(),
      totalTime: Duration(milliseconds: json['totalTimeMs'] as int),
      promptTime: Duration(milliseconds: json['promptTimeMs'] as int),
      generationTime: Duration(milliseconds: json['generationTimeMs'] as int),
      totalTokens: json['totalTokens'] as int,
      promptTokens: json['promptTokens'] as int,
      generatedTokens: json['generatedTokens'] as int,
      memoryUsageMB: (json['memoryUsageMB'] as num).toDouble(),
    );
  }

  @override
  String toString() {
    return 'Performance: ${tokensPerSecond.toStringAsFixed(1)} tok/s, '
        '$totalTokens tokens in ${totalTime.inMilliseconds}ms';
  }
}

/// Performance monitor for tracking llama.cpp operations
class PerformanceMonitor {
  final llama_cpp _llamaCpp;
  final List<PerformanceMetrics> _history = [];
  final int _maxHistorySize;

  PerformanceMonitor(this._llamaCpp, {int maxHistorySize = 100}) : _maxHistorySize = maxHistorySize;

  /// Get performance data from llama.cpp context
  PerformanceMetrics getContextPerformance(Pointer<llama_context> context) {
    final perfData = _llamaCpp.llama_perf_context(context);

    final totalTime = Duration(milliseconds: (perfData.t_start_ms + perfData.t_p_eval_ms + perfData.t_eval_ms).round());
    final promptTime = Duration(milliseconds: perfData.t_p_eval_ms.round());
    final generationTime = Duration(milliseconds: perfData.t_eval_ms.round());

    final tokensPerSecond = generationTime.inMilliseconds > 0
        ? (perfData.n_eval * 1000.0) / generationTime.inMilliseconds
        : 0.0;

    final promptProcessingSpeed = promptTime.inMilliseconds > 0
        ? (perfData.n_p_eval * 1000.0) / promptTime.inMilliseconds
        : 0.0;

    final metrics = PerformanceMetrics(
      tokensPerSecond: tokensPerSecond,
      promptProcessingSpeed: promptProcessingSpeed,
      totalTime: totalTime,
      promptTime: promptTime,
      generationTime: generationTime,
      totalTokens: perfData.n_p_eval + perfData.n_eval,
      promptTokens: perfData.n_p_eval,
      generatedTokens: perfData.n_eval,
      memoryUsageMB: _getMemoryUsage(),
    );

    _addToHistory(metrics);
    return metrics;
  }

  /// Get performance data from sampler
  Map<String, dynamic> getSamplerPerformance(Pointer<llama_sampler> sampler) {
    final perfData = _llamaCpp.llama_perf_sampler(sampler);

    return {
      'sampleTimeMs': perfData.t_sample_ms,
      'sampleCount': perfData.n_sample,
      'samplesPerSecond': perfData.t_sample_ms > 0 ? (perfData.n_sample * 1000.0) / perfData.t_sample_ms : 0.0,
    };
  }

  /// Add metrics to history
  void _addToHistory(PerformanceMetrics metrics) {
    _history.add(metrics);

    // Keep only recent history
    while (_history.length > _maxHistorySize) {
      _history.removeAt(0);
    }
  }

  /// Get average performance over recent history
  PerformanceMetrics? getAveragePerformance({int? lastN}) {
    if (_history.isEmpty) return null;

    final samples = lastN != null ? _history.take(math.min(lastN, _history.length)).toList() : _history;

    if (samples.isEmpty) return null;

    final avgTokensPerSecond = samples.map((m) => m.tokensPerSecond).reduce((a, b) => a + b) / samples.length;

    final avgPromptSpeed = samples.map((m) => m.promptProcessingSpeed).reduce((a, b) => a + b) / samples.length;

    final totalTime = samples.map((m) => m.totalTime.inMilliseconds).reduce((a, b) => a + b);

    final totalTokens = samples.map((m) => m.totalTokens).reduce((a, b) => a + b);

    return PerformanceMetrics(
      tokensPerSecond: avgTokensPerSecond,
      promptProcessingSpeed: avgPromptSpeed,
      totalTime: Duration(milliseconds: (totalTime / samples.length).round()),
      promptTime: Duration(
        milliseconds: samples.map((m) => m.promptTime.inMilliseconds).reduce((a, b) => a + b) ~/ samples.length,
      ),
      generationTime: Duration(
        milliseconds: samples.map((m) => m.generationTime.inMilliseconds).reduce((a, b) => a + b) ~/ samples.length,
      ),
      totalTokens: (totalTokens / samples.length).round(),
      promptTokens: samples.map((m) => m.promptTokens).reduce((a, b) => a + b) ~/ samples.length,
      generatedTokens: samples.map((m) => m.generatedTokens).reduce((a, b) => a + b) ~/ samples.length,
      memoryUsageMB: samples.map((m) => m.memoryUsageMB).reduce((a, b) => a + b) / samples.length,
    );
  }

  /// Get current memory usage (rough estimate)
  double _getMemoryUsage() {
    // This is a rough estimate - would need platform-specific implementation
    return ProcessInfo.currentRss / (1024 * 1024); // Convert to MB
  }

  /// Reset performance counters
  void reset(Pointer<llama_context> context) {
    _llamaCpp.llama_perf_context_reset(context);
    _history.clear();
  }

  /// Print performance report
  void printReport(Pointer<llama_context> context) {
    _llamaCpp.llama_perf_context_print(context);
  }
}

/// Capacity management utilities
class CapacityManager {
  final llama_cpp _llamaCpp;
  final Pointer<llama_context> _context;
  final Pointer<llama_model> _model;

  CapacityManager(this._llamaCpp, this._context, this._model);

  /// Get remaining context space
  int getRemainingContextSpace(int currentPosition) {
    final maxCtx = _llamaCpp.llama_n_ctx(_context);
    return math.max(0, maxCtx - currentPosition);
  }

  /// Check if context limit was reached
  bool wasContextLimitReached(int currentPosition) {
    return currentPosition >= _llamaCpp.llama_n_ctx(_context);
  }

  /// Get context utilization percentage
  double getContextUtilization(int currentPosition) {
    final maxCtx = _llamaCpp.llama_n_ctx(_context);
    return maxCtx > 0 ? currentPosition / maxCtx : 0.0;
  }

  /// Estimate tokens needed for text
  int estimateTokensForText(String text) {
    // Rough estimate: 4 characters per token on average
    return (text.length / 4).ceil();
  }

  /// Check if text will fit in remaining context
  bool willTextFit(String text, int currentPosition) {
    final estimatedTokens = estimateTokensForText(text);
    final remaining = getRemainingContextSpace(currentPosition);
    return estimatedTokens <= remaining;
  }

  /// Get model information
  Map<String, dynamic> getModelInfo() {
    return {
      'nCtx': _llamaCpp.llama_n_ctx(_context),
      'nEmbd': _llamaCpp.llama_model_n_embd(_model),
      'nLayer': _llamaCpp.llama_model_n_layer(_model),
      'nHead': _llamaCpp.llama_model_n_head(_model),
      'nParams': _llamaCpp.llama_model_n_params(_model),
      'hasEncoder': _llamaCpp.llama_model_has_encoder(_model),
      'hasDecoder': _llamaCpp.llama_model_has_decoder(_model),
      'isRecurrent': _llamaCpp.llama_model_is_recurrent(_model),
    };
  }

  /// Get context information
  Map<String, dynamic> getContextInfo() {
    return {
      'nCtx': _llamaCpp.llama_n_ctx(_context),
      'nBatch': _llamaCpp.llama_n_batch(_context),
      'nUbatch': _llamaCpp.llama_n_ubatch(_context),
      'nSeqMax': _llamaCpp.llama_n_seq_max(_context),
      'nThreads': _llamaCpp.llama_n_threads(_context),
      'nThreadsBatch': _llamaCpp.llama_n_threads_batch(_context),
    };
  }

  /// Suggest optimal batch size based on context and model
  int suggestOptimalBatchSize() {
    final nCtx = _llamaCpp.llama_n_ctx(_context);
    final currentBatch = _llamaCpp.llama_n_batch(_context);

    // Suggest batch size as a fraction of context size
    final suggested = math.min(currentBatch, (nCtx * 0.25).round());
    return math.max(1, suggested);
  }
}

/// Text processing utilities
class TextProcessor {
  /// Clean and normalize text for processing
  static String cleanText(String text) {
    String cleaned = text;

    // Normalize whitespace
    cleaned = cleaned.replaceAll(RegExp(r'\s+'), ' ');

    // Remove control characters except newlines and tabs
    cleaned = cleaned.replaceAll(RegExp(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'), '');

    // Trim
    cleaned = cleaned.trim();

    return cleaned;
  }

  /// Split text into sentences
  static List<String> splitIntoSentences(String text) {
    // Simple sentence splitting (could be improved with NLP)
    final sentences = text.split(RegExp(r'[.!?]+\s*')).where((s) => s.trim().isNotEmpty).map((s) => s.trim()).toList();

    return sentences;
  }

  /// Split text into paragraphs
  static List<String> splitIntoParagraphs(String text) {
    return text.split(RegExp(r'\n\s*\n')).where((p) => p.trim().isNotEmpty).map((p) => p.trim()).toList();
  }

  /// Extract keywords from text (simple implementation)
  static List<String> extractKeywords(String text, {int maxKeywords = 10}) {
    // Simple keyword extraction based on word frequency
    final words = text
        .toLowerCase()
        .replaceAll(RegExp(r'[^\w\s]'), '')
        .split(RegExp(r'\s+'))
        .where((w) => w.length > 3) // Filter short words
        .toList();

    // Count word frequency
    final wordCounts = <String, int>{};
    for (final word in words) {
      wordCounts[word] = (wordCounts[word] ?? 0) + 1;
    }

    // Sort by frequency and return top keywords
    final sorted = wordCounts.entries.toList()..sort((a, b) => b.value.compareTo(a.value));

    return sorted.take(maxKeywords).map((e) => e.key).toList();
  }

  /// Calculate text similarity (simple Jaccard similarity)
  static double calculateSimilarity(String text1, String text2) {
    final words1 = text1.toLowerCase().split(RegExp(r'\s+')).toSet();
    final words2 = text2.toLowerCase().split(RegExp(r'\s+')).toSet();

    final intersection = words1.intersection(words2).length;
    final union = words1.union(words2).length;

    return union > 0 ? intersection / union : 0.0;
  }
}

/// System information and capability detection
class SystemInfo {
  /// Check if Metal (GPU) acceleration is available on macOS
  static bool isMetalAvailable() {
    if (!Platform.isMacOS) return false;

    // Simple check - in practice you'd want more sophisticated detection
    try {
      final result = Process.runSync('system_profiler', ['SPDisplaysDataType']);
      return result.stdout.toString().contains('Metal');
    } catch (e) {
      return false;
    }
  }

  /// Get optimal thread count for current system
  static int getOptimalThreadCount() {
    return math.max(1, Platform.numberOfProcessors - 1);
  }

  /// Get available memory in MB
  static int getAvailableMemoryMB() {
    // Platform-specific implementation would be needed
    // This is a placeholder
    return 8192; // 8GB default
  }

  /// Get system capabilities
  static Map<String, dynamic> getCapabilities() {
    return {
      'platform': Platform.operatingSystem,
      'numberOfProcessors': Platform.numberOfProcessors,
      'optimalThreads': getOptimalThreadCount(),
      'metalAvailable': isMetalAvailable(),
      'availableMemoryMB': getAvailableMemoryMB(),
      'is64Bit': Platform.version.contains('64'),
    };
  }

  /// Suggest optimal configuration for current system
  static Map<String, dynamic> suggestOptimalConfig() {
    final capabilities = getCapabilities();

    return {
      'nThreads': capabilities['optimalThreads'],
      'nGpuLayers': capabilities['metalAvailable'] ? 32 : 0,
      'nCtx': capabilities['availableMemoryMB'] > 16000 ? 4096 : 2048,
      'nBatch': capabilities['availableMemoryMB'] > 8000 ? 512 : 256,
      'useMmap': true,
      'useMlock': capabilities['availableMemoryMB'] > 16000,
    };
  }
}

/// Device-specific optimization utilities
class DeviceOptimizer {
  /// Optimize configuration for mobile devices
  static Map<String, dynamic> optimizeForMobile() {
    return {
      'nCtx': 1024, // Smaller context for mobile
      'nBatch': 128,
      'nThreads': math.min(4, Platform.numberOfProcessors),
      'useMmap': true,
      'useMlock': false, // Avoid memory locking on mobile
      'nGpuLayers': 0, // CPU only for mobile
    };
  }

  /// Optimize configuration for desktop
  static Map<String, dynamic> optimizeForDesktop() {
    final capabilities = SystemInfo.getCapabilities();

    return {
      'nCtx': 4096,
      'nBatch': 512,
      'nThreads': capabilities['optimalThreads'],
      'useMmap': true,
      'useMlock': capabilities['availableMemoryMB'] > 16000,
      'nGpuLayers': capabilities['metalAvailable'] ? 32 : 0,
    };
  }

  /// Optimize configuration for server
  static Map<String, dynamic> optimizeForServer() {
    return {
      'nCtx': 8192,
      'nBatch': 1024,
      'nThreads': Platform.numberOfProcessors,
      'useMmap': true,
      'useMlock': true,
      'nGpuLayers': 99, // Offload everything if GPU available
    };
  }

  /// Auto-detect and optimize for current platform
  static Map<String, dynamic> autoOptimize() {
    final memory = SystemInfo.getAvailableMemoryMB();
    final cores = Platform.numberOfProcessors;

    if (Platform.isAndroid || Platform.isIOS) {
      return optimizeForMobile();
    } else if (memory > 32000 && cores > 8) {
      return optimizeForServer();
    } else {
      return optimizeForDesktop();
    }
  }
}

/// Logging utilities with different verbosity levels
class LlamaLogger {
  static bool _verboseLogging = false;
  static final List<String> _logHistory = [];
  static const int _maxLogHistory = 1000;

  /// Enable/disable verbose logging
  static void setVerbose(bool verbose) {
    _verboseLogging = verbose;
  }

  /// Log a message with timestamp
  static void log(String message, {String? category}) {
    final timestamp = DateTime.now().toIso8601String();
    final logEntry = '[$timestamp]${category != null ? ' [$category]' : ''} $message';

    if (_verboseLogging) {
      debugPrint(logEntry);
    }

    _logHistory.add(logEntry);

    // Keep log history manageable
    while (_logHistory.length > _maxLogHistory) {
      _logHistory.removeAt(0);
    }
  }

  /// Log debug information
  static void debug(String message) {
    if (_verboseLogging) {
      log(message, category: 'DEBUG');
    }
  }

  /// Log info message
  static void info(String message) {
    log(message, category: 'INFO');
  }

  /// Log warning
  static void warn(String message) {
    log(message, category: 'WARN');
  }

  /// Log error
  static void error(String message, [Object? error]) {
    final errorMsg = error != null ? '$message: $error' : message;
    log(errorMsg, category: 'ERROR');
  }

  /// Get recent log entries
  static List<String> getRecentLogs({int? count}) {
    count ??= 50;
    return _logHistory.length > count ? _logHistory.sublist(_logHistory.length - count) : List.from(_logHistory);
  }

  /// Clear log history
  static void clearLogs() {
    _logHistory.clear();
  }

  /// Export logs to string
  static String exportLogs() {
    return _logHistory.join('\n');
  }
}

/// Utility for managing temporary files and cleanup
class TempFileManager {
  static final Set<String> _tempFiles = {};

  /// Create a temporary file and track it for cleanup
  static File createTempFile(String prefix, {String? extension}) {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final filename = '$prefix$timestamp${extension ?? '.tmp'}';
    final file = File('temp/$filename');

    _tempFiles.add(file.path);
    return file;
  }

  /// Clean up all tracked temporary files
  static Future<int> cleanupTempFiles() async {
    int deletedCount = 0;

    for (final filePath in _tempFiles.toList()) {
      try {
        final file = File(filePath);
        if (await file.exists()) {
          await file.delete();
          deletedCount++;
        }
        _tempFiles.remove(filePath);
      } catch (e) {
        debugPrint('Failed to delete temp file $filePath: $e');
      }
    }

    return deletedCount;
  }

  /// Clean up temp files older than specified duration
  static Future<int> cleanupOldTempFiles({Duration? olderThan}) async {
    olderThan ??= const Duration(hours: 24);
    final cutoff = DateTime.now().subtract(olderThan);
    int deletedCount = 0;

    final tempDir = Directory('temp');
    if (!await tempDir.exists()) return 0;

    await for (final entity in tempDir.list()) {
      if (entity is File) {
        try {
          final stat = await entity.stat();
          if (stat.modified.isBefore(cutoff)) {
            await entity.delete();
            _tempFiles.remove(entity.path);
            deletedCount++;
          }
        } catch (e) {
          debugPrint('Failed to clean up ${entity.path}: $e');
        }
      }
    }

    return deletedCount;
  }
}

/// Benchmark utilities for testing performance
class LlamaBenchmark {
  /// Run a simple generation benchmark
  static Future<PerformanceMetrics> benchmarkGeneration(
    llama_cpp llamaCpp,
    Pointer<llama_context> context,
    Pointer<llama_sampler> sampler,
    Pointer<llama_vocab> vocab, {
    String prompt = "The quick brown fox",
    int tokens = 50,
  }) async {
    final stopwatch = Stopwatch()..start();

    // This would implement a full benchmark
    // For now, return placeholder metrics
    stopwatch.stop();

    return PerformanceMetrics(
      tokensPerSecond: tokens / (stopwatch.elapsedMilliseconds / 1000.0),
      promptProcessingSpeed: 100.0, // Placeholder
      totalTime: stopwatch.elapsed,
      promptTime: Duration(milliseconds: 100),
      generationTime: Duration(milliseconds: stopwatch.elapsedMilliseconds - 100),
      totalTokens: tokens + 10, // Prompt + generated
      promptTokens: 10,
      generatedTokens: tokens,
      memoryUsageMB: 512.0, // Placeholder
    );
  }

  /// Run comprehensive benchmark suite
  static Future<Map<String, PerformanceMetrics>> runBenchmarkSuite(
    llama_cpp llamaCpp,
    Pointer<llama_context> context,
    Pointer<llama_sampler> sampler,
    Pointer<llama_vocab> vocab,
  ) async {
    final results = <String, PerformanceMetrics>{};

    // Short generation
    results['short'] = await benchmarkGeneration(llamaCpp, context, sampler, vocab, prompt: "Hello", tokens: 10);

    // Medium generation
    results['medium'] = await benchmarkGeneration(
      llamaCpp,
      context,
      sampler,
      vocab,
      prompt: "Write a short story about",
      tokens: 50,
    );

    // Long generation
    results['long'] = await benchmarkGeneration(
      llamaCpp,
      context,
      sampler,
      vocab,
      prompt: "Explain the concept of artificial intelligence",
      tokens: 200,
    );

    return results;
  }
}
