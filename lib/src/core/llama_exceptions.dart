/// Exception handling and safety for LlamaService
///
/// Provides robust error handling, state validation, and safety guardrails
/// for all llama.cpp operations.

import 'dart:ffi' as ffi;

/// Base exception class for all llama.cpp related errors
class LlamaException implements Exception {
  final String message;
  final String? operation;
  final Object? originalError;
  final StackTrace? stackTrace;

  const LlamaException(this.message, {this.operation, this.originalError, this.stackTrace});

  @override
  String toString() {
    final buffer = StringBuffer('LlamaException: $message');
    if (operation != null) {
      buffer.write(' (during: $operation)');
    }
    if (originalError != null) {
      buffer.write('\nCaused by: $originalError');
    }
    return buffer.toString();
  }
}

/// Exception thrown when llama.cpp is not properly initialized
class LlamaNotInitializedException extends LlamaException {
  const LlamaNotInitializedException([String? operation])
    : super('LlamaService not initialized. Call init() first.', operation: operation);
}

/// Exception thrown when model is not loaded
class LlamaModelNotLoadedException extends LlamaException {
  const LlamaModelNotLoadedException([String? operation])
    : super('Model not loaded. Call loadModel() first.', operation: operation);
}

/// Exception thrown when context is not created
class LlamaContextNotCreatedException extends LlamaException {
  const LlamaContextNotCreatedException([String? operation])
    : super('Context not created. Call createContext() first.', operation: operation);
}

/// Exception thrown when context limit is exceeded
class LlamaContextOverflowException extends LlamaException {
  final int currentTokens;
  final int maxTokens;

  const LlamaContextOverflowException(this.currentTokens, this.maxTokens)
    : super('Context overflow: $currentTokens tokens exceeds limit of $maxTokens', operation: 'generation');
}

/// Exception thrown when batch parameters are invalid
class LlamaBatchException extends LlamaException {
  const LlamaBatchException(String message, [String? operation]) : super(message, operation: operation);
}

/// Exception thrown when tokenization fails
class LlamaTokenizationException extends LlamaException {
  final String text;

  const LlamaTokenizationException(this.text, String message)
    : super('Tokenization failed for text: "$text" - $message', operation: 'tokenization');
}

/// Exception thrown when sampling fails
class LlamaSamplingException extends LlamaException {
  const LlamaSamplingException(String message) : super(message, operation: 'sampling');
}

/// Exception thrown when model loading fails
class LlamaModelLoadException extends LlamaException {
  final String modelPath;

  const LlamaModelLoadException(this.modelPath, String message)
    : super('Failed to load model from "$modelPath": $message', operation: 'model_loading');
}

/// Exception thrown when session operations fail
class LlamaSessionException extends LlamaException {
  const LlamaSessionException(String message, [String? operation]) : super(message, operation: operation ?? 'session');
}

/// Exception thrown when embeddings operations fail
class LlamaEmbeddingsException extends LlamaException {
  const LlamaEmbeddingsException(String message) : super(message, operation: 'embeddings');
}

/// Safety utilities for validating llama.cpp state and parameters
class LlamaSafety {
  /// Validates that a pointer is not null and has a valid address
  static void validatePointer<T>(T? pointer, String name, [String? operation]) {
    if (pointer == null) {
      throw LlamaException('$name is null', operation: operation);
    }

    // For FFI pointers, check address
    if (pointer is ffi.Pointer && pointer.address == 0) {
      throw LlamaException('$name has null address', operation: operation);
    }
  }

  /// Validates batch parameters
  static void validateBatch({required int nTokens, required int maxBatch, String? operation}) {
    if (nTokens <= 0) {
      throw LlamaBatchException('Invalid batch size: $nTokens (must be > 0)', operation);
    }

    if (nTokens > maxBatch) {
      throw LlamaBatchException('Batch size $nTokens exceeds maximum $maxBatch', operation);
    }
  }

  /// Validates context capacity
  static void validateContextCapacity({
    required int currentPos,
    required int additionalTokens,
    required int maxCtx,
    String? operation,
  }) {
    final totalTokens = currentPos + additionalTokens;
    if (totalTokens > maxCtx) {
      throw LlamaContextOverflowException(totalTokens, maxCtx);
    }
  }

  /// Validates sampling parameters
  static void validateSamplingParams({
    required double temperature,
    required double topP,
    required int topK,
    required double minP,
    String? operation,
  }) {
    if (temperature < 0.0) {
      throw LlamaSamplingException('Invalid temperature: $temperature (must be >= 0.0)');
    }

    if (topP < 0.0 || topP > 1.0) {
      throw LlamaSamplingException('Invalid top_p: $topP (must be 0.0-1.0)');
    }

    if (topK < 0) {
      throw LlamaSamplingException('Invalid top_k: $topK (must be >= 0)');
    }

    if (minP < 0.0 || minP > 1.0) {
      throw LlamaSamplingException('Invalid min_p: $minP (must be 0.0-1.0)');
    }
  }

  /// Wraps an operation with exception handling
  static T wrapOperation<T>(String operation, T Function() fn, {T? fallback}) {
    try {
      return fn();
    } catch (e, stackTrace) {
      if (e is LlamaException) {
        rethrow;
      }

      if (fallback != null) {
        print('Warning: $operation failed, using fallback: $e');
        return fallback;
      }

      throw LlamaException(
        'Operation failed: ${e.toString()}',
        operation: operation,
        originalError: e,
        stackTrace: stackTrace,
      );
    }
  }

  /// Wraps an async operation with exception handling
  static Future<T> wrapAsyncOperation<T>(String operation, Future<T> Function() fn, {T? fallback}) async {
    try {
      return await fn();
    } catch (e, stackTrace) {
      if (e is LlamaException) {
        rethrow;
      }

      if (fallback != null) {
        print('Warning: $operation failed, using fallback: $e');
        return fallback;
      }

      throw LlamaException(
        'Async operation failed: ${e.toString()}',
        operation: operation,
        originalError: e,
        stackTrace: stackTrace,
      );
    }
  }
}

/// Mixin for classes that need to track disposal state
mixin DisposableMixin {
  bool _disposed = false;

  /// Whether this object has been disposed
  bool get isDisposed => _disposed;

  /// Throws if this object has been disposed
  void checkNotDisposed([String? operation]) {
    if (_disposed) {
      throw LlamaException('Object has been disposed and cannot be used', operation: operation);
    }
  }

  /// Marks this object as disposed
  void markDisposed() {
    _disposed = true;
  }
}
