import 'package:flutter/foundation.dart';

/// State machine and status tracking for LlamaService
///
/// Provides comprehensive status tracking, state transitions, and
/// event notifications for all llama.cpp operations.

// ignore_for_file: dangling_library_doc_comments

/// Represents the current state of the LlamaService
enum LlamaStatus {
  /// Service not yet initialized
  uninitialized('Uninitialized', 'Service not yet initialized'),

  /// Backend initialized but no model loaded
  initialized('Initialized', 'Backend ready, no model loaded'),

  /// Model is currently being loaded
  loadingModel('Loading Model', 'Loading model from file'),

  /// Model loaded but no context created
  modelLoaded('Model Loaded', 'Model loaded, ready to create context'),

  /// Context is being created
  creatingContext('Creating Context', 'Creating inference context'),

  /// Ready for inference operations
  ready('Ready', 'Ready for text generation and embeddings'),

  /// Currently generating text
  generating('Generating', 'Text generation in progress'),

  /// Currently computing embeddings
  computing('Computing', 'Computing text embeddings'),

  /// Currently saving/loading session
  sessionOperation('Session Op', 'Session save/load in progress'),

  /// An error occurred
  error('Error', 'An error occurred during operation'),

  /// Service has been disposed
  disposed('Disposed', 'Service has been disposed and cannot be used');

  const LlamaStatus(this.label, this.description);

  final String label;
  final String description;

  /// Whether the service is in a working state
  bool get isOperational => [initialized, modelLoaded, ready].contains(this);

  /// Whether the service can perform inference
  bool get canInfer => [ready].contains(this);

  /// Whether the service is currently busy
  bool get isBusy => [loadingModel, creatingContext, generating, computing, sessionOperation].contains(this);

  /// Whether the service is in an error state
  bool get hasError => this == error;

  /// Whether the service has been disposed
  bool get isDisposed => this == disposed;
}

/// Represents different types of operations
enum LlamaOperation {
  initialization('Backend Initialization'),
  modelLoading('Model Loading'),
  contextCreation('Context Creation'),
  textGeneration('Text Generation'),
  embeddings('Embeddings Computation'),
  tokenization('Tokenization'),
  sessionSave('Session Save'),
  sessionLoad('Session Load'),
  disposal('Resource Disposal');

  const LlamaOperation(this.description);
  final String description;
}

/// Progress information for long-running operations
class LlamaProgress {
  final LlamaOperation operation;
  final double progress; // 0.0 to 1.0
  final String? message;
  final int? currentStep;
  final int? totalSteps;

  const LlamaProgress({
    required this.operation,
    required this.progress,
    this.message,
    this.currentStep,
    this.totalSteps,
  });

  /// Create progress from step counts
  factory LlamaProgress.fromSteps({
    required LlamaOperation operation,
    required int currentStep,
    required int totalSteps,
    String? message,
  }) {
    return LlamaProgress(
      operation: operation,
      progress: totalSteps > 0 ? currentStep / totalSteps : 0.0,
      message: message,
      currentStep: currentStep,
      totalSteps: totalSteps,
    );
  }

  /// Create indeterminate progress
  factory LlamaProgress.indeterminate({required LlamaOperation operation, String? message}) {
    return LlamaProgress(
      operation: operation,
      progress: -1.0, // Indicates indeterminate
      message: message,
    );
  }

  bool get isIndeterminate => progress < 0.0;
  bool get isComplete => progress >= 1.0;
}

/// Event fired when status changes
class LlamaStatusEvent {
  final LlamaStatus oldStatus;
  final LlamaStatus newStatus;
  final DateTime timestamp;
  final String? message;
  final Object? error;

  const LlamaStatusEvent({
    required this.oldStatus,
    required this.newStatus,
    required this.timestamp,
    this.message,
    this.error,
  });

  bool get isError => error != null;
}

/// Callback types for status and progress events
typedef LlamaStatusCallback = void Function(LlamaStatusEvent event);
typedef LlamaProgressCallback = void Function(LlamaProgress progress);

/// Status manager for tracking LlamaService state
class LlamaStatusManager {
  LlamaStatus _status = LlamaStatus.uninitialized;
  final List<LlamaStatusCallback> _statusCallbacks = [];
  final List<LlamaProgressCallback> _progressCallbacks = [];
  final List<LlamaStatusEvent> _history = [];

  /// Current status
  LlamaStatus get status => _status;

  /// Status history (last 100 events)
  List<LlamaStatusEvent> get history => List.unmodifiable(_history);

  /// Add a status change listener
  void addStatusListener(LlamaStatusCallback callback) {
    _statusCallbacks.add(callback);
  }

  /// Remove a status change listener
  void removeStatusListener(LlamaStatusCallback callback) {
    _statusCallbacks.remove(callback);
  }

  /// Add a progress listener
  void addProgressListener(LlamaProgressCallback callback) {
    _progressCallbacks.add(callback);
  }

  /// Remove a progress listener
  void removeProgressListener(LlamaProgressCallback callback) {
    _progressCallbacks.remove(callback);
  }

  /// Update the current status
  void updateStatus(LlamaStatus newStatus, {String? message, Object? error}) {
    if (_status == newStatus) return;

    final event = LlamaStatusEvent(
      oldStatus: _status,
      newStatus: newStatus,
      timestamp: DateTime.now(),
      message: message,
      error: error,
    );

    _status = newStatus;
    _history.add(event);

    // Keep only last 100 events
    if (_history.length > 100) {
      _history.removeAt(0);
    }

    // Notify listeners
    for (final callback in _statusCallbacks) {
      try {
        callback(event);
      } catch (e) {
        debugPrint('Error in status callback: $e');
      }
    }
  }

  /// Report progress for the current operation
  void reportProgress(LlamaProgress progress) {
    for (final callback in _progressCallbacks) {
      try {
        callback(progress);
      } catch (e) {
        debugPrint('Error in progress callback: $e');
      }
    }
  }

  /// Clear all listeners and history
  void dispose() {
    _statusCallbacks.clear();
    _progressCallbacks.clear();
    _history.clear();
    updateStatus(LlamaStatus.disposed);
  }

  /// Get a human-readable status summary
  String getStatusSummary() {
    final buffer = StringBuffer();
    buffer.writeln('Current Status: ${_status.label}');
    buffer.writeln('Description: ${_status.description}');
    buffer.writeln('Operational: ${_status.isOperational}');
    buffer.writeln('Can Infer: ${_status.canInfer}');
    buffer.writeln('Busy: ${_status.isBusy}');

    if (_history.isNotEmpty) {
      buffer.writeln('\nRecent Events:');
      for (final event in _history.take(5).toList().reversed) {
        buffer.writeln('  ${event.timestamp}: ${event.oldStatus.label} → ${event.newStatus.label}');
        if (event.message != null) {
          buffer.writeln('    ${event.message}');
        }
        if (event.error != null) {
          buffer.writeln('    Error: ${event.error}');
        }
      }
    }

    return buffer.toString();
  }
}

/// Mixin for classes that need status management
mixin StatusMixin {
  late final LlamaStatusManager _statusManager = LlamaStatusManager();

  /// Status manager for this instance
  LlamaStatusManager get statusManager => _statusManager;

  /// Current status
  LlamaStatus get status => _statusManager.status;

  /// Update status with optional message and error
  void updateStatus(LlamaStatus newStatus, {String? message, Object? error}) {
    _statusManager.updateStatus(newStatus, message: message, error: error);
  }

  /// Report progress for current operation
  void reportProgress(LlamaProgress progress) {
    _statusManager.reportProgress(progress);
  }

  /// Dispose status manager
  void disposeStatus() {
    _statusManager.dispose();
  }
}
