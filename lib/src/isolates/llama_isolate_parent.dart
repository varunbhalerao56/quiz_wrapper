/// LlamaIsolateParent - UI-facing service that manages the background isolate
///
/// This runs on the main UI thread and communicates with LlamaIsolateChild.
/// All public methods are async and won't block the UI.

// ignore_for_file: dangling_library_doc_comments

import 'dart:async';
import 'dart:isolate';
import 'package:flutter/foundation.dart';
import 'package:quiz_wrapper/src/isolates/llama_isolate_child.dart';
import 'package:quiz_wrapper/src/isolates/llama_isolate_types.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';

/// Parent service that manages communication with the LlamaChild isolate
class LlamaIsolateParent {
  Isolate? _isolate;
  SendPort? _sendPort;
  final _receivePort = ReceivePort();

  // Status tracking
  IsolateStatus _status = IsolateStatus.uninitialized;
  IsolateStatus get status => _status;

  // Request tracking
  int _requestCounter = 0;
  final Map<String, Completer<String?>> _pendingRequests = {};
  final Map<String, StreamController<String>> _streamControllers = {};

  // Broadcast status changes
  final _statusController = StreamController<IsolateStatus>.broadcast();
  Stream<IsolateStatus> get statusStream => _statusController.stream;

  bool get isInitialized => _isolate != null && _sendPort != null;
  bool get isGenerating => _status == IsolateStatus.generating;

  /// Start the background isolate
  Future<void> start() async {
    if (_isolate != null) {
      debugPrint('[LlamaParent] Already started');
      return;
    }

    debugPrint('[LlamaParent] Starting isolate...');

    _isolate = await Isolate.spawn(llamaIsolateEntry, _receivePort.sendPort, debugName: 'LlamaIsolate');

    // Wait for the isolate to send its SendPort
    final completer = Completer<SendPort>();

    // Set up single listener that handles everything
    _receivePort.listen((message) {
      if (message is SendPort && !completer.isCompleted) {
        _sendPort = message;
        completer.complete(message);
        debugPrint('[LlamaParent] ✓ Isolate SendPort received');
      } else {
        _handleResponse(message);
      }
    });

    await completer.future;
    debugPrint('[LlamaParent] ✓ Isolate ready');
  }

  /// Handle responses from the isolate
  void _handleResponse(dynamic message) {
    if (message is! LlamaResponse) {
      debugPrint('[LlamaParent] ⚠ Received non-response message: ${message.runtimeType}');
      return;
    }

    debugPrint('[LlamaParent] 📩 Received: ${message.runtimeType}');

    switch (message) {
      case StatusResponse(:final status, :final message):
        debugPrint('[LlamaParent] Status changed: $_status -> $status');
        _status = status;
        _statusController.add(status);
        if (message != null) {
          debugPrint('[LlamaParent] Status message: $message');
        }

      case SuccessResponse(:final message):
        if (message != null) {
          debugPrint('[LlamaParent] ✓ Success: $message');
        }

      case ErrorResponse(:final error, :final requestId):
        debugPrint('[LlamaParent] ✗ Error: $error');
        if (requestId != null) {
          _completeRequest(requestId, null);
          _closeStream(requestId);
        }

      case CompleteResponse(:final result, :final requestId, :final tokensGenerated):
        debugPrint('[LlamaParent] ✓ Complete: $tokensGenerated tokens (request: $requestId)');
        _completeRequest(requestId, result);
        _closeStream(requestId);

      case TokenResponse(:final token, :final requestId):
        _streamToken(requestId, token);
    }
  }

  // ==========================================================================
  // Public API
  // ==========================================================================

  /// Initialize the llama.cpp backend
  Future<bool> initBackend() async {
    _ensureStarted();
    debugPrint('[LlamaParent] Initializing backend...');

    _sendCommand(InitBackendCommand());

    // Wait for status to become ready
    await _waitForStatus(IsolateStatus.ready);
    return _status == IsolateStatus.ready;
  }

  /// Load a model from file
  Future<bool> loadModel(String modelPath, {ModelConfig? config}) async {
    _ensureStarted();
    debugPrint('[LlamaParent] Loading model: $modelPath');

    _sendCommand(LoadModelCommand(modelPath: modelPath, config: config ?? const ModelConfig()));

    // Wait for success/error response
    await Future.delayed(const Duration(milliseconds: 500));
    return true; // TODO: Track success/failure properly
  }

  /// Create an inference context
  Future<bool> createContext({ContextConfig? config}) async {
    _ensureStarted();
    debugPrint('[LlamaParent] Creating context...');

    _sendCommand(CreateContextCommand(config: config ?? const ContextConfig()));

    // Wait for status to become ready
    await _waitForStatus(IsolateStatus.ready);
    return _status == IsolateStatus.ready;
  }

  /// Generate text (non-streaming)
  Future<String?> generate(
    String prompt, {
    SamplerConfig? config,
    String? systemPrompt = 'You are a helpful, concise assistant.',
  }) async {
    _ensureStarted();

    final requestId = _generateRequestId();
    final completer = Completer<String?>();
    _pendingRequests[requestId] = completer;

    debugPrint('[LlamaParent] Generating (request: $requestId)...');

    _sendCommand(
      GenerateCommand(
        prompt: prompt,
        config: config ?? const SamplerConfig(),
        requestId: requestId,
        systemPrompt: systemPrompt,
      ),
    );

    return completer.future;
  }

  /// Generate text (streaming)
  Stream<String> generateStream(
    String prompt, {
    SamplerConfig? config,
    String? systemPrompt = 'You are a helpful, concise assistant.',
  }) {
    _ensureStarted();

    final requestId = _generateRequestId();
    final controller = StreamController<String>();
    _streamControllers[requestId] = controller;

    debugPrint('[LlamaParent] Streaming (request: $requestId)...');

    _sendCommand(
      GenerateStreamCommand(
        prompt: prompt,
        config: config ?? const SamplerConfig(),
        requestId: requestId,
        systemPrompt: systemPrompt,
      ),
    );

    return controller.stream;
  }

  /// Stop ongoing generation
  void stopGeneration([String? requestId]) {
    _ensureStarted();

    debugPrint('[LlamaParent] Stopping generation...');
    _sendCommand(StopGenerationCommand(requestId: requestId ?? ''));
  }

  /// Clear the context (reset conversation state)
  void clearContext() {
    _ensureStarted();

    debugPrint('[LlamaParent] Clearing context...');
    _sendCommand(ClearContextCommand());
  }

  /// Dispose all resources and shutdown
  void dispose() {
    debugPrint('[LlamaParent] Disposing...');

    if (_sendPort != null) {
      _sendCommand(DisposeCommand());
    }

    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _sendPort = null;
    _receivePort.close();
    _statusController.close();

    // Complete all pending requests with null
    for (final completer in _pendingRequests.values) {
      if (!completer.isCompleted) {
        completer.complete(null);
      }
    }
    _pendingRequests.clear();

    // Close all stream controllers
    for (final controller in _streamControllers.values) {
      controller.close();
    }
    _streamControllers.clear();

    debugPrint('[LlamaParent] ✓ Disposed');
  }

  // ==========================================================================
  // Internal Helpers
  // ==========================================================================

  void _ensureStarted() {
    if (_sendPort == null) {
      throw StateError('Isolate not started. Call start() first.');
    }
  }

  void _sendCommand(LlamaCommand command) {
    _sendPort!.send(command);
  }

  String _generateRequestId() {
    return 'req_${_requestCounter++}_${DateTime.now().millisecondsSinceEpoch}';
  }

  void _completeRequest(String requestId, String? result) {
    final completer = _pendingRequests.remove(requestId);
    if (completer != null && !completer.isCompleted) {
      completer.complete(result);
    }
  }

  void _streamToken(String requestId, String token) {
    final controller = _streamControllers[requestId];
    if (controller != null && !controller.isClosed) {
      controller.add(token);
    }
  }

  void _closeStream(String requestId) {
    final controller = _streamControllers.remove(requestId);
    if (controller != null && !controller.isClosed) {
      controller.close();
    }
  }

  Future<void> _waitForStatus(IsolateStatus targetStatus, {Duration timeout = const Duration(seconds: 30)}) async {
    if (_status == targetStatus) return;

    await statusStream.firstWhere((status) => status == targetStatus).timeout(timeout);
  }
}
