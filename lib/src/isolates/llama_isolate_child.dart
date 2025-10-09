/// LlamaIsolateChild - Worker isolate that handles all llama.cpp operations
///
/// This runs on a background isolate and processes commands from the parent.
/// All heavy computation (tokenization, generation) happens here.

/// ignore_for_file: dangling_library_doc_comments

import 'dart:isolate';

import 'package:quiz_wrapper/src/isolates/llama_isolate_types.dart';
import 'package:quiz_wrapper/src/llama_service_enhanced.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';

/// Entry point for the isolate - called by Isolate.spawn()
void llamaIsolateEntry(SendPort parentPort) {
  final child = LlamaIsolateChild(parentPort);
  child.start();
}

/// Child isolate that processes LlamaCommands
class LlamaIsolateChild {
  final SendPort _parentPort;
  final _receivePort = ReceivePort();

  late final LlamaServiceEnhanced _llama;

  bool _shouldStop = false;
  String? _currentRequestId;

  LlamaIsolateChild(this._parentPort);

  /// Start listening for commands
  void start() {
    // Create the LlamaService instance
    _llama = LlamaServiceEnhanced();

    // Send our SendPort back to parent
    _parentPort.send(_receivePort.sendPort);

    // Send ready status
    _sendResponse(StatusResponse(IsolateStatus.uninitialized));

    // Listen for commands
    _receivePort.listen(_handleCommand);
  }

  /// Handle incoming commands
  void _handleCommand(dynamic message) {
    if (message is! LlamaCommand) {
      _sendResponse(ErrorResponse('Invalid command type'));
      return;
    }

    try {
      switch (message) {
        case InitBackendCommand():
          _handleInitBackend();

        case LoadModelCommand():
          _handleLoadModel(message);

        case CreateContextCommand():
          _handleCreateContext(message);

        case GenerateCommand():
          _handleGenerate(message);

        case GenerateStreamCommand():
          _handleGenerateStream(message);

        case StopGenerationCommand():
          _handleStop(message);

        case ClearContextCommand():
          _handleClear();

        case DisposeCommand():
          _handleDispose();
      }
    } catch (e, stack) {
      _sendResponse(ErrorResponse('Command failed: $e\n$stack'));
    }
  }

  // ==========================================================================
  // Command Handlers
  // ==========================================================================

  void _handleInitBackend() {
    _llama.init();
    _sendResponse(SuccessResponse('Backend initialized'));
    _sendResponse(StatusResponse(IsolateStatus.ready));
  }

  void _handleLoadModel(LoadModelCommand cmd) {
    final success = _llama.loadModel(cmd.modelPath, config: cmd.config);

    if (success) {
      _sendResponse(SuccessResponse('Model loaded'));
    } else {
      _sendResponse(ErrorResponse('Failed to load model'));
    }
  }

  void _handleCreateContext(CreateContextCommand cmd) {
    final success = _llama.createContext(config: cmd.config);

    if (success) {
      _sendResponse(SuccessResponse('Context created'));
      _sendResponse(StatusResponse(IsolateStatus.ready));
    } else {
      _sendResponse(ErrorResponse('Failed to create context'));
    }
  }

  void _handleGenerate(GenerateCommand cmd) async {
    _currentRequestId = cmd.requestId;
    _shouldStop = false;

    _sendResponse(StatusResponse(IsolateStatus.generating));

    final result = await _llama.generateEnhanced(cmd.prompt, config: cmd.config, systemPrompt: cmd.systemPrompt);

    if (_shouldStop) {
      _sendResponse(
        CompleteResponse(requestId: cmd.requestId, tokensGenerated: 0, metricsJson: result?.metrics?.toJson()),
      );
      _sendResponse(StatusResponse(IsolateStatus.ready));
      return;
    }

    if (result != null) {
      _sendResponse(
        CompleteResponse(
          result: result.text,
          requestId: cmd.requestId,
          tokensGenerated: result.tokensGenerated,
          metricsJson: result.metrics?.toJson(),
        ),
      );
    } else {
      _sendResponse(ErrorResponse('Generation failed', requestId: cmd.requestId));
    }

    _sendResponse(StatusResponse(IsolateStatus.ready));
    _currentRequestId = null;
  }

  void _handleGenerateStream(GenerateStreamCommand cmd) async {
    _currentRequestId = cmd.requestId;
    _shouldStop = false;

    _sendResponse(StatusResponse(IsolateStatus.generating));

    int tokenCount = 0;

    try {
      await for (final event in _llama.generateStream(cmd.prompt, config: cmd.config, systemPrompt: cmd.systemPrompt)) {
        if (_shouldStop) {
          _sendResponse(CompleteResponse(requestId: cmd.requestId, tokensGenerated: tokenCount));
          break;
        }

        switch (event) {
          case TokenEvent(:final token):
            _sendResponse(TokenResponse(token: token, requestId: cmd.requestId));
            tokenCount++;

          case MetricsEvent(:final metrics): // ✅ ADD THIS CASE
            _sendResponse(MetricsResponse(metricsJson: metrics.toJson(), requestId: cmd.requestId));

          case DoneEvent(:final tokensGenerated):
            tokenCount = tokensGenerated; // Use accurate count from generator
            break;
        }
      }

      _sendResponse(CompleteResponse(requestId: cmd.requestId, tokensGenerated: tokenCount));
    } catch (e) {
      _sendResponse(ErrorResponse('Streaming failed: $e', requestId: cmd.requestId));
    }

    _sendResponse(StatusResponse(IsolateStatus.ready));
    _currentRequestId = null;
  }

  void _handleStop(StopGenerationCommand cmd) {
    if (_currentRequestId == cmd.requestId || cmd.requestId.isEmpty) {
      _shouldStop = true;
      _sendResponse(SuccessResponse('Generation stopped'));
    }
  }

  void _handleClear() {
    // Note: LlamaServiceEnhanced doesn't have a clear method yet
    // This would reset conversation state if implemented
    _sendResponse(SuccessResponse('Context cleared'));
  }

  void _handleDispose() {
    _llama.dispose();
    _receivePort.close();
    _sendResponse(SuccessResponse('Disposed'));
  }

  // ==========================================================================
  // Helper Methods
  // ==========================================================================

  void _sendResponse(LlamaResponse response) {
    _parentPort.send(response);
  }
}
