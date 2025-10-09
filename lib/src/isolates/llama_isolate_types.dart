import 'package:quiz_wrapper/src/utils/llama_config.dart';
import 'package:quiz_wrapper/src/utils/llama_helpers.dart';

/// Type-safe message definitions for isolate communication
///
/// Uses sealed classes for compile-time exhaustiveness checking

// ignore_for_file: dangling_library_doc_comments

// ============================================================================
// COMMANDS - Sent from UI thread to isolate
// ============================================================================

/// Base class for all commands sent to the LlamaIsolate
sealed class LlamaCommand {}

/// Initialize the llama.cpp backend
class InitBackendCommand extends LlamaCommand {}

/// Load a model from file
class LoadModelCommand extends LlamaCommand {
  final String modelPath;
  final ModelConfig config;

  LoadModelCommand({required this.modelPath, required this.config});
}

/// Create an inference context
class CreateContextCommand extends LlamaCommand {
  final ContextConfig config;

  CreateContextCommand({required this.config});
}

/// Generate text (non-streaming)
class GenerateCommand extends LlamaCommand {
  final String prompt;
  final SamplerConfig config;
  final String requestId; // For tracking this specific request
  final String? systemPrompt;

  GenerateCommand({required this.prompt, required this.config, required this.requestId, this.systemPrompt});
}

/// Generate text (streaming)
class GenerateStreamCommand extends LlamaCommand {
  final String prompt;
  final SamplerConfig config;
  final String requestId;
  final String? systemPrompt;

  GenerateStreamCommand({required this.prompt, required this.config, required this.requestId, this.systemPrompt});
}

/// Stop ongoing generation
class StopGenerationCommand extends LlamaCommand {
  final String requestId; // Optional: stop specific request

  StopGenerationCommand({required this.requestId});
}

/// Clear the context (reset conversation state)
class ClearContextCommand extends LlamaCommand {}

/// Dispose all resources and shutdown
class DisposeCommand extends LlamaCommand {}

// ============================================================================
// RESPONSES - Sent from isolate to UI thread
// ============================================================================

/// Base class for all responses from the LlamaIsolate
sealed class LlamaResponse {}

/// Confirmation that an operation completed successfully
class SuccessResponse extends LlamaResponse {
  final String? message;

  SuccessResponse([this.message]);
}

/// An error occurred
class ErrorResponse extends LlamaResponse {
  final String error;
  final String? requestId;

  ErrorResponse(this.error, {this.requestId});
}

/// A single token was generated (streaming)
class TokenResponse extends LlamaResponse {
  final String token;
  final String requestId;

  TokenResponse({required this.token, required this.requestId});
}

/// A metrics event was generated (streaming)
class MetricsResponse extends LlamaResponse {
  final Map<String, dynamic> metricsJson; // ✅ JSON for isolate transfer
  final String requestId;

  MetricsResponse({required this.metricsJson, required this.requestId});

  // ✅ Helper to reconstruct metrics
  PerformanceMetrics get metrics => PerformanceMetrics.fromJson(metricsJson);
}

/// Generation completed
class CompleteResponse extends LlamaResponse {
  final String? result;
  final String requestId;
  final int tokensGenerated;
  final Map<String, dynamic>? metricsJson; // ✅ ADD THIS

  CompleteResponse({
    this.result,
    required this.requestId,
    required this.tokensGenerated,
    this.metricsJson, // ✅ ADD THIS
  });

  // ✅ ADD: Helper to get metrics
  PerformanceMetrics? get metrics => metricsJson != null ? PerformanceMetrics.fromJson(metricsJson!) : null;
}

/// Status update
class StatusResponse extends LlamaResponse {
  final IsolateStatus status;
  final String? message;

  StatusResponse(this.status, {this.message});
}

// ============================================================================
// ENUMS
// ============================================================================

/// Status of the isolate
enum IsolateStatus { uninitialized, ready, generating, error }
