/// Advanced text generation patterns for LlamaService
///
/// Provides stepwise generation, streaming, one-shot completion,
/// and various generation patterns with proper context management.

import 'dart:async';
import 'dart:ffi';
import 'dart:math' as math;
import 'package:ffi/ffi.dart';
import '../ffi/llama_ffi.dart';
import '../core/llama_exceptions.dart';
import '../utils/llama_config.dart';

/// Result of a single generation step
class GenerationStep {
  final String text;
  final int token;
  final bool isDone;
  final bool isContextLimit;
  final bool isStopString;
  final int currentPosition;
  final int remainingContext;

  const GenerationStep({
    required this.text,
    required this.token,
    required this.isDone,
    required this.isContextLimit,
    required this.isStopString,
    required this.currentPosition,
    required this.remainingContext,
  });

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'token': token,
      'isDone': isDone,
      'isContextLimit': isContextLimit,
      'isStopString': isStopString,
      'currentPosition': currentPosition,
      'remainingContext': remainingContext,
    };
  }
}

/// Status information for generation
class GenerationStatus {
  final int tokensGenerated;
  final int totalTokens;
  final double tokensPerSecond;
  final Duration elapsed;
  final bool isComplete;
  final String? stopReason;

  const GenerationStatus({
    required this.tokensGenerated,
    required this.totalTokens,
    required this.tokensPerSecond,
    required this.elapsed,
    required this.isComplete,
    this.stopReason,
  });

  /// Progress as a percentage (0.0 to 1.0)
  double get progress => totalTokens > 0 ? tokensGenerated / totalTokens : 0.0;
}

/// Advanced text generator with multiple generation patterns
class LlamaGenerator with DisposableMixin {
  final llama_cpp _llamaCpp;
  final Pointer<llama_context> _context;
  final Pointer<llama_vocab> _vocab;
  final Pointer<llama_sampler> _sampler;

  // Generation state
  List<int>? _currentTokens;
  int _currentPosition = 0;
  final Stopwatch _generationTimer = Stopwatch();

  LlamaGenerator(this._llamaCpp, this._context, this._vocab, this._sampler);

  /// Initialize generation with a prompt
  Future<bool> initializeGeneration(String prompt, {SamplerConfig? config}) async {
    checkNotDisposed('initializeGeneration');
    config ??= const SamplerConfig();

    try {
      // Tokenize prompt
      _currentTokens = await _tokenizePrompt(prompt);
      if (_currentTokens == null || _currentTokens!.isEmpty) {
        throw LlamaTokenizationException(prompt, 'Failed to tokenize prompt');
      }

      // Validate context capacity
      LlamaSafety.validateContextCapacity(
        currentPos: 0,
        additionalTokens: _currentTokens!.length + config.maxTokens,
        maxCtx: _llamaCpp.llama_n_ctx(_context),
        operation: 'initializeGeneration',
      );

      // Process prompt batch
      final tokensPtr = malloc<llama_token>(_currentTokens!.length);
      try {
        for (int i = 0; i < _currentTokens!.length; i++) {
          tokensPtr[i] = _currentTokens![i];
        }

        final batch = _llamaCpp.llama_batch_get_one(tokensPtr, _currentTokens!.length);

        if (_llamaCpp.llama_decode(_context, batch) != 0) {
          throw LlamaException('Failed to decode prompt batch');
        }

        _currentPosition = _currentTokens!.length;
        _generationTimer.reset();
        _generationTimer.start();

        print('✓ Generation initialized with ${_currentTokens!.length} prompt tokens');
        return true;
      } finally {
        malloc.free(tokensPtr);
      }
    } catch (e) {
      if (e is LlamaException) rethrow;
      throw LlamaException('Failed to initialize generation: $e');
    }
  }

  /// Generate the next token (stepwise generation)
  Future<GenerationStep?> getNext({SamplerConfig? config}) async {
    checkNotDisposed('getNext');
    config ??= const SamplerConfig();

    if (_currentTokens == null) {
      throw LlamaException('Generation not initialized. Call initializeGeneration() first.');
    }

    try {
      // Check context limits
      final maxCtx = _llamaCpp.llama_n_ctx(_context);
      if (_currentPosition >= maxCtx) {
        return GenerationStep(
          text: '',
          token: -1,
          isDone: true,
          isContextLimit: true,
          isStopString: false,
          currentPosition: _currentPosition,
          remainingContext: 0,
        );
      }

      // Sample next token
      final newToken = _llamaCpp.llama_sampler_sample(_sampler, _context, -1);

      // Check if end of generation
      if (_llamaCpp.llama_token_is_eog(_vocab, newToken)) {
        return GenerationStep(
          text: '',
          token: newToken,
          isDone: true,
          isContextLimit: false,
          isStopString: false,
          currentPosition: _currentPosition,
          remainingContext: maxCtx - _currentPosition,
        );
      }

      // Detokenize
      final tokenText = await _detokenizeSingle(newToken);

      // Check for stop strings
      // Note: This is simplified - in practice you'd want to check accumulated text
      final isStopString = config.stopStrings.any((stop) => tokenText.contains(stop));

      // Process the token
      final singleTokenPtr = malloc<llama_token>();
      try {
        singleTokenPtr[0] = newToken;
        final batch = _llamaCpp.llama_batch_get_one(singleTokenPtr, 1);

        if (_llamaCpp.llama_decode(_context, batch) != 0) {
          throw LlamaException('Failed to decode token at position $_currentPosition');
        }

        _currentTokens!.add(newToken);
        _currentPosition++;

        return GenerationStep(
          text: tokenText,
          token: newToken,
          isDone: isStopString,
          isContextLimit: false,
          isStopString: isStopString,
          currentPosition: _currentPosition,
          remainingContext: maxCtx - _currentPosition,
        );
      } finally {
        malloc.free(singleTokenPtr);
      }
    } catch (e) {
      if (e is LlamaException) rethrow;
      throw LlamaException('Failed to generate next token: $e');
    }
  }

  /// Generate next token with detailed status
  Future<(GenerationStep?, GenerationStatus)> getNextWithStatus({SamplerConfig? config}) async {
    final startTime = DateTime.now();
    final step = await getNext(config: config);
    final endTime = DateTime.now();

    final elapsed = endTime.difference(startTime);
    final tokensGenerated = _currentPosition - (_currentTokens?.length ?? 0);
    final tokensPerSecond = elapsed.inMilliseconds > 0 ? (tokensGenerated * 1000.0) / elapsed.inMilliseconds : 0.0;

    final status = GenerationStatus(
      tokensGenerated: tokensGenerated,
      totalTokens: config?.maxTokens ?? 100,
      tokensPerSecond: tokensPerSecond,
      elapsed: _generationTimer.elapsed,
      isComplete: step?.isDone ?? false,
      stopReason: step?.isContextLimit == true
          ? 'context_limit'
          : step?.isStopString == true
          ? 'stop_string'
          : step?.isDone == true
          ? 'eog'
          : null,
    );

    return (step, status);
  }

  /// Stream generation (yields tokens as they're generated)
  Stream<GenerationStep> generateStream(String prompt, {SamplerConfig? config}) async* {
    config ??= const SamplerConfig();

    // Initialize generation
    final initialized = await initializeGeneration(prompt, config: config);
    if (!initialized) {
      throw LlamaException('Failed to initialize streaming generation');
    }

    // Generate tokens one by one
    int generated = 0;
    while (generated < config.maxTokens) {
      final step = await getNext(config: config);
      if (step == null) break;

      yield step;
      generated++;

      if (step.isDone) break;
    }
  }

  /// One-shot complete text generation
  Future<String?> generateCompleteText(
    String prompt, {
    SamplerConfig? config,
    void Function(GenerationStatus)? onProgress,
  }) async {
    config ??= const SamplerConfig();

    final buffer = StringBuffer();
    int generated = 0;

    await for (final step in generateStream(prompt, config: config)) {
      buffer.write(step.text);
      generated++;

      // Report progress if callback provided
      if (onProgress != null) {
        final status = GenerationStatus(
          tokensGenerated: generated,
          totalTokens: config.maxTokens,
          tokensPerSecond: _generationTimer.elapsedMilliseconds > 0
              ? (generated * 1000.0) / _generationTimer.elapsedMilliseconds
              : 0.0,
          elapsed: _generationTimer.elapsed,
          isComplete: step.isDone,
          stopReason: step.isContextLimit
              ? 'context_limit'
              : step.isStopString
              ? 'stop_string'
              : step.isDone
              ? 'eog'
              : null,
        );
        onProgress(status);
      }

      if (step.isDone) break;
    }

    return buffer.toString();
  }

  /// Get remaining context space
  int getRemainingContextSpace() {
    checkNotDisposed('getRemainingContextSpace');
    final maxCtx = _llamaCpp.llama_n_ctx(_context);
    return maxCtx - _currentPosition;
  }

  /// Check if context limit was reached
  bool wasContextLimitReached() {
    checkNotDisposed('wasContextLimitReached');
    return _currentPosition >= _llamaCpp.llama_n_ctx(_context);
  }

  /// Reset generation state (clear KV cache)
  Future<void> reset() async {
    checkNotDisposed('reset');

    _currentTokens = null;
    _currentPosition = 0;
    _generationTimer.reset();

    // Clear KV cache - this would need the memory management functions
    // For now, we'll just reset our tracking
    print('✓ Generation state reset');
  }

  /// Get current generation statistics
  GenerationStatus getCurrentStatus() {
    checkNotDisposed('getCurrentStatus');

    final tokensGenerated = _currentTokens != null ? _currentPosition - _currentTokens!.length : 0;

    return GenerationStatus(
      tokensGenerated: tokensGenerated,
      totalTokens: 0, // Would need to track target
      tokensPerSecond: _generationTimer.elapsedMilliseconds > 0
          ? (tokensGenerated * 1000.0) / _generationTimer.elapsedMilliseconds
          : 0.0,
      elapsed: _generationTimer.elapsed,
      isComplete: false,
    );
  }

  /// Tokenize prompt for generation
  Future<List<int>?> _tokenizePrompt(String prompt) async {
    final textPtr = prompt.toNativeUtf8();
    try {
      // First pass: get token count
      final tokenCount = -_llamaCpp.llama_tokenize(
        _vocab,
        textPtr.cast<Char>(),
        prompt.length,
        nullptr,
        0,
        true, // add BOS
        true, // parse special
      );

      if (tokenCount <= 0) return null;

      // Second pass: get tokens
      final tokensPtr = malloc<llama_token>(tokenCount);
      try {
        final actualCount = _llamaCpp.llama_tokenize(
          _vocab,
          textPtr.cast<Char>(),
          prompt.length,
          tokensPtr,
          tokenCount,
          true,
          true,
        );

        if (actualCount < 0) return null;

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

  /// Detokenize a single token
  Future<String> _detokenizeSingle(int token) async {
    final bufferPtr = malloc<Int8>(128);
    try {
      final length = _llamaCpp.llama_token_to_piece(
        _vocab,
        token,
        bufferPtr.cast<Char>(),
        128,
        0, // lstrip
        true, // special
      );

      if (length <= 0) return '';

      final bytes = bufferPtr.cast<Uint8>().asTypedList(length);
      return String.fromCharCodes(bytes);
    } finally {
      malloc.free(bufferPtr);
    }
  }

  void dispose() {
    _generationTimer.stop();
    _currentTokens = null;
    _currentPosition = 0;
    markDisposed();
    print('✓ Generator disposed');
  }
}

/// Controller for streaming text generation
class StreamingGenerationController {
  final StreamController<GenerationStep> _controller = StreamController<GenerationStep>();
  final LlamaGenerator _generator;
  bool _isGenerating = false;
  bool _isPaused = false;

  StreamingGenerationController(this._generator);

  /// Stream of generation steps
  Stream<GenerationStep> get stream => _controller.stream;

  /// Whether generation is currently active
  bool get isGenerating => _isGenerating;

  /// Whether generation is paused
  bool get isPaused => _isPaused;

  /// Start streaming generation
  Future<void> start(String prompt, {SamplerConfig? config}) async {
    if (_isGenerating) {
      throw LlamaException('Generation already in progress');
    }

    _isGenerating = true;
    config ??= const SamplerConfig();

    try {
      await for (final step in _generator.generateStream(prompt, config: config)) {
        if (_isPaused) {
          await Future.delayed(const Duration(milliseconds: 100));
          continue;
        }

        _controller.add(step);

        if (step.isDone) break;
      }
    } catch (e) {
      _controller.addError(e);
    } finally {
      _isGenerating = false;
      _controller.close();
    }
  }

  /// Pause generation
  void pause() {
    _isPaused = true;
  }

  /// Resume generation
  void resume() {
    _isPaused = false;
  }

  /// Stop generation
  void stop() {
    _isGenerating = false;
    _isPaused = false;
    _controller.close();
  }

  /// Dispose the controller
  void dispose() {
    stop();
  }
}

/// Utility class for text chunking and processing
class TextChunker {
  /// Split text into chunks with optional overlap
  static List<String> chunkText(
    String text, {
    int maxChunkSize = 1000,
    int overlapSize = 100,
    ChunkingStrategy strategy = ChunkingStrategy.sentence,
  }) {
    switch (strategy) {
      case ChunkingStrategy.character:
        return _chunkByCharacter(text, maxChunkSize, overlapSize);
      case ChunkingStrategy.word:
        return _chunkByWord(text, maxChunkSize, overlapSize);
      case ChunkingStrategy.sentence:
        return _chunkBySentence(text, maxChunkSize, overlapSize);
      case ChunkingStrategy.paragraph:
        return _chunkByParagraph(text, maxChunkSize, overlapSize);
    }
  }

  static List<String> _chunkByCharacter(String text, int maxSize, int overlap) {
    final chunks = <String>[];
    int start = 0;

    while (start < text.length) {
      final end = math.min(start + maxSize, text.length);
      chunks.add(text.substring(start, end));
      start = end - overlap;
      if (start <= 0) start = end;
    }

    return chunks;
  }

  static List<String> _chunkByWord(String text, int maxSize, int overlap) {
    final words = text.split(RegExp(r'\s+'));
    final chunks = <String>[];
    int start = 0;

    while (start < words.length) {
      final chunk = <String>[];
      int currentSize = 0;
      int i = start;

      while (i < words.length && currentSize + words[i].length <= maxSize) {
        chunk.add(words[i]);
        currentSize += words[i].length + 1; // +1 for space
        i++;
      }

      if (chunk.isNotEmpty) {
        chunks.add(chunk.join(' '));
        start = math.max(start + chunk.length - overlap, start + 1);
      } else {
        start++;
      }
    }

    return chunks;
  }

  static List<String> _chunkBySentence(String text, int maxSize, int overlap) {
    final sentences = text.split(RegExp(r'[.!?]+\s*'));
    final chunks = <String>[];
    int start = 0;

    while (start < sentences.length) {
      final chunk = <String>[];
      int currentSize = 0;
      int i = start;

      while (i < sentences.length && currentSize + sentences[i].length <= maxSize) {
        chunk.add(sentences[i]);
        currentSize += sentences[i].length + 2; // +2 for punctuation and space
        i++;
      }

      if (chunk.isNotEmpty) {
        chunks.add(chunk.join('. ') + (chunk.length > 1 ? '.' : ''));
        start = math.max(start + chunk.length - overlap, start + 1);
      } else {
        start++;
      }
    }

    return chunks;
  }

  static List<String> _chunkByParagraph(String text, int maxSize, int overlap) {
    final paragraphs = text.split(RegExp(r'\n\s*\n'));
    final chunks = <String>[];
    int start = 0;

    while (start < paragraphs.length) {
      final chunk = <String>[];
      int currentSize = 0;
      int i = start;

      while (i < paragraphs.length && currentSize + paragraphs[i].length <= maxSize) {
        chunk.add(paragraphs[i]);
        currentSize += paragraphs[i].length + 2; // +2 for newlines
        i++;
      }

      if (chunk.isNotEmpty) {
        chunks.add(chunk.join('\n\n'));
        start = math.max(start + chunk.length - overlap, start + 1);
      } else {
        start++;
      }
    }

    return chunks;
  }
}

/// Strategies for text chunking
enum ChunkingStrategy { character, word, sentence, paragraph }

/// Post-processing filter for generated text
class SequenceFilter {
  /// Remove unwanted patterns from generated text
  static String filterText(
    String text, {
    bool removeExtraWhitespace = true,
    bool removeRepeatedPhrases = true,
    bool fixCapitalization = false,
    List<String> removePatterns = const [],
  }) {
    String filtered = text;

    // Remove extra whitespace
    if (removeExtraWhitespace) {
      filtered = filtered.replaceAll(RegExp(r'\s+'), ' ').trim();
    }

    // Remove repeated phrases (simple implementation)
    if (removeRepeatedPhrases) {
      filtered = _removeRepeatedPhrases(filtered);
    }

    // Fix capitalization
    if (fixCapitalization) {
      filtered = _fixCapitalization(filtered);
    }

    // Remove custom patterns
    for (final pattern in removePatterns) {
      filtered = filtered.replaceAll(RegExp(pattern), '');
    }

    return filtered;
  }

  static String _removeRepeatedPhrases(String text) {
    // Simple implementation - remove exact repeated sentences
    final sentences = text.split(RegExp(r'[.!?]+'));
    final seen = <String>{};
    final filtered = <String>[];

    for (final sentence in sentences) {
      final trimmed = sentence.trim();
      if (trimmed.isNotEmpty && !seen.contains(trimmed)) {
        seen.add(trimmed);
        filtered.add(sentence);
      }
    }

    return filtered.join('.');
  }

  static String _fixCapitalization(String text) {
    if (text.isEmpty) return text;

    // Capitalize first letter
    String result = text[0].toUpperCase() + text.substring(1);

    // Capitalize after sentence endings
    result = result.replaceAllMapped(
      RegExp(r'([.!?]\s+)([a-z])'),
      (match) => match.group(1)! + match.group(2)!.toUpperCase(),
    );

    return result;
  }
}
