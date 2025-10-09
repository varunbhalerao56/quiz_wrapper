/// Enhanced tokenization utilities for LlamaService
///
/// Provides advanced tokenization, detokenization, token analysis,
/// and text processing capabilities.

import 'dart:ffi';
import 'package:ffi/ffi.dart';
import '../ffi/llama_ffi.dart';
import '../core/llama_exceptions.dart';

/// Result of tokenization operation
class TokenizationResult {
  final List<int> tokens;
  final String originalText;
  final bool addedBos;
  final bool parsedSpecial;
  final Duration processingTime;

  const TokenizationResult({
    required this.tokens,
    required this.originalText,
    required this.addedBos,
    required this.parsedSpecial,
    required this.processingTime,
  });

  /// Number of tokens
  int get length => tokens.length;

  /// Whether tokenization was successful
  bool get isSuccess => tokens.isNotEmpty;

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'tokens': tokens,
      'originalText': originalText,
      'addedBos': addedBos,
      'parsedSpecial': parsedSpecial,
      'processingTimeMs': processingTime.inMilliseconds,
      'length': length,
    };
  }
}

/// Result of detokenization operation
class DetokenizationResult {
  final String text;
  final List<int> originalTokens;
  final bool removedSpecial;
  final bool unparsedSpecial;
  final Duration processingTime;

  const DetokenizationResult({
    required this.text,
    required this.originalTokens,
    required this.removedSpecial,
    required this.unparsedSpecial,
    required this.processingTime,
  });

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'originalTokens': originalTokens,
      'removedSpecial': removedSpecial,
      'unparsedSpecial': unparsedSpecial,
      'processingTimeMs': processingTime.inMilliseconds,
    };
  }
}

/// Advanced tokenizer with enhanced functionality
class LlamaTokenizer {
  final llama_cpp _llamaCpp;
  final Pointer<llama_vocab> _vocab;

  LlamaTokenizer(this._llamaCpp, this._vocab);

  /// Enhanced tokenization with detailed result
  Future<TokenizationResult> tokenizeAdvanced(String text, {bool addBos = true, bool parseSpecial = true}) async {
    final stopwatch = Stopwatch()..start();

    try {
      final tokens = await _tokenizeText(text, addBos, parseSpecial);
      stopwatch.stop();

      return TokenizationResult(
        tokens: tokens,
        originalText: text,
        addedBos: addBos,
        parsedSpecial: parseSpecial,
        processingTime: stopwatch.elapsed,
      );
    } catch (e) {
      stopwatch.stop();
      throw LlamaTokenizationException(text, e.toString());
    }
  }

  /// Enhanced detokenization with detailed result
  Future<DetokenizationResult> detokenizeAdvanced(
    List<int> tokens, {
    bool removeSpecial = false,
    bool unparseSpecial = false,
  }) async {
    final stopwatch = Stopwatch()..start();

    try {
      final text = await _detokenizeTokens(tokens, removeSpecial, unparseSpecial);
      stopwatch.stop();

      return DetokenizationResult(
        text: text,
        originalTokens: tokens,
        removedSpecial: removeSpecial,
        unparsedSpecial: unparseSpecial,
        processingTime: stopwatch.elapsed,
      );
    } catch (e) {
      stopwatch.stop();
      throw LlamaException('Detokenization failed: $e', operation: 'detokenize');
    }
  }

  /// Tokenize text for streaming display (piece by piece)
  Stream<String> tokenizeForDisplay(String text) async* {
    final tokens = await _tokenizeText(text, true, true);

    for (final token in tokens) {
      final piece = await tokenToPiece(token);
      yield piece;
    }
  }

  /// Convert single token to text piece
  Future<String> tokenToPiece(int token, {bool includeSpecial = true}) async {
    final bufferPtr = malloc<Int8>(128);
    try {
      final length = _llamaCpp.llama_token_to_piece(
        _vocab,
        token,
        bufferPtr.cast<Char>(),
        128,
        0, // lstrip
        includeSpecial,
      );

      if (length <= 0) return '';

      final bytes = bufferPtr.cast<Uint8>().asTypedList(length);
      return String.fromCharCodes(bytes);
    } finally {
      malloc.free(bufferPtr);
    }
  }

  /// Get special token IDs
  Map<String, int> getSpecialTokens() {
    return {
      'bos': _llamaCpp.llama_token_bos(_vocab),
      'eos': _llamaCpp.llama_token_eos(_vocab),
      'eot': _llamaCpp.llama_token_eot(_vocab),
      'cls': _llamaCpp.llama_token_cls(_vocab),
      'sep': _llamaCpp.llama_token_sep(_vocab),
      'nl': _llamaCpp.llama_token_nl(_vocab),
      'pad': _llamaCpp.llama_token_pad(_vocab),
    };
  }

  /// Check if token is end-of-generation
  bool isEndOfGeneration(int token) {
    return _llamaCpp.llama_token_is_eog(_vocab, token);
  }

  /// Check if token is control token
  bool isControlToken(int token) {
    return _llamaCpp.llama_token_is_control(_vocab, token);
  }

  /// Get token attributes
  Map<String, bool> getTokenAttributes(int token) {
    final attr = _llamaCpp.llama_token_get_attr(_vocab, token);

    return {
      'undefined': (attr.value & 0) != 0,
      'unknown': (attr.value & 1) != 0,
      'unused': (attr.value & 2) != 0,
      'normal': (attr.value & 4) != 0,
      'control': (attr.value & 8) != 0,
      'user_defined': (attr.value & 16) != 0,
      'byte': (attr.value & 32) != 0,
      'normalized': (attr.value & 64) != 0,
      'lstrip': (attr.value & 128) != 0,
      'rstrip': (attr.value & 256) != 0,
      'single_word': (attr.value & 512) != 0,
    };
  }

  /// Get vocabulary size
  int getVocabSize() {
    return _llamaCpp.llama_n_vocab(_vocab);
  }

  /// Analyze tokenization (for debugging and optimization)
  Future<TokenAnalysis> analyzeTokenization(String text) async {
    final result = await tokenizeAdvanced(text);
    final specialTokens = getSpecialTokens();

    int specialCount = 0;
    int controlCount = 0;
    final tokenDetails = <TokenDetail>[];

    for (final token in result.tokens) {
      final isSpecial = specialTokens.values.contains(token);
      final isControl = isControlToken(token);
      final piece = await tokenToPiece(token);
      final attributes = getTokenAttributes(token);

      if (isSpecial) specialCount++;
      if (isControl) controlCount++;

      tokenDetails.add(
        TokenDetail(token: token, text: piece, isSpecial: isSpecial, isControl: isControl, attributes: attributes),
      );
    }

    return TokenAnalysis(
      originalText: text,
      tokens: result.tokens,
      tokenDetails: tokenDetails,
      specialTokenCount: specialCount,
      controlTokenCount: controlCount,
      averageTokenLength: text.length / result.tokens.length,
      compressionRatio: result.tokens.length / text.length,
    );
  }

  /// Internal tokenization method
  Future<List<int>> _tokenizeText(String text, bool addBos, bool parseSpecial) async {
    final textPtr = text.toNativeUtf8();
    try {
      // First pass: get token count
      final tokenCount = -_llamaCpp.llama_tokenize(
        _vocab,
        textPtr.cast<Char>(),
        text.length,
        nullptr,
        0,
        addBos,
        parseSpecial,
      );

      if (tokenCount <= 0) return [];

      // Second pass: get tokens
      final tokensPtr = malloc<llama_token>(tokenCount);
      try {
        final actualCount = _llamaCpp.llama_tokenize(
          _vocab,
          textPtr.cast<Char>(),
          text.length,
          tokensPtr,
          tokenCount,
          addBos,
          parseSpecial,
        );

        if (actualCount < 0) return [];

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

  /// Internal detokenization method
  Future<String> _detokenizeTokens(List<int> tokens, bool removeSpecial, bool unparseSpecial) async {
    if (tokens.isEmpty) return '';

    final tokensPtr = malloc<llama_token>(tokens.length);
    final textBuffer = malloc<Char>(tokens.length * 16); // Generous buffer

    try {
      for (int i = 0; i < tokens.length; i++) {
        tokensPtr[i] = tokens[i];
      }

      final length = _llamaCpp.llama_detokenize(
        _vocab,
        tokensPtr,
        tokens.length,
        textBuffer,
        tokens.length * 16,
        removeSpecial,
        unparseSpecial,
      );

      if (length < 0) {
        throw LlamaException('Detokenization buffer too small, needed: ${-length}');
      }

      if (length == 0) return '';

      return textBuffer.cast<Utf8>().toDartString();
    } finally {
      malloc.free(tokensPtr);
      malloc.free(textBuffer);
    }
  }
}

/// Detailed information about a single token
class TokenDetail {
  final int token;
  final String text;
  final bool isSpecial;
  final bool isControl;
  final Map<String, bool> attributes;

  const TokenDetail({
    required this.token,
    required this.text,
    required this.isSpecial,
    required this.isControl,
    required this.attributes,
  });

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {'token': token, 'text': text, 'isSpecial': isSpecial, 'isControl': isControl, 'attributes': attributes};
  }
}

/// Analysis result for tokenization
class TokenAnalysis {
  final String originalText;
  final List<int> tokens;
  final List<TokenDetail> tokenDetails;
  final int specialTokenCount;
  final int controlTokenCount;
  final double averageTokenLength;
  final double compressionRatio;

  const TokenAnalysis({
    required this.originalText,
    required this.tokens,
    required this.tokenDetails,
    required this.specialTokenCount,
    required this.controlTokenCount,
    required this.averageTokenLength,
    required this.compressionRatio,
  });

  /// Get summary statistics
  Map<String, dynamic> getSummary() {
    return {
      'originalLength': originalText.length,
      'tokenCount': tokens.length,
      'specialTokens': specialTokenCount,
      'controlTokens': controlTokenCount,
      'averageTokenLength': averageTokenLength.toStringAsFixed(2),
      'compressionRatio': compressionRatio.toStringAsFixed(3),
      'efficiency': (originalText.length / tokens.length).toStringAsFixed(2),
    };
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'originalText': originalText,
      'tokens': tokens,
      'tokenDetails': tokenDetails.map((d) => d.toJson()).toList(),
      'summary': getSummary(),
    };
  }
}
