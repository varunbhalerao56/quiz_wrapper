/// Session management and persistence for LlamaService
///
/// Provides session save/load functionality, state persistence,
/// and context management across application restarts.

// ignore_for_file: dangling_library_doc_comments

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:quiz_wrapper/src/chat/llama_chat.dart';
import 'package:quiz_wrapper/src/core/llama_exceptions.dart';
import 'package:quiz_wrapper/src/ffi/llama_ffi.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';



/// Represents a saved session state
class SessionData {
  final String sessionId;
  final DateTime createdAt;
  final DateTime lastUsedAt;
  final String modelPath;
  final LlamaServiceConfig config;
  final ChatHistory? chatHistory;
  final List<int> tokens;
  final Map<String, dynamic> metadata;

  const SessionData({
    required this.sessionId,
    required this.createdAt,
    required this.lastUsedAt,
    required this.modelPath,
    required this.config,
    this.chatHistory,
    required this.tokens,
    this.metadata = const {},
  });

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'sessionId': sessionId,
      'createdAt': createdAt.toIso8601String(),
      'lastUsedAt': lastUsedAt.toIso8601String(),
      'modelPath': modelPath,
      'config': config.toJson(),
      'chatHistory': chatHistory?.toJson(),
      'tokens': tokens,
      'metadata': metadata,
    };
  }

  /// Create from JSON
  factory SessionData.fromJson(Map<String, dynamic> json) {
    return SessionData(
      sessionId: json['sessionId'] as String,
      createdAt: DateTime.parse(json['createdAt'] as String),
      lastUsedAt: DateTime.parse(json['lastUsedAt'] as String),
      modelPath: json['modelPath'] as String,
      config: LlamaServiceConfig.fromJson(json['config']),
      chatHistory: json['chatHistory'] != null ? ChatHistory.fromJson(json['chatHistory']) : null,
      tokens: (json['tokens'] as List).cast<int>(),
      metadata: json['metadata'] as Map<String, dynamic>? ?? {},
    );
  }

  /// Create updated session with new timestamp
  SessionData copyWith({
    DateTime? lastUsedAt,
    ChatHistory? chatHistory,
    List<int>? tokens,
    Map<String, dynamic>? metadata,
  }) {
    return SessionData(
      sessionId: sessionId,
      createdAt: createdAt,
      lastUsedAt: lastUsedAt ?? DateTime.now(),
      modelPath: modelPath,
      config: config,
      chatHistory: chatHistory ?? this.chatHistory,
      tokens: tokens ?? this.tokens,
      metadata: metadata ?? this.metadata,
    );
  }
}

/// Session manager for handling persistence
class LlamaSessionManager {
  final llama_cpp llamaCpp;
  final String _sessionsDirectory;
  final Map<String, SessionData> _loadedSessions = {};

  LlamaSessionManager(this.llamaCpp, {String? sessionsDirectory})
    : _sessionsDirectory = sessionsDirectory ?? 'sessions';

  /// Initialize session manager (create directory if needed)
  Future<void> initialize() async {
    final dir = Directory(_sessionsDirectory);
    if (!await dir.exists()) {
      await dir.create(recursive: true);
      debugPrint('✓ Created sessions directory: $_sessionsDirectory');
    }
  }

  /// Save session state to file
  Future<bool> saveSession(
    String sessionId,
    Pointer<llama_context> context,
    String modelPath,
    LlamaServiceConfig config, {
    ChatHistory? chatHistory,
    List<int>? tokens,
    Map<String, dynamic>? metadata,
  }) async {
    try {
      // Get current tokens if not provided
      tokens ??= await _getCurrentTokens(context);

      // Create session data
      final sessionData = SessionData(
        sessionId: sessionId,
        createdAt: _loadedSessions[sessionId]?.createdAt ?? DateTime.now(),
        lastUsedAt: DateTime.now(),
        modelPath: modelPath,
        config: config,
        chatHistory: chatHistory,
        tokens: tokens,
        metadata: metadata ?? {},
      );

      // Save to JSON file
      final sessionFile = File('$_sessionsDirectory/$sessionId.json');
      await sessionFile.writeAsString(jsonEncode(sessionData.toJson()));

      // Save llama.cpp state
      final stateFile = '$_sessionsDirectory/$sessionId.state';
      final stateFilePtr = stateFile.toNativeUtf8();
      final tokensPtr = malloc<llama_token>(tokens.length);

      try {
        for (int i = 0; i < tokens.length; i++) {
          tokensPtr[i] = tokens[i];
        }

        final success = llamaCpp.llama_state_save_file(context, stateFilePtr.cast<Char>(), tokensPtr, tokens.length);

        if (!success) {
          throw LlamaSessionException('Failed to save llama.cpp state');
        }

        _loadedSessions[sessionId] = sessionData;
        debugPrint('✓ Session saved: $sessionId');
        return true;
      } finally {
        malloc.free(stateFilePtr);
        malloc.free(tokensPtr);
      }
    } catch (e) {
      debugPrint('✗ Failed to save session $sessionId: $e');
      return false;
    }
  }

  /// Load session state from file
  Future<SessionData?> loadSession(String sessionId, Pointer<llama_context> context) async {
    try {
      // Load JSON metadata
      final sessionFile = File('$_sessionsDirectory/$sessionId.json');
      if (!await sessionFile.exists()) {
        debugPrint('Session file not found: $sessionId');
        return null;
      }

      final jsonContent = await sessionFile.readAsString();
      final sessionData = SessionData.fromJson(jsonDecode(jsonContent));

      // Load llama.cpp state
      final stateFile = '$_sessionsDirectory/$sessionId.state';
      final stateFilePtr = stateFile.toNativeUtf8();
      final tokensPtr = malloc<llama_token>(sessionData.tokens.length);
      final tokenCountPtr = malloc<Size>();

      try {
        final success = llamaCpp.llama_state_load_file(
          context,
          stateFilePtr.cast<Char>(),
          tokensPtr,
          sessionData.tokens.length,
          tokenCountPtr,
        );

        if (!success) {
          throw LlamaSessionException('Failed to load llama.cpp state');
        }

        final loadedTokenCount = tokenCountPtr.value;
        debugPrint('✓ Session loaded: $sessionId ($loadedTokenCount tokens)');

        _loadedSessions[sessionId] = sessionData;
        return sessionData;
      } finally {
        malloc.free(stateFilePtr);
        malloc.free(tokensPtr);
        malloc.free(tokenCountPtr);
      }
    } catch (e) {
      debugPrint('✗ Failed to load session $sessionId: $e');
      return null;
    }
  }

  /// List all available sessions
  Future<List<String>> listSessions() async {
    final dir = Directory(_sessionsDirectory);
    if (!await dir.exists()) return [];

    final sessions = <String>[];
    await for (final entity in dir.list()) {
      if (entity is File && entity.path.endsWith('.json')) {
        final name = entity.path.split('/').last.replaceAll('.json', '');
        sessions.add(name);
      }
    }

    sessions.sort();
    return sessions;
  }

  /// Get session metadata without loading full state
  Future<SessionData?> getSessionMetadata(String sessionId) async {
    final sessionFile = File('$_sessionsDirectory/$sessionId.json');
    if (!await sessionFile.exists()) return null;

    try {
      final jsonContent = await sessionFile.readAsString();
      return SessionData.fromJson(jsonDecode(jsonContent));
    } catch (e) {
      debugPrint('Failed to read session metadata: $e');
      return null;
    }
  }

  /// Delete a session
  Future<bool> deleteSession(String sessionId) async {
    try {
      final sessionFile = File('$_sessionsDirectory/$sessionId.json');
      final stateFile = File('$_sessionsDirectory/$sessionId.state');

      if (await sessionFile.exists()) {
        await sessionFile.delete();
      }

      if (await stateFile.exists()) {
        await stateFile.delete();
      }

      _loadedSessions.remove(sessionId);
      debugPrint('✓ Session deleted: $sessionId');
      return true;
    } catch (e) {
      debugPrint('✗ Failed to delete session $sessionId: $e');
      return false;
    }
  }

  /// Clean up old sessions
  Future<int> cleanupOldSessions({Duration? olderThan}) async {
    olderThan ??= const Duration(days: 30);
    final cutoff = DateTime.now().subtract(olderThan);

    final sessions = await listSessions();
    int deletedCount = 0;

    for (final sessionId in sessions) {
      final metadata = await getSessionMetadata(sessionId);
      if (metadata != null && metadata.lastUsedAt.isBefore(cutoff)) {
        if (await deleteSession(sessionId)) {
          deletedCount++;
        }
      }
    }

    debugPrint('✓ Cleaned up $deletedCount old sessions');
    return deletedCount;
  }

  /// Get current tokens from context (helper method)
  Future<List<int>> _getCurrentTokens(Pointer<llama_context> context) async {
    // This is a simplified implementation
    // In practice, you'd need to track tokens during generation
    // or use llama.cpp's internal state
    return []; // Placeholder
  }

  /// Get session statistics
  Future<Map<String, dynamic>> getSessionStatistics() async {
    final sessions = await listSessions();
    final totalSessions = sessions.length;
    int totalSize = 0;
    DateTime? oldestSession;
    DateTime? newestSession;

    for (final sessionId in sessions) {
      final metadata = await getSessionMetadata(sessionId);
      if (metadata != null) {
        // Estimate size (rough)
        totalSize += metadata.tokens.length * 4; // 4 bytes per token estimate

        if (oldestSession == null || metadata.createdAt.isBefore(oldestSession)) {
          oldestSession = metadata.createdAt;
        }

        if (newestSession == null || metadata.createdAt.isAfter(newestSession)) {
          newestSession = metadata.createdAt;
        }
      }
    }

    return {
      'totalSessions': totalSessions,
      'estimatedSizeBytes': totalSize,
      'estimatedSizeMB': (totalSize / (1024 * 1024)).toStringAsFixed(2),
      'oldestSession': oldestSession?.toIso8601String(),
      'newestSession': newestSession?.toIso8601String(),
      'loadedSessions': _loadedSessions.length,
    };
  }
}

/// Context overflow strategies
enum OverflowStrategy {
  /// Throw an error when context is full
  error,

  /// Remove oldest tokens (sliding window)
  truncateHead,

  /// Summarize old context and continue
  summarize,

  /// Clear context and start fresh
  reset,
}

/// Context manager for handling overflow and optimization
class ContextManager {
  final llama_cpp _llamaCpp;
  final Pointer<llama_context> _context;
  final OverflowStrategy _overflowStrategy;

  int _currentPosition = 0;
  final List<int> _tokenHistory = [];

  ContextManager(this._llamaCpp, this._context, {OverflowStrategy overflowStrategy = OverflowStrategy.truncateHead})
    : _overflowStrategy = overflowStrategy;

  /// Current position in context
  int get currentPosition => _currentPosition;

  /// Maximum context size
  int get maxContext => _llamaCpp.llama_n_ctx(_context);

  /// Remaining context space
  int get remainingSpace => maxContext - _currentPosition;

  /// Whether context is near full (>80%)
  bool get isNearFull => _currentPosition > (maxContext * 0.8);

  /// Whether context is full
  bool get isFull => _currentPosition >= maxContext;

  /// Add tokens to context with overflow handling
  Future<bool> addTokens(List<int> tokens) async {
    if (tokens.isEmpty) return true;

    // Check if we need to handle overflow
    if (_currentPosition + tokens.length > maxContext) {
      await _handleOverflow(tokens.length);
    }

    // Add tokens to history
    _tokenHistory.addAll(tokens);
    _currentPosition += tokens.length;

    return true;
  }

  /// Handle context overflow based on strategy
  Future<void> _handleOverflow(int additionalTokens) async {
    final needed = _currentPosition + additionalTokens - maxContext;

    switch (_overflowStrategy) {
      case OverflowStrategy.error:
        throw LlamaContextOverflowException(_currentPosition + additionalTokens, maxContext);

      case OverflowStrategy.truncateHead:
        await _truncateHead(needed);
        break;

      case OverflowStrategy.summarize:
        await _summarizeAndTruncate(needed);
        break;

      case OverflowStrategy.reset:
        await _resetContext();
        break;
    }
  }

  /// Remove tokens from the beginning of context
  Future<void> _truncateHead(int tokensToRemove) async {
    if (tokensToRemove <= 0) return;

    // Clear KV cache for removed tokens
    final memory = _llamaCpp.llama_get_memory(_context);
    _llamaCpp.llama_memory_seq_rm(memory, 0, 0, tokensToRemove);

    // Update tracking
    _tokenHistory.removeRange(0, math.min(tokensToRemove, _tokenHistory.length));
    _currentPosition = math.max(0, _currentPosition - tokensToRemove);

    debugPrint('✓ Truncated $tokensToRemove tokens from context head');
  }

  /// Summarize old context and truncate (placeholder implementation)
  Future<void> _summarizeAndTruncate(int tokensToRemove) async {
    // This would require a separate summarization model or service
    // For now, fall back to simple truncation
    debugPrint('Summarization not implemented, falling back to truncation');
    await _truncateHead(tokensToRemove);
  }

  /// Reset context completely
  Future<void> _resetContext() async {
    final memory = _llamaCpp.llama_get_memory(_context);
    _llamaCpp.llama_memory_clear(memory, true);

    _tokenHistory.clear();
    _currentPosition = 0;

    debugPrint('✓ Context reset completely');
  }

  /// Get context utilization percentage
  double getUtilization() {
    return maxContext > 0 ? _currentPosition / maxContext : 0.0;
  }

  /// Get context statistics
  Map<String, dynamic> getStatistics() {
    return {
      'currentPosition': _currentPosition,
      'maxContext': maxContext,
      'remainingSpace': remainingSpace,
      'utilization': getUtilization(),
      'isNearFull': isNearFull,
      'isFull': isFull,
      'tokenHistoryLength': _tokenHistory.length,
      'overflowStrategy': _overflowStrategy.name,
    };
  }
}

/// Session persistence manager
class SessionPersistence {
  final String _baseDirectory;
  final llama_cpp llamaCpp;

  SessionPersistence(this.llamaCpp, {String? baseDirectory}) : _baseDirectory = baseDirectory ?? 'llama_sessions';

  /// Initialize persistence (create directories)
  Future<void> initialize() async {
    final dirs = [_baseDirectory, '$_baseDirectory/sessions', '$_baseDirectory/states', '$_baseDirectory/backups'];

    for (final dir in dirs) {
      final directory = Directory(dir);
      if (!await directory.exists()) {
        await directory.create(recursive: true);
      }
    }

    debugPrint('✓ Session persistence initialized');
  }

  /// Save session with automatic backup
  Future<bool> saveSessionWithBackup(String sessionId, SessionData sessionData, Pointer<llama_context> context) async {
    try {
      // Create backup of existing session if it exists
      final sessionFile = File('$_baseDirectory/sessions/$sessionId.json');
      if (await sessionFile.exists()) {
        final backupFile = File('$_baseDirectory/backups/${sessionId}_${DateTime.now().millisecondsSinceEpoch}.json');
        await sessionFile.copy(backupFile.path);
      }

      // Save new session
      await sessionFile.writeAsString(jsonEncode(sessionData.toJson()));

      // Save llama.cpp state
      final stateFile = '$_baseDirectory/states/$sessionId.state';
      final stateFilePtr = stateFile.toNativeUtf8();
      final tokensPtr = malloc<llama_token>(sessionData.tokens.length);

      try {
        for (int i = 0; i < sessionData.tokens.length; i++) {
          tokensPtr[i] = sessionData.tokens[i];
        }

        final success = llamaCpp.llama_state_save_file(
          context,
          stateFilePtr.cast<Char>(),
          tokensPtr,
          sessionData.tokens.length,
        );

        if (!success) {
          throw LlamaSessionException('Failed to save llama.cpp state');
        }

        debugPrint('✓ Session saved with backup: $sessionId');
        return true;
      } finally {
        malloc.free(stateFilePtr);
        malloc.free(tokensPtr);
      }
    } catch (e) {
      throw LlamaSessionException('Failed to save session: $e');
    }
  }

  /// Load session with error recovery
  Future<SessionData?> loadSessionWithRecovery(String sessionId, Pointer<llama_context> context) async {
    try {
      // Try to load main session
      final sessionData = await _loadSessionData(sessionId);
      if (sessionData == null) return null;

      // Try to load llama.cpp state
      final stateLoaded = await _loadLlamaState(sessionId, context, sessionData.tokens);
      if (!stateLoaded) {
        debugPrint('Failed to load state, trying backup...');
        // Could implement backup recovery here
      }

      debugPrint('✓ Session loaded: $sessionId');
      return sessionData;
    } catch (e) {
      debugPrint('✗ Failed to load session $sessionId: $e');
      return null;
    }
  }

  /// Load session metadata only
  Future<SessionData?> _loadSessionData(String sessionId) async {
    final sessionFile = File('$_baseDirectory/sessions/$sessionId.json');
    if (!await sessionFile.exists()) return null;

    try {
      final jsonContent = await sessionFile.readAsString();
      return SessionData.fromJson(jsonDecode(jsonContent));
    } catch (e) {
      throw LlamaSessionException('Failed to parse session data: $e');
    }
  }

  /// Load llama.cpp state
  Future<bool> _loadLlamaState(String sessionId, Pointer<llama_context> context, List<int> expectedTokens) async {
    final stateFile = '$_baseDirectory/states/$sessionId.state';
    if (!File(stateFile).existsSync()) return false;

    final stateFilePtr = stateFile.toNativeUtf8();
    final tokensPtr = malloc<llama_token>(expectedTokens.length);
    final tokenCountPtr = malloc<Size>();

    try {
      final success = llamaCpp.llama_state_load_file(
        context,
        stateFilePtr.cast<Char>(),
        tokensPtr,
        expectedTokens.length,
        tokenCountPtr,
      );

      return success;
    } finally {
      malloc.free(stateFilePtr);
      malloc.free(tokensPtr);
      malloc.free(tokenCountPtr);
    }
  }

  /// Get all session metadata
  Future<List<SessionData>> getAllSessionMetadata() async {
    final sessions = <SessionData>[];
    final dir = Directory('$_baseDirectory/sessions');

    if (!await dir.exists()) return sessions;

    await for (final entity in dir.list()) {
      if (entity is File && entity.path.endsWith('.json')) {
        try {
          final jsonContent = await entity.readAsString();
          final sessionData = SessionData.fromJson(jsonDecode(jsonContent));
          sessions.add(sessionData);
        } catch (e) {
          debugPrint('Failed to load session metadata from ${entity.path}: $e');
        }
      }
    }

    // Sort by last used (most recent first)
    sessions.sort((a, b) => b.lastUsedAt.compareTo(a.lastUsedAt));
    return sessions;
  }

  /// Clean up old sessions and backups
  Future<int> cleanup({Duration? olderThan}) async {
    olderThan ??= const Duration(days: 30);
    final cutoff = DateTime.now().subtract(olderThan);
    int deletedCount = 0;

    // Clean up old sessions
    final sessions = await getAllSessionMetadata();
    for (final session in sessions) {
      if (session.lastUsedAt.isBefore(cutoff)) {
        await _deleteSessionFiles(session.sessionId);
        deletedCount++;
      }
    }

    // Clean up old backups
    final backupDir = Directory('$_baseDirectory/backups');
    if (await backupDir.exists()) {
      await for (final entity in backupDir.list()) {
        if (entity is File) {
          final stat = await entity.stat();
          if (stat.modified.isBefore(cutoff)) {
            await entity.delete();
            deletedCount++;
          }
        }
      }
    }

    debugPrint('✓ Cleaned up $deletedCount old files');
    return deletedCount;
  }

  /// Delete all files for a session
  Future<void> _deleteSessionFiles(String sessionId) async {
    final files = [File('$_baseDirectory/sessions/$sessionId.json'), File('$_baseDirectory/states/$sessionId.state')];

    for (final file in files) {
      if (await file.exists()) {
        await file.delete();
      }
    }
  }

  /// Get storage statistics
  Future<Map<String, dynamic>> getStorageStatistics() async {
    final sessions = await getAllSessionMetadata();
    int totalSize = 0;
    int totalTokens = 0;

    for (final session in sessions) {
      totalTokens += session.tokens.length;

      // Estimate file sizes
      final sessionFile = File('$_baseDirectory/sessions/${session.sessionId}.json');
      final stateFile = File('$_baseDirectory/states/${session.sessionId}.state');

      if (await sessionFile.exists()) {
        totalSize += await sessionFile.length();
      }

      if (await stateFile.exists()) {
        totalSize += await stateFile.length();
      }
    }

    return {
      'totalSessions': sessions.length,
      'totalSizeBytes': totalSize,
      'totalSizeMB': (totalSize / (1024 * 1024)).toStringAsFixed(2),
      'totalTokens': totalTokens,
      'averageTokensPerSession': sessions.isNotEmpty ? (totalTokens / sessions.length).round() : 0,
    };
  }
}
