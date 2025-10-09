/// Chat utilities and message handling for LlamaService
///
/// Provides conversation management, message formatting, chat templates,
/// and history management for chat-based interactions.

import 'dart:convert';

/// Represents a single message in a conversation
class ChatMessage {
  final String role;
  final String content;
  final DateTime timestamp;
  final Map<String, dynamic>? metadata;

  ChatMessage({required this.role, required this.content, DateTime? timestamp, this.metadata})
    : timestamp = timestamp ?? DateTime.now();

  /// Create a system message
  factory ChatMessage.system(String content, {Map<String, dynamic>? metadata}) {
    return ChatMessage(role: 'system', content: content, metadata: metadata);
  }

  /// Create a user message
  factory ChatMessage.user(String content, {Map<String, dynamic>? metadata}) {
    return ChatMessage(role: 'user', content: content, metadata: metadata);
  }

  /// Create an assistant message
  factory ChatMessage.assistant(String content, {Map<String, dynamic>? metadata}) {
    return ChatMessage(role: 'assistant', content: content, metadata: metadata);
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'role': role,
      'content': content,
      'timestamp': timestamp.toIso8601String(),
      if (metadata != null) 'metadata': metadata,
    };
  }

  /// Create from JSON
  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(
      role: json['role'] as String,
      content: json['content'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
      metadata: json['metadata'] as Map<String, dynamic>?,
    );
  }

  @override
  String toString() => '$role: $content';
}

/// Manages conversation history with automatic trimming
class ChatHistory {
  final List<ChatMessage> _messages = [];
  final int _maxMessages;
  final int _maxTokensEstimate;
  final bool _keepSystemMessages;

  ChatHistory({int maxMessages = 100, int maxTokensEstimate = 4000, bool keepSystemMessages = true})
    : _maxMessages = maxMessages,
      _maxTokensEstimate = maxTokensEstimate,
      _keepSystemMessages = keepSystemMessages;

  /// All messages in chronological order
  List<ChatMessage> get messages => List.unmodifiable(_messages);

  /// System messages only
  List<ChatMessage> get systemMessages => _messages.where((m) => m.role == 'system').toList();

  /// User messages only
  List<ChatMessage> get userMessages => _messages.where((m) => m.role == 'user').toList();

  /// Assistant messages only
  List<ChatMessage> get assistantMessages => _messages.where((m) => m.role == 'assistant').toList();

  /// Most recent conversation pairs (user + assistant)
  List<ChatMessage> get recentPairs => _getRecentPairs(5);

  /// Add a message to the history
  void addMessage(ChatMessage message) {
    _messages.add(message);
    _autoTrim();
  }

  /// Add multiple messages
  void addMessages(List<ChatMessage> messages) {
    _messages.addAll(messages);
    _autoTrim();
  }

  /// Add a system message
  void addSystem(String content, {Map<String, dynamic>? metadata}) {
    addMessage(ChatMessage.system(content, metadata: metadata));
  }

  /// Add a user message
  void addUser(String content, {Map<String, dynamic>? metadata}) {
    addMessage(ChatMessage.user(content, metadata: metadata));
  }

  /// Add an assistant message
  void addAssistant(String content, {Map<String, dynamic>? metadata}) {
    addMessage(ChatMessage.assistant(content, metadata: metadata));
  }

  /// Get messages for current context (trimmed if necessary)
  List<ChatMessage> getCurrentContext({int? maxTokens}) {
    maxTokens ??= _maxTokensEstimate;

    final context = <ChatMessage>[];
    int estimatedTokens = 0;

    // Always include system messages if keeping them
    if (_keepSystemMessages) {
      for (final msg in systemMessages) {
        context.add(msg);
        estimatedTokens += _estimateTokens(msg.content);
      }
    }

    // Add recent messages in reverse order until we hit token limit
    final nonSystemMessages = _messages.where((m) => m.role != 'system').toList().reversed;

    for (final msg in nonSystemMessages) {
      final msgTokens = _estimateTokens(msg.content);
      if (estimatedTokens + msgTokens > maxTokens) break;

      context.insert(_keepSystemMessages ? systemMessages.length : 0, msg);
      estimatedTokens += msgTokens;
    }

    return context;
  }

  /// Get recent conversation pairs (user + assistant)
  List<ChatMessage> _getRecentPairs(int pairCount) {
    final pairs = <ChatMessage>[];
    final nonSystem = _messages.where((m) => m.role != 'system').toList();

    // Work backwards to get recent pairs
    for (int i = nonSystem.length - 1; i >= 0 && pairs.length < pairCount * 2; i--) {
      pairs.insert(0, nonSystem[i]);
    }

    return pairs;
  }

  /// Estimate token count for text (rough approximation)
  int _estimateTokens(String text) {
    // Rough estimate: ~4 characters per token
    return (text.length / 4).ceil();
  }

  /// Auto-trim history based on limits
  void _autoTrim() {
    // Trim by message count
    while (_messages.length > _maxMessages) {
      // Remove oldest non-system message
      final index = _messages.indexWhere((m) => m.role != 'system');
      if (index != -1) {
        _messages.removeAt(index);
      } else {
        break; // Only system messages left
      }
    }

    // Trim by estimated token count
    while (_estimateTokens(_messages.map((m) => m.content).join(' ')) > _maxTokensEstimate) {
      final index = _messages.indexWhere((m) => m.role != 'system');
      if (index != -1) {
        _messages.removeAt(index);
      } else {
        break;
      }
    }
  }

  /// Clear all messages
  void clear() {
    _messages.clear();
  }

  /// Clear non-system messages
  void clearConversation() {
    _messages.removeWhere((m) => m.role != 'system');
  }

  /// Export to JSON
  Map<String, dynamic> toJson() {
    return {
      'messages': _messages.map((m) => m.toJson()).toList(),
      'maxMessages': _maxMessages,
      'maxTokensEstimate': _maxTokensEstimate,
      'keepSystemMessages': _keepSystemMessages,
    };
  }

  /// Import from JSON
  factory ChatHistory.fromJson(Map<String, dynamic> json) {
    final history = ChatHistory(
      maxMessages: json['maxMessages'] as int? ?? 100,
      maxTokensEstimate: json['maxTokensEstimate'] as int? ?? 4000,
      keepSystemMessages: json['keepSystemMessages'] as bool? ?? true,
    );

    final messages = (json['messages'] as List?)?.map((m) => ChatMessage.fromJson(m)).toList() ?? [];

    history.addMessages(messages);
    return history;
  }

  /// Get conversation summary
  String getSummary() {
    final total = _messages.length;
    final system = systemMessages.length;
    final user = userMessages.length;
    final assistant = assistantMessages.length;

    return 'Messages: $total (system: $system, user: $user, assistant: $assistant)';
  }
}

/// Chat template formatter for different model formats
class ChatTemplateFormatter {
  /// Format messages using ChatML template
  static String formatChatML(List<ChatMessage> messages) {
    final buffer = StringBuffer();

    for (final message in messages) {
      buffer.writeln('<|im_start|>${message.role}');
      buffer.writeln(message.content);
      buffer.writeln('<|im_end|>');
    }

    // Add assistant start for completion
    buffer.write('<|im_start|>assistant\n');

    return buffer.toString();
  }

  /// Format messages using Alpaca template
  static String formatAlpaca(List<ChatMessage> messages) {
    final buffer = StringBuffer();
    String? systemPrompt;

    // Extract system message
    for (final message in messages) {
      if (message.role == 'system') {
        systemPrompt = message.content;
        break;
      }
    }

    // Add instruction header
    if (systemPrompt != null) {
      buffer.writeln(
        'Below is an instruction that describes a task. Write a response that appropriately completes the request.',
      );
      buffer.writeln();
      buffer.writeln('### Instruction:');
      buffer.writeln(systemPrompt);
      buffer.writeln();
    }

    // Add conversation
    for (final message in messages) {
      if (message.role == 'system') continue;

      if (message.role == 'user') {
        buffer.writeln('### Input:');
        buffer.writeln(message.content);
        buffer.writeln();
      } else if (message.role == 'assistant') {
        buffer.writeln('### Response:');
        buffer.writeln(message.content);
        buffer.writeln();
      }
    }

    // Add response prompt
    buffer.write('### Response:');

    return buffer.toString();
  }

  /// Format messages using Llama template
  static String formatLlama(List<ChatMessage> messages) {
    final buffer = StringBuffer();
    String? systemPrompt;

    // Extract system message
    for (final message in messages) {
      if (message.role == 'system') {
        systemPrompt = message.content;
        break;
      }
    }

    buffer.write('<s>');

    // Process conversation
    for (int i = 0; i < messages.length; i++) {
      final message = messages[i];
      if (message.role == 'system') continue;

      if (message.role == 'user') {
        buffer.write('[INST] ');
        if (systemPrompt != null && i == 0) {
          buffer.write('<<SYS>>\n$systemPrompt\n<</SYS>>\n\n');
        }
        buffer.write('${message.content} [/INST]');
      } else if (message.role == 'assistant') {
        buffer.write(' ${message.content} </s><s>');
      }
    }

    return buffer.toString();
  }

  /// Format messages using Gemini template
  static String formatGemini(List<ChatMessage> messages) {
    final buffer = StringBuffer();

    for (final message in messages) {
      switch (message.role) {
        case 'system':
          buffer.writeln('*System: ${message.content}*\n');
          break;
        case 'user':
          buffer.writeln('**User:** ${message.content}\n');
          break;
        case 'assistant':
          buffer.writeln('**Assistant:** ${message.content}\n');
          break;
      }
    }

    buffer.write('**Assistant:**');
    return buffer.toString();
  }

  /// Auto-detect and apply appropriate template
  static String formatAuto(List<ChatMessage> messages, {String? modelName}) {
    if (modelName != null) {
      final name = modelName.toLowerCase();

      if (name.contains('llama')) {
        return formatLlama(messages);
      } else if (name.contains('alpaca')) {
        return formatAlpaca(messages);
      } else if (name.contains('gemini')) {
        return formatGemini(messages);
      }
    }

    // Default to ChatML
    return formatChatML(messages);
  }
}

/// Supported chat template formats
enum ChatTemplate {
  chatML('ChatML', '<|im_start|>role\ncontent<|im_end|>'),
  alpaca('Alpaca', '### Instruction:\n### Response:'),
  llama('Llama', '[INST] content [/INST]'),
  gemini('Gemini', '**Role:** content'),
  custom('Custom', 'User-defined template');

  const ChatTemplate(this.name, this.example);

  final String name;
  final String example;
}

/// Configuration for chat behavior
class ChatConfig {
  final ChatTemplate template;
  final String? customTemplate;
  final bool includeSystemInHistory;
  final int maxHistoryMessages;
  final int maxHistoryTokens;
  final bool autoTrim;
  final String? defaultSystemPrompt;

  const ChatConfig({
    this.template = ChatTemplate.chatML,
    this.customTemplate,
    this.includeSystemInHistory = true,
    this.maxHistoryMessages = 50,
    this.maxHistoryTokens = 4000,
    this.autoTrim = true,
    this.defaultSystemPrompt,
  });

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'template': template.name,
      'customTemplate': customTemplate,
      'includeSystemInHistory': includeSystemInHistory,
      'maxHistoryMessages': maxHistoryMessages,
      'maxHistoryTokens': maxHistoryTokens,
      'autoTrim': autoTrim,
      'defaultSystemPrompt': defaultSystemPrompt,
    };
  }

  /// Create from JSON
  factory ChatConfig.fromJson(Map<String, dynamic> json) {
    final templateName = json['template'] as String? ?? 'ChatML';
    final template = ChatTemplate.values.firstWhere((t) => t.name == templateName, orElse: () => ChatTemplate.chatML);

    return ChatConfig(
      template: template,
      customTemplate: json['customTemplate'] as String?,
      includeSystemInHistory: json['includeSystemInHistory'] as bool? ?? true,
      maxHistoryMessages: json['maxHistoryMessages'] as int? ?? 50,
      maxHistoryTokens: json['maxHistoryTokens'] as int? ?? 4000,
      autoTrim: json['autoTrim'] as bool? ?? true,
      defaultSystemPrompt: json['defaultSystemPrompt'] as String?,
    );
  }
}

/// Main chat service for managing conversations
class LlamaChatService {
  final ChatHistory _history;
  final ChatConfig _config;

  LlamaChatService({ChatConfig? config, ChatHistory? history})
    : _config = config ?? const ChatConfig(),
      _history =
          history ??
          ChatHistory(
            maxMessages: config?.maxHistoryMessages ?? 50,
            maxTokensEstimate: config?.maxHistoryTokens ?? 4000,
            keepSystemMessages: config?.includeSystemInHistory ?? true,
          ) {
    // Add default system prompt if provided
    if (_config.defaultSystemPrompt != null) {
      _history.addSystem(_config.defaultSystemPrompt!);
    }
  }

  /// Current chat history
  ChatHistory get history => _history;

  /// Chat configuration
  ChatConfig get config => _config;

  /// Add a user message and get formatted prompt for generation
  String addUserMessage(String content, {Map<String, dynamic>? metadata}) {
    _history.addUser(content, metadata: metadata);
    return formatCurrentConversation();
  }

  /// Add assistant response to history
  void addAssistantResponse(String content, {Map<String, dynamic>? metadata}) {
    _history.addAssistant(content, metadata: metadata);
  }

  /// Format current conversation for the model
  String formatCurrentConversation() {
    final messages = _history.getCurrentContext(maxTokens: _config.maxHistoryTokens);

    switch (_config.template) {
      case ChatTemplate.chatML:
        return ChatTemplateFormatter.formatChatML(messages);
      case ChatTemplate.alpaca:
        return ChatTemplateFormatter.formatAlpaca(messages);
      case ChatTemplate.llama:
        return ChatTemplateFormatter.formatLlama(messages);
      case ChatTemplate.gemini:
        return ChatTemplateFormatter.formatGemini(messages);
      case ChatTemplate.custom:
        if (_config.customTemplate != null) {
          return _formatCustomTemplate(messages, _config.customTemplate!);
        }
        return ChatTemplateFormatter.formatChatML(messages);
    }
  }

  /// Format using custom template
  String _formatCustomTemplate(List<ChatMessage> messages, String template) {
    // Simple template substitution
    // Template variables: {role}, {content}, {messages}

    String result = template;

    // Replace {messages} with formatted message list
    final messageList = messages.map((m) => '${m.role}: ${m.content}').join('\n');
    result = result.replaceAll('{messages}', messageList);

    // For single message templates, use the last message
    if (messages.isNotEmpty) {
      final lastMessage = messages.last;
      result = result.replaceAll('{role}', lastMessage.role);
      result = result.replaceAll('{content}', lastMessage.content);
    }

    return result;
  }

  /// Start a new conversation (keep system messages)
  void newConversation() {
    if (_config.includeSystemInHistory) {
      final systemMessages = _history.systemMessages;
      _history.clear();
      _history.addMessages(systemMessages);
    } else {
      _history.clear();
    }
  }

  /// Export conversation to JSON
  String exportToJson() {
    return jsonEncode({
      'config': _config.toJson(),
      'history': _history.toJson(),
      'exportedAt': DateTime.now().toIso8601String(),
    });
  }

  /// Import conversation from JSON
  static LlamaChatService importFromJson(String jsonString) {
    final data = jsonDecode(jsonString) as Map<String, dynamic>;

    final config = ChatConfig.fromJson(data['config'] ?? {});
    final history = ChatHistory.fromJson(data['history'] ?? {});

    return LlamaChatService(config: config, history: history);
  }

  /// Get conversation statistics
  Map<String, dynamic> getStatistics() {
    final messages = _history.messages;
    final totalChars = messages.map((m) => m.content.length).fold(0, (a, b) => a + b);
    final estimatedTokens = (totalChars / 4).ceil();

    return {
      'totalMessages': messages.length,
      'systemMessages': _history.systemMessages.length,
      'userMessages': _history.userMessages.length,
      'assistantMessages': _history.assistantMessages.length,
      'totalCharacters': totalChars,
      'estimatedTokens': estimatedTokens,
      'conversationStarted': messages.isNotEmpty ? messages.first.timestamp.toIso8601String() : null,
      'lastMessage': messages.isNotEmpty ? messages.last.timestamp.toIso8601String() : null,
    };
  }

  /// Search messages by content
  List<ChatMessage> searchMessages(String query, {bool caseSensitive = false}) {
    final searchQuery = caseSensitive ? query : query.toLowerCase();

    return _history.messages.where((message) {
      final content = caseSensitive ? message.content : message.content.toLowerCase();
      return content.contains(searchQuery);
    }).toList();
  }

  /// Get messages by role
  List<ChatMessage> getMessagesByRole(String role) {
    return _history.messages.where((m) => m.role == role).toList();
  }

  /// Get messages in date range
  List<ChatMessage> getMessagesInRange(DateTime start, DateTime end) {
    return _history.messages.where((m) => m.timestamp.isAfter(start) && m.timestamp.isBefore(end)).toList();
  }
}

/// Utility functions for chat operations
class ChatUtils {
  /// Extract code blocks from assistant messages
  static List<String> extractCodeBlocks(String content) {
    final codeBlocks = <String>[];
    final regex = RegExp(r'```(?:\w+)?\n(.*?)\n```', multiLine: true, dotAll: true);

    for (final match in regex.allMatches(content)) {
      final code = match.group(1);
      if (code != null) {
        codeBlocks.add(code);
      }
    }

    return codeBlocks;
  }

  /// Clean up assistant responses (remove artifacts, fix formatting)
  static String cleanResponse(String response) {
    String cleaned = response;

    // Remove common artifacts
    cleaned = cleaned.replaceAll(RegExp(r'^(Assistant:|AI:|Bot:)\s*', multiLine: true), '');
    cleaned = cleaned.replaceAll(RegExp(r'<\|.*?\|>'), ''); // Remove special tokens

    // Fix spacing
    cleaned = cleaned.replaceAll(RegExp(r'\n{3,}'), '\n\n'); // Max 2 newlines
    cleaned = cleaned.replaceAll(RegExp(r' {2,}'), ' '); // Max 1 space

    return cleaned.trim();
  }

  /// Detect if response seems incomplete
  static bool isIncompleteResponse(String response) {
    // Check for common incomplete patterns
    final incomplete = [
      RegExp(r'\.\.\.$'), // Ends with ellipsis
      RegExp(r'[^.!?]$'), // Doesn't end with punctuation
      RegExp(r'\b(and|but|or|so|then|because|since|while|if|when|where|how|what|why)$', caseSensitive: false),
    ];

    return incomplete.any((pattern) => pattern.hasMatch(response.trim()));
  }

  /// Estimate reading time for text
  static Duration estimateReadingTime(String text) {
    // Average reading speed: 200 words per minute
    final wordCount = text.split(RegExp(r'\s+')).length;
    final minutes = (wordCount / 200.0).ceil();
    return Duration(minutes: minutes);
  }

  /// Count tokens in conversation (rough estimate)
  static int estimateConversationTokens(List<ChatMessage> messages) {
    final totalChars = messages.map((m) => m.content.length).fold(0, (a, b) => a + b);
    return (totalChars / 4).ceil(); // Rough estimate
  }
}
