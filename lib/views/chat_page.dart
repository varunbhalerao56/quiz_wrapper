/// Auto-loading chat page with conversation interface
///
/// Automatically loads the model and provides a chat interface
/// with message history and real-time streaming responses.

import 'package:flutter/material.dart';
import '../src/llama_service_enhanced.dart';
import '../src/utils/llama_config.dart';
import '../src/chat/llama_chat.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  LlamaServiceEnhanced? _llama;
  LlamaChatService? _chatService;
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  bool _isLoading = true;
  bool _isGenerating = false;
  String _loadingStatus = 'Initializing...';
  String _currentResponse = '';

  final List<ChatMessage> _messages = [];

  // Model path - you might want to make this configurable
  final String _modelPath =
      '/Users/skywar56/Documents/Flutter/quiz_wrapper/assets/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';

  @override
  void initState() {
    super.initState();
    _autoLoadEverything();
  }

  @override
  void dispose() {
    _llama?.dispose();
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  /// Auto-load everything needed for chat
  Future<void> _autoLoadEverything() async {
    try {
      setState(() {
        _loadingStatus = 'Creating service...';
      });

      _llama = LlamaServiceEnhanced();
      _chatService = LlamaChatService(
        config: const ChatConfig(
          template: ChatTemplate.chatML,
          maxHistoryMessages: 20,
          maxHistoryTokens: 2000,
          defaultSystemPrompt: 'You are a helpful AI assistant. Be concise and friendly.',
        ),
      );

      setState(() {
        _loadingStatus = 'Initializing backend...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      _llama!.init();

      setState(() {
        _loadingStatus = 'Loading model...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      final modelLoaded = _llama!.loadModel(
        _modelPath,
        config: const ModelConfig(
          useMmap: true,
          nGpuLayers: 0, // CPU for now
        ),
      );

      if (!modelLoaded) {
        throw Exception('Failed to load model');
      }

      setState(() {
        _loadingStatus = 'Creating context...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      final contextCreated = _llama!.createContext(config: const ContextConfig(nCtx: 2048, nThreads: 4));

      if (!contextCreated) {
        throw Exception('Failed to create context');
      }

      setState(() {
        _isLoading = false;
        _loadingStatus = 'Ready for chat!';
      });

      // Add welcome message
      _addSystemMessage('Chat is ready! Ask me anything.');
    } catch (e) {
      setState(() {
        _isLoading = false;
        _loadingStatus = 'Error: $e';
      });
    }
  }

  /// Add a system message to the chat
  void _addSystemMessage(String content) {
    setState(() {
      _messages.add(ChatMessage.system(content));
    });
    _scrollToBottom();
  }

  /// Add a user message to the chat
  void _addUserMessage(String content) {
    setState(() {
      _messages.add(ChatMessage.user(content));
    });
    _chatService!.addUserMessage(content);
    _scrollToBottom();
  }

  // Helper method removed - functionality integrated into streaming response

  /// Send a message and get AI response
  Future<void> _sendMessage() async {
    if (_messageController.text.trim().isEmpty || _isGenerating) return;

    final userMessage = _messageController.text.trim();
    _messageController.clear();

    // Add user message
    _addUserMessage(userMessage);

    setState(() {
      _isGenerating = true;
      _currentResponse = '';
    });

    try {
      // Format the conversation for the model
      final formattedPrompt = _chatService!.formatCurrentConversation();

      // Stream the response
      await for (final token in _llama!.generateStream(
        formattedPrompt,
        config: SamplerConfig.balanced.copyWith(maxTokens: 150, stopStrings: ['<|im_end|>', 'User:', 'Human:']),
      )) {
        setState(() {
          _currentResponse += token;
          // Update the last message if it's an assistant message being built
          if (_messages.isNotEmpty && _messages.last.role == 'assistant') {
            _messages[_messages.length - 1] = ChatMessage.assistant(_currentResponse);
          } else {
            _messages.add(ChatMessage.assistant(_currentResponse));
          }
        });
        _scrollToBottom();
      }

      // Finalize the response
      if (_currentResponse.isNotEmpty) {
        _chatService!.addAssistantResponse(_currentResponse);
      }
    } catch (e) {
      _addSystemMessage('Error generating response: $e');
    } finally {
      setState(() {
        _isGenerating = false;
        _currentResponse = '';
      });
    }
  }

  /// Scroll to bottom of chat
  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  /// Clear chat history
  void _clearChat() {
    setState(() {
      _messages.clear();
      _currentResponse = '';
    });
    _chatService?.history.clear();
    _addSystemMessage('Chat cleared. Start a new conversation!');
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(title: const Text('AI Chat'), backgroundColor: Theme.of(context).colorScheme.inversePrimary),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 24),
              Text(_loadingStatus, style: const TextStyle(fontSize: 18), textAlign: TextAlign.center),
              const SizedBox(height: 16),
              const Text('This may take a moment...', style: TextStyle(color: Colors.grey)),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Chat'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [IconButton(icon: const Icon(Icons.clear_all), onPressed: _clearChat, tooltip: 'Clear Chat')],
      ),
      body: Column(
        children: [
          // Chat messages
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                return _buildMessageBubble(message);
              },
            ),
          ),

          // Input area
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              border: Border(top: BorderSide(color: Theme.of(context).dividerColor)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _messageController,
                    enabled: !_isGenerating,
                    decoration: InputDecoration(
                      hintText: _isGenerating ? 'AI is responding...' : 'Type your message...',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(24)),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                    maxLines: null,
                  ),
                ),
                const SizedBox(width: 12),
                FloatingActionButton(
                  onPressed: _isGenerating ? null : _sendMessage,
                  child: _isGenerating
                      ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Build a message bubble
  Widget _buildMessageBubble(ChatMessage message) {
    final isUser = message.role == 'user';
    final isSystem = message.role == 'system';

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              backgroundColor: isSystem ? Colors.grey : Colors.blue,
              child: Icon(isSystem ? Icons.info : Icons.smart_toy, color: Colors.white, size: 20),
            ),
            const SizedBox(width: 12),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: isUser
                    ? Theme.of(context).colorScheme.primary
                    : isSystem
                    ? Colors.grey[100]
                    : Theme.of(context).colorScheme.secondary,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                message.content,
                style: TextStyle(
                  color: isUser
                      ? Theme.of(context).colorScheme.onPrimary
                      : isSystem
                      ? Colors.grey[700]
                      : Theme.of(context).colorScheme.onSecondary,
                  fontSize: 16,
                ),
              ),
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 12),
            CircleAvatar(
              backgroundColor: Theme.of(context).colorScheme.primary,
              child: const Icon(Icons.person, color: Colors.white, size: 20),
            ),
          ],
        ],
      ),
    );
  }
}
