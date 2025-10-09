import 'package:flutter/material.dart';
import 'dart:io';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Llama Chat',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple), useMaterial3: true),
      home: const ChatPage(),
    );
  }
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final TextEditingController _controller = TextEditingController();
  final List<ChatMessage> _messages = [];
  final ScrollController _scrollController = ScrollController();

  LlamaParent? _llamaParent;
  bool _isInitialized = false;
  bool _isLoading = false;
  String _currentResponse = '';

  Future<void> _initializeLlama() async {
    try {
      Llama.libraryPath = "bin/macos-arm64/libllama.dylib";

      print("Llama.libraryPath: ${Llama.libraryPath}");

      ContextParams contextParams = ContextParams();
      contextParams.nPredict = -1;
      contextParams.nCtx = 8192;

      final samplerParams = SamplerParams();
      samplerParams.temp = 0.7;
      samplerParams.topK = 64;
      samplerParams.topP = 0.95;
      samplerParams.penaltyRepeat = 1.1;

      String modelPath = "/Users/skywar56/Documents/Flutter/quiz_wrapper/assets/gemma-3-4b-it-Q4_K_M.gguf";

      final loadCommand = LlamaLoad(
        path: modelPath,
        modelParams: ModelParams(),
        contextParams: contextParams,
        samplingParams: samplerParams,
      );

      _llamaParent = LlamaParent(loadCommand);

      print("Initializing llama parent");
      await _llamaParent!.init().timeout(const Duration(seconds: 70));

      _llamaParent!.stream.listen((response) {
        setState(() {
          _currentResponse += response;
        });
        _scrollToBottom();
      });

      _llamaParent!.completions.listen((event) {
        if (event.success) {
          setState(() {
            if (_currentResponse.isNotEmpty) {
              _messages.add(ChatMessage(text: _currentResponse, isUser: false));
              _currentResponse = '';
            }
          });
          _scrollToBottom();
        } else {
          print("Completion error: ${event.errorDetails}");
          _showError("Completion error: ${event.errorDetails}");
        }
      });

      print("Initialization success");
      setState(() {
        _isInitialized = true;
        _isLoading = false;
      });
    } catch (e) {
      print("Initialization error: $e");
      _showError("Initialization error: $e");
      setState(() => _isLoading = false);
    }
  }

  void _sendMessage() {
    if (_controller.text.trim().isEmpty || !_isInitialized) return;

    final userMessage = _controller.text.trim();
    setState(() {
      _messages.add(ChatMessage(text: userMessage, isUser: true));
      _controller.clear();
    });

    _scrollToBottom();

    final prompt = _getPrompt(userMessage);
    _llamaParent?.sendPrompt(prompt);
  }

  String _getPrompt(String content) {
    ChatHistory history = ChatHistory()
      ..addMessage(role: Role.user, content: content)
      ..addMessage(role: Role.assistant, content: "");
    return history.exportFormat(ChatFormat.gemini, leaveLastAssistantOpen: true);
  }

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

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message), backgroundColor: Colors.red));
  }

  @override
  void dispose() {
    _llamaParent?.dispose();
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Llama Chat'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: () async {
              await _initializeLlama();
            },
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Column(
        children: [
          if (_isLoading)
            const LinearProgressIndicator()
          else if (!_isInitialized)
            const Padding(padding: EdgeInsets.all(16.0), child: Text('Failed to initialize model')),
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length + (_currentResponse.isEmpty ? 0 : 1),
              itemBuilder: (context, index) {
                if (index < _messages.length) {
                  return MessageBubble(message: _messages[index]);
                } else {
                  return MessageBubble(message: ChatMessage(text: _currentResponse, isUser: false));
                }
              },
            ),
          ),
          Container(
            padding: const EdgeInsets.all(8.0),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [BoxShadow(color: Colors.grey.withOpacity(0.3), spreadRadius: 1, blurRadius: 3)],
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    enabled: _isInitialized,
                    decoration: const InputDecoration(
                      hintText: 'Type a message...',
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: _isInitialized ? _sendMessage : null,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class ChatMessage {
  final String text;
  final bool isUser;

  ChatMessage({required this.text, required this.isUser});
}

class MessageBubble extends StatelessWidget {
  final ChatMessage message;

  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: message.isUser ? Theme.of(context).colorScheme.primary : Colors.grey[300],
          borderRadius: BorderRadius.circular(16),
        ),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
        child: Text(message.text, style: TextStyle(color: message.isUser ? Colors.white : Colors.black87)),
      ),
    );
  }
}
