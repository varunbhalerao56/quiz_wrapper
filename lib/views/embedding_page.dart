/// Embedding demonstration page
///
/// Shows text embeddings computation, similarity search,
/// and basic RAG (Retrieval Augmented Generation) functionality.

import 'package:flutter/material.dart';
import 'dart:math' as math;
import '../src/llama_service_enhanced.dart';
import '../src/utils/llama_config.dart';

class EmbeddingPage extends StatefulWidget {
  const EmbeddingPage({super.key});

  @override
  State<EmbeddingPage> createState() => _EmbeddingPageState();
}

class _EmbeddingPageState extends State<EmbeddingPage> {
  LlamaServiceEnhanced? _llama;
  final TextEditingController _queryController = TextEditingController();
  final TextEditingController _documentController = TextEditingController();

  bool _isLoading = true;
  bool _isComputing = false;
  String _loadingStatus = 'Initializing...';

  // Sample documents for RAG demo
  final List<String> _documents = [
    'The capital of France is Paris. It is known for the Eiffel Tower and excellent cuisine.',
    'Python is a popular programming language used for AI, web development, and data science.',
    'Machine learning is a subset of artificial intelligence that enables computers to learn from data.',
    'The weather today is sunny with a temperature of 75 degrees Fahrenheit.',
    'Flutter is Google\'s UI toolkit for building natively compiled applications.',
    'Quantum computing uses quantum mechanical phenomena to process information.',
  ];

  // Store computed embeddings (placeholder - in real app you'd use proper vector DB)
  final Map<String, List<double>> _documentEmbeddings = {};
  // Removed unused _searchResults field

  String _queryEmbeddingInfo = '';
  String _similarityResults = '';

  @override
  void initState() {
    super.initState();
    _autoLoadModel();
  }

  @override
  void dispose() {
    _llama?.dispose();
    _queryController.dispose();
    _documentController.dispose();
    super.dispose();
  }

  /// Auto-load the model for embeddings
  Future<void> _autoLoadModel() async {
    try {
      setState(() {
        _loadingStatus = 'Creating service...';
      });

      _llama = LlamaServiceEnhanced();

      setState(() {
        _loadingStatus = 'Initializing backend...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      _llama!.init();

      setState(() {
        _loadingStatus = 'Loading model for embeddings...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      final modelLoaded = _llama!.loadModel(
        '/Users/skywar56/Documents/Flutter/quiz_wrapper/assets/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        config: const ModelConfig(useMmap: true),
      );

      if (!modelLoaded) {
        throw Exception('Failed to load model');
      }

      setState(() {
        _loadingStatus = 'Creating context for embeddings...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      final contextCreated = _llama!.createContext(
        config: const ContextConfig(
          nCtx: 2048,
          nThreads: 4,
          embeddings: true, // Enable embeddings mode
        ),
      );

      if (!contextCreated) {
        throw Exception('Failed to create context');
      }

      setState(() {
        _loadingStatus = 'Pre-computing document embeddings...';
      });
      await Future.delayed(const Duration(milliseconds: 100));

      // Pre-compute embeddings for sample documents
      await _precomputeDocumentEmbeddings();

      setState(() {
        _isLoading = false;
        _loadingStatus = 'Ready for embeddings!';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _loadingStatus = 'Error: $e';
      });
    }
  }

  /// Pre-compute embeddings for sample documents
  Future<void> _precomputeDocumentEmbeddings() async {
    for (int i = 0; i < _documents.length; i++) {
      setState(() {
        _loadingStatus = 'Computing embeddings ${i + 1}/${_documents.length}...';
      });

      // Simulate embedding computation (replace with actual computation when available)
      await Future.delayed(const Duration(milliseconds: 200));

      // Generate mock embeddings for demo
      final embedding = _generateMockEmbedding(_documents[i]);
      _documentEmbeddings[_documents[i]] = embedding;
    }
  }

  /// Generate mock embeddings for demo (replace with real embeddings later)
  List<double> _generateMockEmbedding(String text) {
    final random = math.Random(text.hashCode);
    return List.generate(384, (i) => (random.nextDouble() - 0.5) * 2.0);
  }

  /// Compute similarity between two embeddings
  double _computeSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) return 0.0;

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA == 0.0 || normB == 0.0) return 0.0;

    return dotProduct / (math.sqrt(normA) * math.sqrt(normB));
  }

  /// Search for similar documents
  Future<void> _searchSimilarDocuments() async {
    if (_queryController.text.trim().isEmpty) return;

    setState(() {
      _isComputing = true;
      _queryEmbeddingInfo = 'Computing query embedding...';
      _similarityResults = '';
    });

    try {
      final query = _queryController.text.trim();

      // Compute embedding for query (mock for demo)
      await Future.delayed(const Duration(milliseconds: 300));
      final queryEmbedding = _generateMockEmbedding(query);

      setState(() {
        _queryEmbeddingInfo =
            '''Query: "$query"
Embedding dimensions: ${queryEmbedding.length}
Sample values: [${queryEmbedding.take(5).map((v) => v.toStringAsFixed(3)).join(', ')}...]''';
      });

      // Compute similarities
      final similarities = <SimilarityResult>[];
      for (final doc in _documents) {
        final docEmbedding = _documentEmbeddings[doc]!;
        final similarity = _computeSimilarity(queryEmbedding, docEmbedding);
        similarities.add(SimilarityResult(text: doc, similarity: similarity, index: _documents.indexOf(doc)));
      }

      // Sort by similarity (highest first)
      similarities.sort((a, b) => b.similarity.compareTo(a.similarity));

      // Display results
      final buffer = StringBuffer();
      buffer.writeln('🔍 Similarity Search Results:\n');

      for (int i = 0; i < similarities.length; i++) {
        final result = similarities[i];
        final percentage = (result.similarity * 100).toStringAsFixed(1);
        buffer.writeln('${i + 1}. Similarity: $percentage%');
        buffer.writeln('   "${result.text}"\n');
      }

      setState(() {
        _similarityResults = buffer.toString();
      });
    } catch (e) {
      setState(() {
        _similarityResults = 'Error computing similarities: $e';
      });
    } finally {
      setState(() {
        _isComputing = false;
      });
    }
  }

  /// Perform RAG (Retrieval Augmented Generation)
  Future<void> _performRAG() async {
    if (_queryController.text.trim().isEmpty) return;

    setState(() {
      _isComputing = true;
      _similarityResults = 'Performing RAG...';
    });

    try {
      final query = _queryController.text.trim();

      // Find most relevant document
      final queryEmbedding = _generateMockEmbedding(query);
      double bestSimilarity = -1.0;
      String bestDocument = '';

      for (final doc in _documents) {
        final docEmbedding = _documentEmbeddings[doc]!;
        final similarity = _computeSimilarity(queryEmbedding, docEmbedding);
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestDocument = doc;
        }
      }

      // Create RAG prompt
      final ragPrompt =
          '''Context: $bestDocument

Question: $query

Answer based on the context above:''';

      setState(() {
        _similarityResults =
            '''🤖 RAG Generation in progress...

📄 Most relevant context (${(bestSimilarity * 100).toStringAsFixed(1)}% similarity):
"$bestDocument"

🤔 Question: "$query"

💭 Generating answer...''';
      });

      // Generate answer using the context
      final answer = await _llama!.generateEnhanced(
        ragPrompt,
        config: SamplerConfig.precise.copyWith(maxTokens: 100, stopStrings: ['Context:', 'Question:']),
      );

      setState(() {
        _similarityResults =
            '''✅ RAG Complete!

📄 Context used (${(bestSimilarity * 100).toStringAsFixed(1)}% similarity):
"$bestDocument"

🤔 Question: "$query"

🤖 AI Answer:
${answer ?? "Failed to generate answer"}''';
      });
    } catch (e) {
      setState(() {
        _similarityResults = 'RAG Error: $e';
      });
    } finally {
      setState(() {
        _isComputing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Embeddings Demo'),
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 24),
              Text(_loadingStatus, style: const TextStyle(fontSize: 18), textAlign: TextAlign.center),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Embeddings Demo'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Info card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('🧠 Embeddings & RAG Demo', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    const Text(
                      'This demo shows text embeddings and Retrieval Augmented Generation (RAG). '
                      'Enter a query to find similar documents or get AI answers based on relevant context.',
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Sample Documents (${_documents.length}):',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    ...(_documents.map(
                      (doc) => Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Text('• $doc', style: const TextStyle(fontSize: 12)),
                      ),
                    )),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Query input
            TextField(
              controller: _queryController,
              decoration: const InputDecoration(
                labelText: 'Enter your query',
                hintText: 'e.g., "What is the capital of France?" or "Tell me about programming"',
                border: OutlineInputBorder(),
              ),
              maxLines: 2,
            ),

            const SizedBox(height: 16),

            // Action buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isComputing ? null : _searchSimilarDocuments,
                    icon: const Icon(Icons.search),
                    label: const Text('Find Similar'),
                    style: ElevatedButton.styleFrom(backgroundColor: Colors.blue, foregroundColor: Colors.white),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isComputing ? null : _performRAG,
                    icon: const Icon(Icons.auto_awesome),
                    label: const Text('RAG Answer'),
                    style: ElevatedButton.styleFrom(backgroundColor: Colors.green, foregroundColor: Colors.white),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 24),

            // Query embedding info
            if (_queryEmbeddingInfo.isNotEmpty) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('🔢 Query Embedding', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 8),
                      Text(_queryEmbeddingInfo, style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],

            // Results
            if (_similarityResults.isNotEmpty) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('📊 Results', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 8),
                      Text(_similarityResults, style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                    ],
                  ),
                ),
              ),
            ],

            // Add document section
            const SizedBox(height: 24),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('📝 Add Custom Document', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    const Text('Add your own document to the knowledge base:'),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _documentController,
                      decoration: const InputDecoration(
                        labelText: 'Document content',
                        hintText: 'Enter text to add to the knowledge base...',
                        border: OutlineInputBorder(),
                      ),
                      maxLines: 3,
                    ),
                    const SizedBox(height: 12),
                    ElevatedButton.icon(
                      onPressed: _isComputing ? null : _addDocument,
                      icon: const Icon(Icons.add),
                      label: const Text('Add Document'),
                    ),
                  ],
                ),
              ),
            ),

            // Status indicator
            if (_isComputing) ...[
              const SizedBox(height: 24),
              const Center(
                child: Column(
                  children: [CircularProgressIndicator(), SizedBox(height: 12), Text('Computing embeddings...')],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Add a custom document
  Future<void> _addDocument() async {
    if (_documentController.text.trim().isEmpty) return;

    final newDoc = _documentController.text.trim();
    _documentController.clear();

    setState(() {
      _isComputing = true;
      _similarityResults = 'Adding document and computing embedding...';
    });

    try {
      // Add to documents list
      _documents.add(newDoc);

      // Compute embedding (mock for demo)
      await Future.delayed(const Duration(milliseconds: 500));
      final embedding = _generateMockEmbedding(newDoc);
      _documentEmbeddings[newDoc] = embedding;

      setState(() {
        _similarityResults =
            '''✅ Document Added!

📄 New document: "$newDoc"
🔢 Embedding computed (${embedding.length} dimensions)
📚 Total documents: ${_documents.length}

You can now search against this document!''';
      });
    } catch (e) {
      setState(() {
        _similarityResults = 'Error adding document: $e';
      });
    } finally {
      setState(() {
        _isComputing = false;
      });
    }
  }
}

/// Simple similarity result for demo
class SimilarityResult {
  final String text;
  final double similarity;
  final int index;

  const SimilarityResult({required this.text, required this.similarity, required this.index});
}
