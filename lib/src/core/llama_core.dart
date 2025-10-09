/// Core runtime and initialization for LlamaService
///
/// Provides backend initialization, library loading, model management,
/// and core llama.cpp runtime functionality.

import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import '../ffi/llama_ffi.dart';
import '../core/llama_exceptions.dart';
import '../utils/llama_config.dart';
import '../utils/llama_status.dart';
import '../utils/llama_helpers.dart';

/// Core runtime manager for llama.cpp
class LlamaCoreRuntime with DisposableMixin, StatusMixin {
  late final DynamicLibrary _lib;
  late final llama_cpp _llamaCpp;
  bool _backendInitialized = false;

  /// Initialize the core runtime
  Future<bool> initialize({String? libraryPath}) async {
    checkNotDisposed('initialize');

    try {
      updateStatus(LlamaStatus.uninitialized, message: 'Initializing core runtime');

      // Load library
      _lib = await _loadLibrary(libraryPath);
      _llamaCpp = llama_cpp(_lib);

      // Initialize backend
      LlamaLogger.info('Initializing llama.cpp backend');
      _llamaCpp.llama_backend_init();
      _backendInitialized = true;

      updateStatus(LlamaStatus.initialized, message: 'Core runtime initialized');
      LlamaLogger.info('✓ LlamaCore initialized successfully');

      return true;
    } catch (e, stackTrace) {
      updateStatus(LlamaStatus.error, message: 'Initialization failed', error: e);
      throw LlamaException(
        'Failed to initialize core runtime: $e',
        operation: 'initialize',
        originalError: e,
        stackTrace: stackTrace,
      );
    }
  }

  /// Load the llama.cpp dynamic library
  Future<DynamicLibrary> _loadLibrary(String? libraryPath) async {
    libraryPath ??= _getDefaultLibraryPath();

    try {
      LlamaLogger.debug('Loading library from: $libraryPath');
      return DynamicLibrary.open(libraryPath);
    } catch (e) {
      // Try alternative paths
      final alternativePaths = _getAlternativeLibraryPaths();

      for (final altPath in alternativePaths) {
        try {
          LlamaLogger.debug('Trying alternative path: $altPath');
          return DynamicLibrary.open(altPath);
        } catch (e) {
          continue;
        }
      }

      throw LlamaException('Failed to load llama.cpp library from any path');
    }
  }

  /// Get default library path for current platform
  String _getDefaultLibraryPath() {
    if (Platform.isMacOS) {
      return 'llama.framework/llama';
    } else if (Platform.isLinux) {
      return 'libllama.so';
    } else if (Platform.isWindows) {
      return 'llama.dll';
    } else if (Platform.isAndroid) {
      return 'libllama.so';
    } else if (Platform.isIOS) {
      return 'llama.framework/llama';
    } else {
      throw LlamaException('Unsupported platform: ${Platform.operatingSystem}');
    }
  }

  /// Get alternative library paths to try
  List<String> _getAlternativeLibraryPaths() {
    final alternatives = <String>[];

    if (Platform.isMacOS) {
      alternatives.addAll([
        '/usr/local/lib/libllama.dylib',
        '/opt/homebrew/lib/libllama.dylib',
        'Frameworks/llama.framework/llama',
      ]);
    } else if (Platform.isLinux) {
      alternatives.addAll(['/usr/lib/libllama.so', '/usr/local/lib/libllama.so', './libllama.so']);
    } else if (Platform.isWindows) {
      alternatives.addAll(['llama.dll', './llama.dll', 'bin/llama.dll']);
    }

    return alternatives;
  }

  /// Get the llama.cpp FFI bindings
  llama_cpp get llamaCpp {
    checkNotDisposed('llamaCpp');
    if (!_backendInitialized) {
      throw LlamaNotInitializedException('llamaCpp');
    }
    return _llamaCpp;
  }

  /// Check if backend is initialized
  bool get isInitialized => _backendInitialized && !isDisposed;

  /// Get system information
  String getSystemInfo() {
    checkNotDisposed('getSystemInfo');
    if (!_backendInitialized) {
      throw LlamaNotInitializedException('getSystemInfo');
    }

    final infoPtr = _llamaCpp.llama_print_system_info();
    if (infoPtr == nullptr) {
      return 'System info not available';
    }

    return infoPtr.cast<Utf8>().toDartString();
  }

  /// Get llama.cpp capabilities
  Map<String, bool> getCapabilities() {
    checkNotDisposed('getCapabilities');
    if (!_backendInitialized) {
      throw LlamaNotInitializedException('getCapabilities');
    }

    return {
      'mmap': _llamaCpp.llama_supports_mmap(),
      'mlock': _llamaCpp.llama_supports_mlock(),
      'gpu_offload': _llamaCpp.llama_supports_gpu_offload(),
      'rpc': _llamaCpp.llama_supports_rpc(),
    };
  }

  /// Get maximum supported values
  Map<String, int> getMaximums() {
    checkNotDisposed('getMaximums');
    if (!_backendInitialized) {
      throw LlamaNotInitializedException('getMaximums');
    }

    return {'devices': _llamaCpp.llama_max_devices(), 'parallel_sequences': _llamaCpp.llama_max_parallel_sequences()};
  }

  void dispose() {
    if (_backendInitialized) {
      try {
        _llamaCpp.llama_backend_free();
        _backendInitialized = false;
        LlamaLogger.info('✓ LlamaCore backend freed');
      } catch (e) {
        LlamaLogger.error('Error freeing backend', e);
      }
    }

    updateStatus(LlamaStatus.disposed, message: 'Core runtime disposed');
    disposeStatus();
    markDisposed();
  }
}

/// Model manager for handling model loading and management
class LlamaModelManager with DisposableMixin {
  final llama_cpp _llamaCpp;
  Pointer<llama_model>? _model;
  Pointer<llama_vocab>? _vocab;
  String? _modelPath;
  ModelConfig? _modelConfig;

  LlamaModelManager(this._llamaCpp);

  /// Load a model from file
  Future<bool> loadModel(String modelPath, {ModelConfig? config}) async {
    checkNotDisposed('loadModel');
    config ??= const ModelConfig();

    if (!File(modelPath).existsSync()) {
      throw LlamaModelLoadException(modelPath, 'File does not exist');
    }

    try {
      LlamaLogger.info('Loading model: $modelPath');

      // Get default parameters and configure
      final modelParams = _llamaCpp.llama_model_default_params();
      _configureModelParams(modelParams, config);

      // Load model
      final pathPtr = modelPath.toNativeUtf8();
      try {
        _model = _llamaCpp.llama_load_model_from_file(pathPtr.cast<Char>(), modelParams);

        if (_model == nullptr || _model!.address == 0) {
          throw LlamaModelLoadException(modelPath, 'llama_load_model_from_file returned null');
        }

        // Get vocab handle
        _vocab = _llamaCpp.llama_model_get_vocab(_model!);
        if (_vocab == nullptr || _vocab!.address == 0) {
          LlamaLogger.warn('Failed to get vocab handle');
        }

        _modelPath = modelPath;
        _modelConfig = config;

        LlamaLogger.info('✓ Model loaded successfully');
        _logModelInfo();

        return true;
      } finally {
        malloc.free(pathPtr);
      }
    } catch (e) {
      if (e is LlamaException) rethrow;
      throw LlamaModelLoadException(modelPath, e.toString());
    }
  }

  /// Configure model parameters from config
  void _configureModelParams(llama_model_params params, ModelConfig config) {
    params.n_gpu_layers = config.nGpuLayers;
    params.main_gpu = config.mainGpu;
    params.use_mmap = config.useMmap;
    params.use_mlock = config.useMlock;
    params.vocab_only = config.vocabOnly;
    params.check_tensors = config.checkTensors;
    params.use_extra_bufts = config.useExtraBuffers;

    // Configure tensor split if provided
    if (config.tensorSplit != null && config.tensorSplit!.isNotEmpty) {
      final tensorSplitPtr = malloc<Float>(config.tensorSplit!.length);
      for (int i = 0; i < config.tensorSplit!.length; i++) {
        tensorSplitPtr[i] = config.tensorSplit![i];
      }
      params.tensor_split = tensorSplitPtr;
      // Note: This pointer should be freed when the model is freed
    }

    LlamaLogger.debug('Model params configured: GPU layers=${config.nGpuLayers}, mmap=${config.useMmap}');
  }

  /// Log model information
  void _logModelInfo() {
    if (_model == null) return;

    final info = {
      'n_embd': _llamaCpp.llama_model_n_embd(_model!),
      'n_layer': _llamaCpp.llama_model_n_layer(_model!),
      'n_head': _llamaCpp.llama_model_n_head(_model!),
      'n_params': _llamaCpp.llama_model_n_params(_model!),
      'size_bytes': _llamaCpp.llama_model_size(_model!),
      'has_encoder': _llamaCpp.llama_model_has_encoder(_model!),
      'has_decoder': _llamaCpp.llama_model_has_decoder(_model!),
      'is_recurrent': _llamaCpp.llama_model_is_recurrent(_model!),
    };

    LlamaLogger.info('Model info: ${info.toString()}');
  }

  /// Get model metadata
  Map<String, String> getModelMetadata() {
    checkNotDisposed('getModelMetadata');
    if (_model == null) {
      throw LlamaModelNotLoadedException('getModelMetadata');
    }

    final metadata = <String, String>{};
    final metaCount = _llamaCpp.llama_model_meta_count(_model!);

    for (int i = 0; i < metaCount; i++) {
      final keyBuffer = malloc<Char>(256);
      final valueBuffer = malloc<Char>(1024);

      try {
        final keyLen = _llamaCpp.llama_model_meta_key_by_index(_model!, i, keyBuffer, 256);
        final valueLen = _llamaCpp.llama_model_meta_val_str_by_index(_model!, i, valueBuffer, 1024);

        if (keyLen > 0 && valueLen > 0) {
          final key = keyBuffer.cast<Utf8>().toDartString();
          final value = valueBuffer.cast<Utf8>().toDartString();
          metadata[key] = value;
        }
      } finally {
        malloc.free(keyBuffer);
        malloc.free(valueBuffer);
      }
    }

    return metadata;
  }

  /// Get model description
  String getModelDescription() {
    checkNotDisposed('getModelDescription');
    if (_model == null) {
      throw LlamaModelNotLoadedException('getModelDescription');
    }

    final buffer = malloc<Char>(1024);
    try {
      final length = _llamaCpp.llama_model_desc(_model!, buffer, 1024);
      if (length > 0) {
        return buffer.cast<Utf8>().toDartString();
      }
      return 'No description available';
    } finally {
      malloc.free(buffer);
    }
  }

  /// Check if model supports embeddings
  bool supportsEmbeddings() {
    checkNotDisposed('supportsEmbeddings');
    if (_model == null) return false;

    return _llamaCpp.llama_n_embd(_model!) > 0;
  }

  /// Get loaded model pointer
  Pointer<llama_model>? get model => _model;

  /// Get vocab pointer
  Pointer<llama_vocab>? get vocab => _vocab;

  /// Get model path
  String? get modelPath => _modelPath;

  /// Get model configuration
  ModelConfig? get modelConfig => _modelConfig;

  /// Check if model is loaded
  bool get isLoaded => _model != null && _model!.address != 0;

  void dispose() {
    if (_model != null && _model!.address != 0) {
      _llamaCpp.llama_model_free(_model!);
      _model = null;
      _vocab = null;
      LlamaLogger.info('✓ Model freed');
    }

    _modelPath = null;
    _modelConfig = null;
    markDisposed();
  }
}

/// Context manager for handling inference contexts
class LlamaContextManager with DisposableMixin {
  final llama_cpp _llamaCpp;
  final Pointer<llama_model> _model;

  Pointer<llama_context>? _context;
  ContextConfig? _contextConfig;

  LlamaContextManager(this._llamaCpp, this._model);

  /// Create a new context
  Future<bool> createContext({ContextConfig? config}) async {
    checkNotDisposed('createContext');
    config ??= const ContextConfig();

    if (_context != null && _context!.address != 0) {
      LlamaLogger.warn('Context already exists, disposing old one');
      _disposeContext();
    }

    try {
      LlamaLogger.info('Creating context with config: ${config.toJson()}');

      // Get default parameters and configure
      final ctxParams = _llamaCpp.llama_context_default_params();
      _configureContextParams(ctxParams, config);

      // Create context
      _context = _llamaCpp.llama_new_context_with_model(_model, ctxParams);

      if (_context == nullptr || _context!.address == 0) {
        throw LlamaException('llama_new_context_with_model returned null');
      }

      _contextConfig = config;

      // Log actual context info
      final actualCtx = _llamaCpp.llama_n_ctx(_context!);
      LlamaLogger.info('✓ Context created successfully');
      LlamaLogger.info('  Requested n_ctx: ${config.nCtx}');
      LlamaLogger.info('  Actual n_ctx: $actualCtx');
      LlamaLogger.info('  Threads: ${config.nThreads}');
      LlamaLogger.info('  Embeddings: ${config.embeddings}');

      return true;
    } catch (e) {
      if (e is LlamaException) rethrow;
      throw LlamaException('Failed to create context: $e', operation: 'createContext');
    }
  }

  /// Configure context parameters from config
  void _configureContextParams(llama_context_params params, ContextConfig config) {
    params.n_ctx = config.nCtx;
    params.n_batch = config.nBatch;
    params.n_ubatch = config.nUbatch;
    params.n_seq_max = config.nSeqMax;
    params.n_threads = config.nThreads;
    params.n_threads_batch = config.nThreadsBatch;
    params.rope_scaling_typeAsInt = config.ropeScalingType;
    params.pooling_typeAsInt = config.poolingType;
    params.attention_typeAsInt = config.attentionType;
    params.rope_freq_base = config.ropeFreqBase;
    params.rope_freq_scale = config.ropeFreqScale;
    params.yarn_ext_factor = config.yarnExtFactor;
    params.yarn_attn_factor = config.yarnAttnFactor;
    params.yarn_beta_fast = config.yarnBetaFast;
    params.yarn_beta_slow = config.yarnBetaSlow;
    params.yarn_orig_ctx = config.yarnOrigCtx;
    params.defrag_thold = config.defragThold;
    params.embeddings = config.embeddings;
    params.offload_kqv = config.offloadKqv;
    params.no_perf = config.noPerf;
  }

  /// Update thread configuration at runtime
  void setThreads({int? nThreads, int? nThreadsBatch}) {
    checkNotDisposed('setThreads');
    if (_context == null) {
      throw LlamaContextNotCreatedException('setThreads');
    }

    final currentThreads = _llamaCpp.llama_n_threads(_context!);
    final currentBatchThreads = _llamaCpp.llama_n_threads_batch(_context!);

    final newThreads = nThreads ?? currentThreads;
    final newBatchThreads = nThreadsBatch ?? currentBatchThreads;

    if (newThreads != currentThreads || newBatchThreads != currentBatchThreads) {
      _llamaCpp.llama_set_n_threads(_context!, newThreads, newBatchThreads);
      LlamaLogger.info('✓ Updated threads: $newThreads (batch: $newBatchThreads)');
    }
  }

  /// Clear KV cache
  void clearKvCache() {
    checkNotDisposed('clearKvCache');
    if (_context == null) {
      throw LlamaContextNotCreatedException('clearKvCache');
    }

    final memory = _llamaCpp.llama_get_memory(_context!);
    _llamaCpp.llama_memory_clear(memory, true);
    LlamaLogger.info('✓ KV cache cleared');
  }

  /// Get context information
  Map<String, dynamic> getContextInfo() {
    checkNotDisposed('getContextInfo');
    if (_context == null) {
      throw LlamaContextNotCreatedException('getContextInfo');
    }

    return {
      'n_ctx': _llamaCpp.llama_n_ctx(_context!),
      'n_batch': _llamaCpp.llama_n_batch(_context!),
      'n_ubatch': _llamaCpp.llama_n_ubatch(_context!),
      'n_seq_max': _llamaCpp.llama_n_seq_max(_context!),
      'n_threads': _llamaCpp.llama_n_threads(_context!),
      'n_threads_batch': _llamaCpp.llama_n_threads_batch(_context!),
      'pooling_type': _llamaCpp.llama_pooling_type$1(_context!).value,
    };
  }

  /// Get context pointer
  Pointer<llama_context>? get context => _context;

  /// Get context configuration
  ContextConfig? get contextConfig => _contextConfig;

  /// Check if context is created
  bool get isCreated => _context != null && _context!.address != 0;

  /// Dispose context only (keep manager alive)
  void _disposeContext() {
    if (_context != null && _context!.address != 0) {
      _llamaCpp.llama_free(_context!);
      _context = null;
      LlamaLogger.info('✓ Context disposed');
    }
  }

  void dispose() {
    _disposeContext();
    _contextConfig = null;
    markDisposed();
  }
}

/// Library loader with platform-specific handling
class LibraryLoader {
  /// Load libraries in correct order for the platform
  static Future<Map<String, bool>> loadPlatformLibraries() async {
    final results = <String, bool>{};

    if (Platform.isMacOS) {
      return await _loadMacOSLibraries();
    } else if (Platform.isAndroid) {
      return await _loadAndroidLibraries();
    } else if (Platform.isLinux) {
      return await _loadLinuxLibraries();
    } else if (Platform.isWindows) {
      return await _loadWindowsLibraries();
    }

    return results;
  }

  /// Load macOS framework libraries
  static Future<Map<String, bool>> _loadMacOSLibraries() async {
    final libraries = [
      'ggml-base.framework/ggml-base',
      'ggml-blas.framework/ggml-blas',
      'ggml-cpu.framework/ggml-cpu',
      'ggml-metal.framework/ggml-metal',
      'ggml.framework/ggml',
      'llama.framework/llama',
      'mtmd.framework/mtmd',
    ];

    final results = <String, bool>{};
    final loadedLibs = <String, DynamicLibrary>{};

    for (final libPath in libraries) {
      final libName = libPath.split('/').last;
      try {
        final dylib = DynamicLibrary.open(libPath);
        loadedLibs[libName] = dylib;
        results[libName] = true;
        LlamaLogger.debug('✓ Loaded: $libName');
      } catch (e) {
        results[libName] = false;
        LlamaLogger.warn('✗ Failed to load $libName: $e');
      }
    }

    return results;
  }

  /// Load Android shared libraries
  static Future<Map<String, bool>> _loadAndroidLibraries() async {
    final libraries = ['libggml.so', 'libllama.so'];

    final results = <String, bool>{};

    for (final libName in libraries) {
      try {
        DynamicLibrary.open(libName);
        results[libName] = true;
        LlamaLogger.debug('✓ Loaded: $libName');
      } catch (e) {
        results[libName] = false;
        LlamaLogger.warn('✗ Failed to load $libName: $e');
      }
    }

    return results;
  }

  /// Load Linux shared libraries
  static Future<Map<String, bool>> _loadLinuxLibraries() async {
    final libraries = ['libggml.so', 'libllama.so'];

    final results = <String, bool>{};

    for (final libName in libraries) {
      try {
        DynamicLibrary.open(libName);
        results[libName] = true;
        LlamaLogger.debug('✓ Loaded: $libName');
      } catch (e) {
        results[libName] = false;
        LlamaLogger.warn('✗ Failed to load $libName: $e');
      }
    }

    return results;
  }

  /// Load Windows DLL libraries
  static Future<Map<String, bool>> _loadWindowsLibraries() async {
    final libraries = ['ggml.dll', 'llama.dll'];

    final results = <String, bool>{};

    for (final libName in libraries) {
      try {
        DynamicLibrary.open(libName);
        results[libName] = true;
        LlamaLogger.debug('✓ Loaded: $libName');
      } catch (e) {
        results[libName] = false;
        LlamaLogger.warn('✗ Failed to load $libName: $e');
      }
    }

    return results;
  }
}
