import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

// Opaque types
final class LlamaModel extends Opaque {}

final class LlamaContext extends Opaque {}

final class LlamaVocab extends Opaque {}

final class LlamaSampler extends Opaque {}

// llama_batch struct
final class LlamaBatch extends Struct {
  @Int32()
  external int n_tokens;

  external Pointer<Int32> token;
  external Pointer<Float> embd;
  external Pointer<Int32> pos;
  external Pointer<Int32> n_seq_id;
  external Pointer<Pointer<Int32>> seq_id;
  external Pointer<Int8> logits;
}

// llama_sampler_chain_params struct
final class LlamaSamplerChainParams extends Struct {
  @Bool()
  external bool no_perf;
}

// llama_model_params struct
final class LlamaModelParams extends Struct {
  external Pointer<Void> devices;
  external Pointer<Void> tensor_buft_overrides;

  @Int32()
  external int n_gpu_layers;

  @Uint32()
  external int split_mode;

  @Int32()
  external int main_gpu;

  external Pointer<Float> tensor_split;
  external Pointer<Void> progress_callback;
  external Pointer<Void> progress_callback_user_data;
  external Pointer<Void> kv_overrides;

  @Bool()
  external bool vocab_only;

  @Bool()
  external bool use_mmap;

  @Bool()
  external bool use_mlock;

  @Bool()
  external bool check_tensors;

  @Bool()
  external bool use_extra_bufts;
}

// llama_context_params struct
final class LlamaContextParams extends Struct {
  @Uint32()
  external int n_ctx;

  @Uint32()
  external int n_batch;

  @Uint32()
  external int n_ubatch;

  @Uint32()
  external int n_seq_max;

  @Uint32()
  external int n_threads;

  @Uint32()
  external int n_threads_batch;

  @Int32()
  external int rope_scaling_type;

  @Int32()
  external int pooling_type;

  @Int32()
  external int attention_type;

  @Float()
  external double rope_freq_base;

  @Float()
  external double rope_freq_scale;

  @Float()
  external double yarn_ext_factor;

  @Float()
  external double yarn_attn_factor;

  @Float()
  external double yarn_beta_fast;

  @Float()
  external double yarn_beta_slow;

  @Uint32()
  external int yarn_orig_ctx;

  @Float()
  external double defrag_thold;

  external Pointer<Void> cb_eval;
  external Pointer<Void> cb_eval_user_data;

  @Int32()
  external int type_k;

  @Int32()
  external int type_v;

  @Bool()
  external bool logits_all;

  @Bool()
  external bool embeddings;

  @Bool()
  external bool offload_kqv;

  @Bool()
  external bool flash_attn;

  @Bool()
  external bool no_perf;

  external Pointer<Void> abort_callback;
  external Pointer<Void> abort_callback_data;
}

// Native function signatures - Model
typedef LlamaBackendInitNative = Void Function();
typedef LlamaBackendInit = void Function();

typedef LlamaBackendFreeNative = Void Function();
typedef LlamaBackendFree = void Function();

typedef LlamaModelDefaultParamsNative = LlamaModelParams Function();
typedef LlamaModelDefaultParams = LlamaModelParams Function();

typedef LlamaLoadModelFromFileNative = Pointer<LlamaModel> Function(Pointer<Utf8>, LlamaModelParams);
typedef LlamaLoadModelFromFile = Pointer<LlamaModel> Function(Pointer<Utf8>, LlamaModelParams);

typedef LlamaFreeModelNative = Void Function(Pointer<LlamaModel>);
typedef LlamaFreeModel = void Function(Pointer<LlamaModel>);

// Native function signatures - Context
typedef LlamaContextDefaultParamsNative = LlamaContextParams Function();
typedef LlamaContextDefaultParams = LlamaContextParams Function();

typedef LlamaNewContextWithModelNative = Pointer<LlamaContext> Function(Pointer<LlamaModel>, LlamaContextParams);
typedef LlamaNewContextWithModel = Pointer<LlamaContext> Function(Pointer<LlamaModel>, LlamaContextParams);

typedef LlamaFreeContextNative = Void Function(Pointer<LlamaContext>);
typedef LlamaFreeContext = void Function(Pointer<LlamaContext>);

typedef LlamaNCtxNative = Uint32 Function(Pointer<LlamaContext>);
typedef LlamaNCtx = int Function(Pointer<LlamaContext>);

// Native function signatures - Vocab
typedef LlamaModelGetVocabNative = Pointer<LlamaVocab> Function(Pointer<LlamaModel>);
typedef LlamaModelGetVocab = Pointer<LlamaVocab> Function(Pointer<LlamaModel>);

// Native function signatures - Tokenization
typedef LlamaTokenizeNative =
    Int32 Function(Pointer<LlamaVocab>, Pointer<Utf8>, Int32, Pointer<Int32>, Int32, Bool, Bool);
typedef LlamaTokenize = int Function(Pointer<LlamaVocab>, Pointer<Utf8>, int, Pointer<Int32>, int, bool, bool);

typedef LlamaTokenToPieceNative = Int32 Function(Pointer<LlamaVocab>, Int32, Pointer<Int8>, Int32, Int32, Bool);
typedef LlamaTokenToPiece = int Function(Pointer<LlamaVocab>, int, Pointer<Int8>, int, int, bool);

typedef LlamaTokenBosNative = Int32 Function(Pointer<LlamaVocab>);
typedef LlamaTokenBos = int Function(Pointer<LlamaVocab>);

typedef LlamaTokenEosNative = Int32 Function(Pointer<LlamaVocab>);
typedef LlamaTokenEos = int Function(Pointer<LlamaVocab>);

typedef LlamaTokenIsEogNative = Bool Function(Pointer<LlamaVocab>, Int32);
typedef LlamaTokenIsEog = bool Function(Pointer<LlamaVocab>, int);

// Native function signatures - Batch
typedef LlamaBatchGetOneNative = LlamaBatch Function(Pointer<Int32>, Int32);
typedef LlamaBatchGetOne = LlamaBatch Function(Pointer<Int32>, int);

typedef LlamaBatchInitNative = LlamaBatch Function(Int32, Int32, Int32);
typedef LlamaBatchInit = LlamaBatch Function(int, int, int);

typedef LlamaBatchFreeNative = Void Function(LlamaBatch);
typedef LlamaBatchFree = void Function(LlamaBatch);

// Native function signatures - Decode
typedef LlamaDecodeNative = Int32 Function(Pointer<LlamaContext>, LlamaBatch);
typedef LlamaDecode = int Function(Pointer<LlamaContext>, LlamaBatch);

typedef LlamaGetLogitsIthNative = Pointer<Float> Function(Pointer<LlamaContext>, Int32);
typedef LlamaGetLogitsIth = Pointer<Float> Function(Pointer<LlamaContext>, int);

typedef LlamaNVocabNative = Int32 Function(Pointer<LlamaVocab>);
typedef LlamaNVocab = int Function(Pointer<LlamaVocab>);

// Native function signatures - Sampler
typedef LlamaSamplerChainDefaultParamsNative = LlamaSamplerChainParams Function();
typedef LlamaSamplerChainDefaultParams = LlamaSamplerChainParams Function();

typedef LlamaSamplerChainInitNative = Pointer<LlamaSampler> Function(LlamaSamplerChainParams);
typedef LlamaSamplerChainInit = Pointer<LlamaSampler> Function(LlamaSamplerChainParams);

typedef LlamaSamplerChainAddNative = Void Function(Pointer<LlamaSampler>, Pointer<LlamaSampler>);
typedef LlamaSamplerChainAdd = void Function(Pointer<LlamaSampler>, Pointer<LlamaSampler>);

typedef LlamaSamplerInitGreedyNative = Pointer<LlamaSampler> Function();
typedef LlamaSamplerInitGreedy = Pointer<LlamaSampler> Function();

typedef LlamaSamplerSampleNative = Int32 Function(Pointer<LlamaSampler>, Pointer<LlamaContext>, Int32);
typedef LlamaSamplerSample = int Function(Pointer<LlamaSampler>, Pointer<LlamaContext>, int);

typedef LlamaSamplerFreeNative = Void Function(Pointer<LlamaSampler>);
typedef LlamaSamplerFree = void Function(Pointer<LlamaSampler>);

class LlamaSimple {
  late final DynamicLibrary _lib;

  // Model functions
  late final LlamaBackendInit _backendInit;
  late final LlamaBackendFree _backendFree;
  late final LlamaModelDefaultParams _modelDefaultParams;
  late final LlamaLoadModelFromFile _loadModelFromFile;
  late final LlamaFreeModel _freeModel;

  // Context functions
  late final LlamaContextDefaultParams _contextDefaultParams;
  late final LlamaNewContextWithModel _newContextWithModel;
  late final LlamaFreeContext _freeContext;
  late final LlamaNCtx _nCtx;

  // Vocab functions
  late final LlamaModelGetVocab _modelGetVocab;

  // Tokenization functions
  late final LlamaTokenize _tokenize;
  late final LlamaTokenToPiece _tokenToPiece;
  late final LlamaTokenBos _tokenBos;
  late final LlamaTokenEos _tokenEos;
  late final LlamaTokenIsEog _tokenIsEog;

  // Batch functions
  late final LlamaBatchGetOne _batchGetOne;
  late final LlamaBatchInit _batchInit;
  late final LlamaBatchFree _batchFree;

  // Decode functions
  late final LlamaDecode _decode;
  late final LlamaGetLogitsIth _getLogitsIth;
  late final LlamaNVocab _nVocab;

  // Sampler functions
  late final LlamaSamplerChainDefaultParams _samplerChainDefaultParams;
  late final LlamaSamplerChainInit _samplerChainInit;
  late final LlamaSamplerChainAdd _samplerChainAdd;
  late final LlamaSamplerInitGreedy _samplerInitGreedy;
  late final LlamaSamplerSample _samplerSample;
  late final LlamaSamplerFree _samplerFree;

  Pointer<LlamaModel>? _model;
  Pointer<LlamaContext>? _context;
  Pointer<LlamaVocab>? _vocab;
  Pointer<LlamaSampler>? _sampler;
  bool _initialized = false;

  LlamaSimple() {
    _lib = DynamicLibrary.open('llama.framework/llama');
    _bindFunctions();
  }

  void _bindFunctions() {
    // Model functions
    _backendInit = _lib.lookup<NativeFunction<LlamaBackendInitNative>>('llama_backend_init').asFunction();
    _backendFree = _lib.lookup<NativeFunction<LlamaBackendFreeNative>>('llama_backend_free').asFunction();
    _modelDefaultParams = _lib
        .lookup<NativeFunction<LlamaModelDefaultParamsNative>>('llama_model_default_params')
        .asFunction();
    _loadModelFromFile = _lib
        .lookup<NativeFunction<LlamaLoadModelFromFileNative>>('llama_load_model_from_file')
        .asFunction();
    _freeModel = _lib.lookup<NativeFunction<LlamaFreeModelNative>>('llama_free_model').asFunction();

    // Context functions
    _contextDefaultParams = _lib
        .lookup<NativeFunction<LlamaContextDefaultParamsNative>>('llama_context_default_params')
        .asFunction();
    _newContextWithModel = _lib
        .lookup<NativeFunction<LlamaNewContextWithModelNative>>('llama_new_context_with_model')
        .asFunction();
    _freeContext = _lib.lookup<NativeFunction<LlamaFreeContextNative>>('llama_free').asFunction();
    _nCtx = _lib.lookup<NativeFunction<LlamaNCtxNative>>('llama_n_ctx').asFunction();

    // Vocab functions
    _modelGetVocab = _lib.lookup<NativeFunction<LlamaModelGetVocabNative>>('llama_model_get_vocab').asFunction();

    // Tokenization functions
    _tokenize = _lib.lookup<NativeFunction<LlamaTokenizeNative>>('llama_tokenize').asFunction();
    _tokenToPiece = _lib.lookup<NativeFunction<LlamaTokenToPieceNative>>('llama_token_to_piece').asFunction();
    _tokenBos = _lib.lookup<NativeFunction<LlamaTokenBosNative>>('llama_token_bos').asFunction();
    _tokenEos = _lib.lookup<NativeFunction<LlamaTokenEosNative>>('llama_token_eos').asFunction();
    _tokenIsEog = _lib.lookup<NativeFunction<LlamaTokenIsEogNative>>('llama_token_is_eog').asFunction();

    // Batch functions
    _batchGetOne = _lib.lookup<NativeFunction<LlamaBatchGetOneNative>>('llama_batch_get_one').asFunction();
    _batchInit = _lib.lookup<NativeFunction<LlamaBatchInitNative>>('llama_batch_init').asFunction();
    _batchFree = _lib.lookup<NativeFunction<LlamaBatchFreeNative>>('llama_batch_free').asFunction();

    // Decode functions
    _decode = _lib.lookup<NativeFunction<LlamaDecodeNative>>('llama_decode').asFunction();
    _getLogitsIth = _lib.lookup<NativeFunction<LlamaGetLogitsIthNative>>('llama_get_logits_ith').asFunction();
    _nVocab = _lib.lookup<NativeFunction<LlamaNVocabNative>>('llama_n_vocab').asFunction();

    // Sampler functions
    _samplerChainDefaultParams = _lib
        .lookup<NativeFunction<LlamaSamplerChainDefaultParamsNative>>('llama_sampler_chain_default_params')
        .asFunction();
    _samplerChainInit = _lib
        .lookup<NativeFunction<LlamaSamplerChainInitNative>>('llama_sampler_chain_init')
        .asFunction();
    _samplerChainAdd = _lib.lookup<NativeFunction<LlamaSamplerChainAddNative>>('llama_sampler_chain_add').asFunction();
    _samplerInitGreedy = _lib
        .lookup<NativeFunction<LlamaSamplerInitGreedyNative>>('llama_sampler_init_greedy')
        .asFunction();
    _samplerSample = _lib.lookup<NativeFunction<LlamaSamplerSampleNative>>('llama_sampler_sample').asFunction();
    _samplerFree = _lib.lookup<NativeFunction<LlamaSamplerFreeNative>>('llama_sampler_free').asFunction();
  }

  void init() {
    if (_initialized) return;
    _backendInit();
    _initialized = true;
  }

  bool loadModel(String modelPath) {
    if (!_initialized) {
      throw StateError('Must call init() first');
    }

    if (!File(modelPath).existsSync()) {
      print('Model file not found: $modelPath');
      return false;
    }

    // Get default params
    final modelParams = _modelDefaultParams();

    // Enable memory mapping
    modelParams.use_mmap = true;
    modelParams.use_mlock = false;
    modelParams.n_gpu_layers = 0; // CPU only for now

    print('Model params:');
    print('  use_mmap: ${modelParams.use_mmap}');
    print('  use_mlock: ${modelParams.use_mlock}');
    print('  n_gpu_layers: ${modelParams.n_gpu_layers}');

    // Load model
    final pathPtr = modelPath.toNativeUtf8();
    try {
      _model = _loadModelFromFile(pathPtr, modelParams);

      if (_model == nullptr || _model!.address == 0) {
        print('Failed to load model');
        return false;
      }

      print('✓ Model loaded successfully with memory mapping!');

      // Get vocab handle for tokenization
      _vocab = _modelGetVocab(_model!);
      if (_vocab == nullptr || _vocab!.address == 0) {
        print('Warning: Failed to get vocab handle');
      }

      return true;
    } finally {
      malloc.free(pathPtr);
    }
  }

  bool createContext({int nCtx = 2048, int nThreads = 4}) {
    if (_model == null || _model!.address == 0) {
      print('Must load model first');
      return false;
    }

    if (_context != null && _context!.address != 0) {
      print('Context already exists');
      return false;
    }

    // Get default context params
    final ctxParams = _contextDefaultParams();

    // Set our custom params
    ctxParams.n_ctx = nCtx;
    ctxParams.n_threads = nThreads;
    ctxParams.n_threads_batch = nThreads;

    print('Context params:');
    print('  n_ctx: ${ctxParams.n_ctx}');
    print('  n_threads: ${ctxParams.n_threads}');

    // Create context
    _context = _newContextWithModel(_model!, ctxParams);

    if (_context == nullptr || _context!.address == 0) {
      print('Failed to create context');
      return false;
    }

    final actualCtx = _nCtx(_context!);
    print('✓ Context created successfully! Actual n_ctx: $actualCtx');

    // Initialize greedy sampler
    final samplerParams = _samplerChainDefaultParams();
    samplerParams.no_perf = false;
    _sampler = _samplerChainInit(samplerParams);

    if (_sampler == nullptr || _sampler!.address == 0) {
      print('Failed to create sampler');
      return false;
    }

    final greedySampler = _samplerInitGreedy();
    _samplerChainAdd(_sampler!, greedySampler);
    print('✓ Greedy sampler initialized');

    return true;
  }

  List<int> tokenizeText(String text, {bool addBos = true, bool special = true}) {
    if (_model == null || _model!.address == 0) {
      throw StateError('Must load model first');
    }

    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Vocab not available');
    }

    final textPtr = text.toNativeUtf8();
    try {
      // First pass: get token count (negative return means needed size)
      final tokenCount = -_tokenize(_vocab!, textPtr, text.length, nullptr, 0, addBos, special);

      if (tokenCount <= 0) {
        print('Tokenization returned invalid count: $tokenCount');
        return [];
      }

      print('Need $tokenCount tokens for "$text"');

      // Allocate buffer for tokens
      final tokensPtr = malloc<Int32>(tokenCount);
      try {
        // Second pass: get actual tokens
        final actualCount = _tokenize(_vocab!, textPtr, text.length, tokensPtr, tokenCount, addBos, special);

        if (actualCount < 0) {
          print('Failed to get tokens, error: $actualCount');
          return [];
        }

        // Convert to Dart list
        final tokens = <int>[];
        for (int i = 0; i < actualCount; i++) {
          tokens.add(tokensPtr[i]);
        }

        print('✓ Tokenized "$text" into ${tokens.length} tokens: $tokens');
        return tokens;
      } finally {
        malloc.free(tokensPtr);
      }
    } finally {
      malloc.free(textPtr);
    }
  }

  String detokenize(int token) {
    if (_model == null || _model!.address == 0) {
      throw StateError('Must load model first');
    }

    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Vocab not available');
    }

    // Allocate buffer for the token string (128 bytes should be enough)
    final bufferPtr = malloc<Int8>(128);
    try {
      final length = _tokenToPiece(
        _vocab!,
        token,
        bufferPtr,
        128,
        0, // lstrip (0 = no strip)
        true, // special (render special tokens)
      );

      if (length < 0) {
        print('Failed to detokenize token $token, error: $length');
        return '';
      }

      if (length == 0) {
        return '';
      }

      // Convert to Dart string
      final bytes = bufferPtr.cast<Uint8>().asTypedList(length);
      return String.fromCharCodes(bytes);
    } finally {
      malloc.free(bufferPtr);
    }
  }

  int getBosToken() {
    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Must load model first');
    }
    return _tokenBos(_vocab!);
  }

  int getEosToken() {
    if (_vocab == null || _vocab!.address == 0) {
      throw StateError('Must load model first');
    }
    return _tokenEos(_vocab!);
  }

  String? generate(String prompt, {int maxTokens = 32}) {
    if (_context == null || _context!.address == 0) {
      print('Must create context first');
      return null;
    }

    if (_sampler == null || _sampler!.address == 0) {
      print('Sampler not initialized');
      return null;
    }

    print('\n=== Starting generation ===');
    print('Prompt: "$prompt"');

    // Tokenize the prompt
    final tokens = tokenizeText(prompt);
    if (tokens.isEmpty) {
      print('Failed to tokenize prompt');
      return null;
    }

    final nPrompt = tokens.length;
    print('Prompt tokens: $nPrompt');

    // Create batch for prompt using llama_batch_get_one
    final tokensPtr = malloc<Int32>(nPrompt);
    for (int i = 0; i < nPrompt; i++) {
      tokensPtr[i] = tokens[i];
    }

    try {
      var batch = _batchGetOne(tokensPtr, nPrompt);

      print('Decoding prompt batch...');
      if (_decode(_context!, batch) != 0) {
        print('Failed to decode prompt');
        return null;
      }
      print('✓ Prompt decoded');

      // Now generate tokens
      final result = StringBuffer();
      int nDecoded = 0;
      int nPos = nPrompt;

      final singleTokenPtr = malloc<Int32>();
      try {
        while (nDecoded < maxTokens && nPos < _nCtx(_context!)) {
          // Sample next token from the logits
          // -1 means use logits from last token in batch
          final newToken = _samplerSample(_sampler!, _context!, -1);

          // Check if end of generation
          if (_tokenIsEog(_vocab!, newToken)) {
            print('\n✓ Hit end-of-generation token');
            break;
          }

          // Detokenize and add to result
          final tokenText = detokenize(newToken);
          result.write(tokenText);
          print(tokenText); // Print as we generate

          // Prepare batch with single new token
          singleTokenPtr[0] = newToken;
          batch = _batchGetOne(singleTokenPtr, 1);

          // Decode the new token
          if (_decode(_context!, batch) != 0) {
            print('\nFailed to decode token $nDecoded');
            break;
          }

          nDecoded++;
          nPos++;
        }
      } finally {
        malloc.free(singleTokenPtr);
      }

      print('\n\n=== Generation complete ===');
      print('Generated $nDecoded tokens');

      return result.toString();
    } finally {
      malloc.free(tokensPtr);
    }
  }

  void dispose() {
    if (_sampler != null && _sampler!.address != 0) {
      _samplerFree(_sampler!);
      _sampler = null;
    }

    if (_context != null && _context!.address != 0) {
      _freeContext(_context!);
      _context = null;
    }

    if (_model != null && _model!.address != 0) {
      _freeModel(_model!);
      _model = null;
    }

    if (_initialized) {
      _backendFree();
      _initialized = false;
    }
  }
}
