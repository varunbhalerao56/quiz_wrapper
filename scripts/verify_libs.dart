import 'dart:io';
import 'dart:ffi';

import 'package:flutter/foundation.dart';

void main() {
  debugPrint('Verifying llama.cpp libraries...\n');

  final libraries = [
    'bin/macos-arm64/libllama.dylib',
    'bin/macos-arm64/libggml.dylib',
    'bin/macos-arm64/libggml-base.dylib',
    'bin/macos-arm64/libggml-metal.dylib',
    'bin/macos-arm64/libggml-cpu.dylib',
    'bin/macos-arm64/libmtmd.dylib',
  ];

  var allGood = true;

  for (final libPath in libraries) {
    final exists = File(libPath).existsSync();
    if (!exists) {
      debugPrint('✗ $libPath - NOT FOUND');
      allGood = false;
      continue;
    }

    try {
      DynamicLibrary.open(libPath);
      final size = File(libPath).lengthSync();
      final sizeMB = (size / (1024 * 1024)).toStringAsFixed(2);
      debugPrint('✓ $libPath - LOADABLE ($sizeMB MB)');
    } catch (e) {
      debugPrint('✗ $libPath - CANNOT LOAD: $e');
      allGood = false;
    }
  }

  debugPrint('\n${allGood ? "SUCCESS!" : "FAILED"}');
  exit(allGood ? 0 : 1);
}
