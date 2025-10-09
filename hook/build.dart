import 'dart:io';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as path;

const teamId = 'VF63K2KVXG';

void main(List<String> args) async {
  await build(args, (input, output) async {
    // Only proceed if we should build code assets
    if (!input.config.buildCodeAssets) {
      return;
    }

    final packageRoot = input.packageRoot.toFilePath();
    final llamaCppDir = path.join(packageRoot, 'third_party', 'llama.cpp');

    // Add llama.cpp as a dependency
    output.dependencies.add(input.packageRoot.resolve('third_party/llama.cpp/'));

    final binDir = path.join(packageRoot, 'bin', 'macos-arm64');

    // Check if libraries already exist
    final libllama = File(path.join(binDir, 'libllama.dylib'));

    if (!libllama.existsSync()) {
      final buildScript = path.join(packageRoot, 'scripts', 'build_macos.sh');

      // Make script executable
      await Process.run('chmod', ['+x', buildScript]);

      // Run build script
      final result = await Process.run('bash', [buildScript, llamaCppDir, teamId], workingDirectory: packageRoot);

      if (result.exitCode != 0) {
        throw Exception('Build failed: ${result.stderr}');
      }
    }

    // Add built libraries as code assets
    final libraries = [
      'libllama.dylib',
      'libggml.dylib',
      'libggml-base.dylib',
      'libggml-metal.dylib',
      'libggml-cpu.dylib',
      'libggml-blas.dylib',
      'libmtmd.dylib',
    ];

    for (final lib in libraries) {
      final libPath = path.join(binDir, lib);
      if (File(libPath).existsSync()) {
        output.assets.code.add(
          CodeAsset(package: input.packageName, name: lib, file: Uri.file(libPath), linkMode: DynamicLoadingBundled()),
        );
      }
    }
  });
}
