import 'package:flutter/material.dart';
import 'package:quiz_wrapper/home.dart';
import 'package:quiz_wrapper/src/isolates/llama_isolate_parent.dart';
import 'package:quiz_wrapper/src/utils/llama_config.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma Chat',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple), useMaterial3: true),
      home: SafeArea(child: const HomeScreen()),
    );
  }
}
