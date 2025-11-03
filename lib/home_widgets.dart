// part of 'home.dart';

// // ==================== Reusable Widgets ====================

// class InitializationButtons extends StatelessWidget {
//   final bool isProcessing;
//   final VoidCallback onStepByStep;
//   final VoidCallback onInitAll;

//   const InitializationButtons({
//     super.key,
//     required this.isProcessing,
//     required this.onStepByStep,
//     required this.onInitAll,
//   });

//   @override
//   Widget build(BuildContext context) {
//     return Row(
//       children: [
//         Expanded(
//           child: ElevatedButton.icon(
//             onPressed: isProcessing ? null : onStepByStep,
//             icon: const Icon(Icons.stairs),
//             label: const Text('Init Step by Step'),
//             style: ElevatedButton.styleFrom(
//               backgroundColor: Colors.blue,
//               foregroundColor: Colors.white,
//               padding: const EdgeInsets.all(16),
//             ),
//           ),
//         ),
//         const SizedBox(width: 8),
//         Expanded(
//           child: ElevatedButton.icon(
//             onPressed: isProcessing ? null : onInitAll,
//             icon: const Icon(Icons.flash_on),
//             label: const Text('Init All'),
//             style: ElevatedButton.styleFrom(
//               backgroundColor: Colors.green,
//               foregroundColor: Colors.white,
//               padding: const EdgeInsets.all(16),
//             ),
//           ),
//         ),
//       ],
//     );
//   }
// }

// class StepByStepCard extends StatelessWidget {
//   final bool isolateStarted;
//   final bool backendInit;
//   final bool modelLoaded;
//   final bool contextCreated;
//   final bool isProcessing;
//   final VoidCallback onStartIsolate;
//   final VoidCallback onInitBackend;
//   final VoidCallback onLoadModel;
//   final VoidCallback onCreateContext;

//   const StepByStepCard({
//     super.key,
//     required this.isolateStarted,
//     required this.backendInit,
//     required this.modelLoaded,
//     required this.contextCreated,
//     required this.isProcessing,
//     required this.onStartIsolate,
//     required this.onInitBackend,
//     required this.onLoadModel,
//     required this.onCreateContext,
//   });

//   @override
//   Widget build(BuildContext context) {
//     return Card(
//       elevation: 2,
//       child: Padding(
//         padding: const EdgeInsets.all(16),
//         child: Column(
//           crossAxisAlignment: CrossAxisAlignment.stretch,
//           children: [
//             const Text(
//               'Manual Initialization',
//               style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
//               textAlign: TextAlign.center,
//             ),
//             const SizedBox(height: 16),

//             InitializationStep(
//               stepNumber: 0,
//               title: 'Start Isolate',
//               completed: isolateStarted,
//               onPressed: !isolateStarted && !isProcessing ? onStartIsolate : null,
//             ),
//             const SizedBox(height: 8),

//             InitializationStep(
//               stepNumber: 1,
//               title: 'Initialize Backend',
//               completed: backendInit,
//               enabled: isolateStarted,
//               onPressed: isolateStarted && !backendInit && !isProcessing ? onInitBackend : null,
//             ),
//             const SizedBox(height: 8),

//             InitializationStep(
//               stepNumber: 2,
//               title: 'Load Model',
//               completed: modelLoaded,
//               enabled: backendInit,
//               onPressed: backendInit && !modelLoaded && !isProcessing ? onLoadModel : null,
//             ),
//             const SizedBox(height: 8),

//             InitializationStep(
//               stepNumber: 3,
//               title: 'Create Context',
//               completed: contextCreated,
//               enabled: modelLoaded,
//               onPressed: modelLoaded && !contextCreated && !isProcessing ? onCreateContext : null,
//             ),

//             if (isProcessing) ...[const SizedBox(height: 16), const Center(child: CircularProgressIndicator())],
//           ],
//         ),
//       ),
//     );
//   }
// }

// class InitializationStep extends StatelessWidget {
//   final int stepNumber;
//   final String title;
//   final bool completed;
//   final bool enabled;
//   final VoidCallback? onPressed;

//   const InitializationStep({
//     super.key,
//     required this.stepNumber,
//     required this.title,
//     required this.completed,
//     this.enabled = true,
//     this.onPressed,
//   });

//   @override
//   Widget build(BuildContext context) {
//     return Container(
//       decoration: BoxDecoration(
//         color: completed ? Colors.green[50] : (enabled ? Colors.white : Colors.grey[100]),
//         borderRadius: BorderRadius.circular(8),
//         border: Border.all(color: completed ? Colors.green : (enabled ? Colors.blue : Colors.grey[300]!), width: 2),
//       ),
//       child: ListTile(
//         leading: CircleAvatar(
//           backgroundColor: completed ? Colors.green : (enabled ? Colors.blue : Colors.grey),
//           child: completed
//               ? const Icon(Icons.check, color: Colors.white)
//               : Text('$stepNumber', style: const TextStyle(color: Colors.white)),
//         ),
//         title: Text(
//           title,
//           style: TextStyle(fontWeight: FontWeight.bold, color: enabled ? Colors.black : Colors.grey),
//         ),
//         trailing: onPressed != null
//             ? ElevatedButton(
//                 onPressed: onPressed,
//                 style: ElevatedButton.styleFrom(backgroundColor: Colors.blue, foregroundColor: Colors.white),
//                 child: const Text('Run'),
//               )
//             : completed
//             ? const Icon(Icons.check_circle, color: Colors.green)
//             : null,
//       ),
//     );
//   }
// }

// class GenerationControls extends StatelessWidget {
//   final GenerationMode selectedMode;
//   final bool useStreaming;
//   final bool isGenerating;
//   final TextEditingController promptController;
//   final Function(GenerationMode) onModeChanged;
//   final Function(bool) onStreamingChanged;
//   final VoidCallback onGenerate;

//   const GenerationControls({
//     super.key,
//     required this.selectedMode,
//     required this.useStreaming,
//     required this.isGenerating,
//     required this.promptController,
//     required this.onModeChanged,
//     required this.onStreamingChanged,
//     required this.onGenerate,
//   });

//   @override
//   Widget build(BuildContext context) {
//     return Column(
//       crossAxisAlignment: CrossAxisAlignment.stretch,
//       children: [
//         // Mode selection
//         const Text('Generation Mode', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
//         const SizedBox(height: 8),
//         ModeSelector(selectedMode: selectedMode, onModeChanged: onModeChanged),
//         const SizedBox(height: 16),

//         // Streaming toggle
//         SwitchListTile(
//           title: const Text('Use Streaming'),
//           subtitle: const Text('Generate tokens in real-time'),
//           value: useStreaming,
//           onChanged: onStreamingChanged,
//         ),
//         const SizedBox(height: 16),

//         // Prompt input
//         TextField(
//           controller: promptController,
//           decoration: const InputDecoration(
//             labelText: 'Prompt',
//             hintText: 'Enter your prompt here...',
//             border: OutlineInputBorder(),
//           ),
//           maxLines: 8,
//           enabled: !isGenerating,
//         ),
//         const SizedBox(height: 16),

//         // Submit button
//         GenerateButton(isGenerating: isGenerating, onPressed: onGenerate),
//       ],
//     );
//   }
// }

// class ModeSelector extends StatelessWidget {
//   final GenerationMode selectedMode;
//   final Function(GenerationMode) onModeChanged;

//   const ModeSelector({super.key, required this.selectedMode, required this.onModeChanged});

//   @override
//   Widget build(BuildContext context) {
//     return SegmentedButton<GenerationMode>(
//       segments: const [
//         ButtonSegment(value: GenerationMode.json, label: Text('JSON'), icon: Icon(Icons.code)),
//         ButtonSegment(value: GenerationMode.precise, label: Text('Precise'), icon: Icon(Icons.gps_fixed)),
//         ButtonSegment(value: GenerationMode.creative, label: Text('Creative'), icon: Icon(Icons.auto_awesome)),
//         ButtonSegment(value: GenerationMode.balanced, label: Text('Balanced'), icon: Icon(Icons.balance)),
//       ],
//       selected: {selectedMode},
//       onSelectionChanged: (Set<GenerationMode> newSelection) {
//         onModeChanged(newSelection.first);
//       },
//     );
//   }
// }

// class GenerateButton extends StatelessWidget {
//   final bool isGenerating;
//   final VoidCallback onPressed;

//   const GenerateButton({super.key, required this.isGenerating, required this.onPressed});

//   @override
//   Widget build(BuildContext context) {
//     return ElevatedButton.icon(
//       onPressed: isGenerating ? null : onPressed,
//       icon: isGenerating
//           ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
//           : const Icon(Icons.send),
//       label: Text(isGenerating ? 'Generating...' : 'Generate'),
//       style: ElevatedButton.styleFrom(
//         backgroundColor: Colors.deepPurple,
//         foregroundColor: Colors.white,
//         padding: const EdgeInsets.all(16),
//       ),
//     );
//   }
// }

// class ResponseDisplay extends StatelessWidget {
//   final String response;

//   const ResponseDisplay({super.key, required this.response});

//   @override
//   Widget build(BuildContext context) {
//     return Column(
//       crossAxisAlignment: CrossAxisAlignment.stretch,
//       children: [
//         const Text('Output', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
//         const SizedBox(height: 8),
//         Container(
//           width: double.infinity,
//           constraints: const BoxConstraints(minHeight: 200),
//           padding: const EdgeInsets.all(16),
//           decoration: BoxDecoration(
//             color: Colors.grey[100],
//             borderRadius: BorderRadius.circular(8),
//             border: Border.all(color: Colors.grey[300]!),
//           ),
//           child: SelectableText(response, style: const TextStyle(fontSize: 14, fontFamily: 'monospace')),
//         ),
//       ],
//     );
//   }
// }
