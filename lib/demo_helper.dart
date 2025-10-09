/// Simple state management for test steps
class TestState {
  final String status;
  final bool isLoading;
  final String? error;

  const TestState({required this.status, this.isLoading = false, this.error});

  factory TestState.initial(String status) {
    return TestState(status: status, isLoading: false);
  }

  TestState loading(String message) {
    return TestState(status: message, isLoading: true);
  }

  TestState success(String message) {
    return TestState(status: message, isLoading: false);
  }

  TestState failure(String message) {
    return TestState(status: message, isLoading: false, error: message);
  }
}
