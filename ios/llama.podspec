Pod::Spec.new do |s|
  s.name         = 'llama'
  s.version      = '0.1.1'
  s.summary      = 'Dart binding for llama.cpp'
  s.description  = 'High-level Dart / Flutter bindings for llama.cpp.'
  s.homepage     = 'https://github.com/varunbhalerao56/quiz_wrapper'
  s.authors      = { 'Varun Bhalerao' => 'varunbhalerao5902@gmail.com' }

  # 1. SOURCE PATH: Remains '.' as it's relative to the 'ios' directory.
  s.source       = { :path => '.' }

  # 2. PLATFORM: Set to '13.0' as requested (aligns with MIN_IOS_VERSION in your script).
  s.platform     = :ios, '13.0'


  s.source_files = []

  # 4. VENDOR FRAMEWORK: Corrected to use the path we determined previously.
  # The filename 'Llama.xcframework' is typically capitalized (as in your build scripts).
  s.vendored_frameworks = '../dist/Llama.xcframework'

  s.dependency 'Flutter'
end