# Lightstorm (working name)

Simplified version of Firestorm targeting C instead of machine code directly.

## Local Build Setup

### Install Dependencies

You need `ninja`, `cmake` (at least 3.28) and `llvm` 19.

On macOS:

```bash
brew install ninja cmake hyperfine llvm@19
```

On Ubuntu 24.04:

```bash
sudo apt-get install ninja-build cmake
```

To install LLVM 19 follow the instructions [here](https://apt.llvm.org).

### Checkout

```bash
git clone git@github.com:DragonRuby/lightstorm.git --recursive
```

### Build & install

```bash
# On Ubuntu
cmake --workflow --preset lightstorm-ubuntu-install
# On macOS
cmake --workflow --preset lightstorm-macos-install
```

## Build "Hello World"

```bash
> echo 'puts "Hello, Lightstorm"' > hello.rb
> ./install.dir/bin/lightstorm hello.rb -o hello.rb.c
> clang hello.rb.c -o hello_lightstorm \
  -L./install.dir/lib/ \
  -isystem./third_party/mruby/include -isystem./third_party/mruby/build/host/include/ \
   -llightstorm_runtime_main -llightstorm_mruby -lm
> ./hello_lightstorm
Hello, Lightstorm
```

## Build and run tests

Build a test (`tests/integration/loads.rb`):

```bash
> cd build.dir
> ninja loads.rb.exe
> ../lightstorm/tests/integration/Output/loads.rb.tmp.exe
1
-1
42
-42
1000
-1000
1000000
-1000000

42.0
a string 42
true
false
main
hello
[:a_sym]
```

Run integration tests

```bash
pip install lit filecheck
ninja run-integration-tests
```

Run benchmarks

```bash
ninja run-benchmarks
```
