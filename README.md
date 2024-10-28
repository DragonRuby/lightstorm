# Lightstorm (working name)

Simplified version of Firestorm targeting C instead of machine code directly.

# Local Build Setup

Brew:
```
brew install ninja cmake ccache hyperfine
```

Get the sources

```bash
git clone git@github.com:DragonRuby/lightstorm.git --recursive
git clone git@github.com:llvm/llvm-project.git; cd llvm-project; git checkout d401987fe349a87c53fe25829215b080b70c0c1a
```

Create toolchain dir

```bash
sudo mkdir /opt/lightstorm.toolchain.dir
sudo chown `whoami` /opt/lightstorm.toolchain.dir
```

Build and install LLVM (+clang +MLIR):

```bash
mkdir lightstorm-llvm; cd lightstorm-llvm
cmake -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_INSTALL_PREFIX=/opt/lightstorm.toolchain.dir/llvm \
  ../llvm-project/llvm

ninja
ninja install

cd ..
```

Build lightstorm

```bash
mkdir lightstorm-build; cd lightstorm-build
cmake -G Ninja -DCMAKE_PREFIX_PATH=/opt/lightstorm.toolchain.dir/llvm \
  -DCMAKE_INSTALL_PREFIX=/opt/lightstorm.toolchain.dir/lightstorm \
  ../lightstorm

ninja
```

Build a test (`tests/integration/loads.rb`):

```bash
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
