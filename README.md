# Lightstorm (working name)

Simplified version of Firestorm targeting C instead of machine code directly.

# Local Build Setup

Get the sources

```bash
git clone git@github.com:DragonRuby/lightstorm.git --recursive
git clone git@github.com:llvm/llvm-project.git -b release/17.x
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
cd ..
```

Build lightstorm

```bash
mkdir lightstorm-build; cd lightstorm-build
cmake -G Ninja -DCMAKE_PREFIX_PATH=/opt/lightstorm.toolchain.dir/llvm \
  -DCMAKE_INSTALL_PREFIX=/opt/lightstorm.toolchain.dir/lightstorm \
  ../lightstorm
```

Build test:

```bash
> ninja 00.rb.exe
> ./tests/end2end/00.rb.exe
42

```
