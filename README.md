# Lightstorm (working name)

Simplified version of Firestorm targeting C instead of machine code directly.

# Local Build Setup

Brew:

```bash
brew install ninja cmake ccache hyperfine llvm@19
```

Get the sources

```bash
git clone git@github.com:DragonRuby/lightstorm.git --recursive
```

Create toolchain dir

```bash
sudo mkdir /opt/lightstorm.toolchain.dir
sudo chown `whoami` /opt/lightstorm.toolchain.dir
```

Build lightstorm

```bash
mkdir lightstorm-build; cd lightstorm-build
cmake -G Ninja -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm@19/ \
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
