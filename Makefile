CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
BUILD    := build

SRC := src/memory_guard.cpp
HDR := include/aero/memory_guard.hpp

# Native compute kernels. Separate from the guard so the guard's tests do not
# drag in the kernel TU and vice versa.
KSRC := src/kernels.cpp
KHDR := include/aero/kernels.hpp
KLIB := $(BUILD)/libaero_kernels.dylib
KFLAGS := -O2 -ffast-math -fno-math-errno -pthread

PY  := .venv-cml/bin/python
PYTOOLS := $(wildcard tools/*.py)
PYLIB   := $(wildcard engine/*.py)
PYTESTS := $(wildcard tests/test_*.py)

.PHONY: all test test-cpp test-engine test-server test-wrappers demo probe \
        check-py kernels guard engine-cpp clean

all: $(BUILD)/test_memory_guard $(BUILD)/test_metal_guard $(BUILD)/aero-load-demo $(BUILD)/aero-bwprobe $(BUILD)/test_kernels $(BUILD)/test_tensor $(BUILD)/test_gguf $(KLIB)

# ---- native kernels ----
# The test binary links the kernel TU directly, so `make test-cpp` proves the
# C++ is right independently of whether Python can load the dylib.
$(BUILD)/test_kernels: $(KSRC) tests/test_kernels.cpp $(KHDR)
	@mkdir -p $(BUILD)
	$(CXX) -std=c++17 $(CXXFLAGS:%Wall=%Wno-unused-parameter) $(KSRC) tests/test_kernels.cpp -o $@ $(KFLAGS)

$(KLIB): $(KSRC) $(KHDR)
	@mkdir -p $(BUILD)
	$(CXX) -std=c++17 $(CXXFLAGS:%Wall=%Wno-unused-parameter) -fPIC -shared $(KSRC) -o $@ $(KFLAGS)

kernels: $(KLIB)
	@echo "native kernels: $(KLIB)"

# ---- the C++ engine modules (tensor primitives first; the layer loop follows) --
TSRC := src/tensor.cpp
THDR := include/aero/tensor.hpp

$(BUILD)/test_tensor: $(TSRC) tests/test_tensor.cpp $(THDR)
	@mkdir -p $(BUILD)
	$(CXX) -std=c++17 $(CXXFLAGS) $(TSRC) tests/test_tensor.cpp -o $@ -pthread

# gguf reader. Needs kernels.cpp too: it dequantises and matvecs through the fused
# kernels, and the test proves that path works on a real file.
GSRC := src/gguf.cpp
GHDR := include/aero/gguf.hpp
MODEL_DIR ?= $(HOME)/.cache/aero-models
REAL_MODEL ?= $(MODEL_DIR)/Qwen3.8-4B-Q4_K_M.gguf

$(BUILD)/test_gguf: $(GSRC) $(KSRC) tests/test_gguf_cpp.cpp $(GHDR) $(KHDR)
	@mkdir -p $(BUILD)
	$(CXX) -std=c++17 $(CXXFLAGS) $(GSRC) $(KSRC) tests/test_gguf_cpp.cpp -o $@ -pthread

engine-cpp: $(BUILD)/test_tensor $(BUILD)/test_gguf
	@./$(BUILD)/test_tensor
	@echo
	@if [ -f "$(REAL_MODEL)" ]; then \
		AERO_MAX_RAM=$${AERO_MAX_RAM:-5G} ./$(BUILD)/test_gguf "$(REAL_MODEL)"; \
	 else echo "gguf test: SKIPPED (no $(REAL_MODEL))"; fi

# The memory guard as a shared library, so Python arms the SAME guard the C++
# tests exercise. A reimplementation in Python would be a second thing to keep
# correct; a shared library cannot drift.
GLIB := $(BUILD)/libaero_guard.dylib

$(GLIB): $(SRC) $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC -shared $(SRC) -o $@

guard: $(GLIB)
	@echo "memory guard: $(GLIB)"

all: $(GLIB)
kernels-all: $(KLIB) $(GLIB)


$(BUILD)/test_memory_guard: $(SRC) tests/test_memory_guard.cpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC) tests/test_memory_guard.cpp -o $@

# Metal test: proves phys_footprint sees GPU-only memory and rss does not.
$(BUILD)/test_metal_guard: $(SRC) tests/test_metal_guard.mm $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -fobjc-arc -x objective-c++ $(SRC) tests/test_metal_guard.mm \
		-o $@ -framework Foundation -framework Metal

$(BUILD)/aero-load-demo: $(SRC) tools/aero-load-demo.cpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC) tools/aero-load-demo.cpp -o $@

$(BUILD)/aero-bwprobe: $(SRC) tools/aero-bwprobe.mm $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -fobjc-arc -x objective-c++ $(SRC) tools/aero-bwprobe.mm \
		-o $@ -framework Foundation -framework Metal

# Syntax-check every Python source (engine, tools, tests).
check-py:
	@if [ -x "$(PY)" ]; then \
		$(PY) -m py_compile $(PYLIB) $(PYTOOLS) $(PYTESTS) \
			&& echo "py_compile: ok"; \
	else \
		echo "py_compile: SKIPPED (no $(PY); create the venv to include Python in the gate)"; \
	fi

test-cpp: all
	@./$(BUILD)/test_memory_guard
	@echo
	@./$(BUILD)/test_metal_guard
	@echo
	@./$(BUILD)/test_kernels
	@echo
	@./$(BUILD)/test_tensor
	@echo
	@if [ -f "$(REAL_MODEL)" ]; then \
		AERO_MAX_RAM=$${AERO_MAX_RAM:-5G} ./$(BUILD)/test_gguf "$(REAL_MODEL)"; \
	 else echo "gguf test: SKIPPED (no $(REAL_MODEL))"; fi

test-engine: check-py
	@if [ -x "$(PY)" ]; then $(PY) tests/test_engine.py; \
	 else echo "engine test: SKIPPED (no $(PY))"; fi

test-server: check-py
	@if [ -x "$(PY)" ]; then $(PY) tests/test_server.py; \
	 else echo "server test: SKIPPED (no $(PY))"; fi

# Model-side suites: GGUF parse + dequant, transformer core, SSM, optimisations.
test-model: check-py
	@if [ -x "$(PY)" ]; then \
		$(PY) tests/test_quant.py; echo; \
		$(PY) tests/test_transformer.py; echo; \
		$(PY) -W error::RuntimeWarning tests/test_optimizations.py; echo; \
		$(PY) -W error::RuntimeWarning tests/test_ssm.py; echo; \
		$(PY) -W error::RuntimeWarning tests/test_model.py; echo; \
		AERO_SKIP_SLOW=1 $(PY) -W error::RuntimeWarning tests/test_kv.py 2>&1 | tail -25; echo; \
		AERO_SKIP_SLOW=1 $(PY) -W error::RuntimeWarning tests/test_prerope.py 2>&1 | tail -20; echo; \
		$(PY) -W error::RuntimeWarning tests/test_tokenizer.py 2>&1 | tail -30; echo; \
		$(PY) -W error::RuntimeWarning tests/test_native.py 2>&1 | tail -25; echo; \
		AERO_SKIP_SLOW=1 $(PY) -W error::RuntimeWarning tests/test_coherence.py 2>&1 | tail -20; \
		$(PY) tests/test_arch.py 2>&1 | tail -30; echo; \
		AERO_SKIP_SLOW=1 $(PY) tests/test_runtime_gguf.py 2>&1 | tail -20; \
	 else echo "model tests: SKIPPED (no $(PY))"; fi

# Entry-point smoke test. The unit suites import the engine as a library; they
# never execute the wrappers. That gap is exactly how a completely broken
# aero-serve (relative-import failure when run as a script) passed 118 checks.
# Invoke every entry point the way a user does.
test-wrappers: all check-py
	@./aero --limits > /dev/null && echo "aero       : ok"
	@./aero-chat --limits > /dev/null && echo "aero-chat  : ok"
	@./aero-serve --limits > /dev/null && echo "aero-serve : ok"
	@printf '/exit\n' | ./aero-chat --runtime echo --model smoke > /dev/null \
		&& echo "aero-chat  : REPL loop ok"
	@printf 'hi\n/exit\n' | ./aero-chat --runtime echo --model smoke \
		| grep -q "echo:smoke" && echo "aero-chat  : answered ok"
	@printf 'hi\n/exit\n' | ./aero-chat --runtime echo --model smoke 2>$(BUILD)/pipe.err \
		| head -2 > /dev/null; \
		if grep -q BrokenPipe $(BUILD)/pipe.err; then \
			echo "aero-chat  : BrokenPipeError on early pipe close -- FAIL"; exit 1; \
		else echo "aero-chat  : clean close when the pipe closes early"; fi
	@./aero-serve --runtime gguf --model nope.gguf --limits > /dev/null \
		&& echo "aero-serve : limits path ok"

# Canonical gate.
test: test-cpp test-engine test-server test-wrappers test-model

demo: $(BUILD)/aero-load-demo

probe: $(BUILD)/aero-bwprobe

clean:
	@rm -rf $(BUILD)
