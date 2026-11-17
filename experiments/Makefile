HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
# Optional: specify target arch, e.g., make ARCH=--offload-arch=gfx906
ARCH ?=

TARGETS = 01_device_info_and_isa \
          02_coalescing_and_vec \
          03_lds_bank_conflict \
          04_branch_divergence \
          05_occupancy_and_ilp \
          06_shuffle_vs_lds

.PHONY: all clean isa run

all: $(TARGETS)

%: %.hip common.h
	$(HIPCC) $(CXXFLAGS) $(ARCH) $< -o $@

isa:
	mkdir -p isa_dump
	$(HIPCC) $(CXXFLAGS) $(ARCH) -save-temps 01_device_info_and_isa.hip -o isa_dump/01_test
	$(HIPCC) $(CXXFLAGS) $(ARCH) -save-temps 04_branch_divergence.hip -o isa_dump/04_test
	$(HIPCC) $(CXXFLAGS) $(ARCH) -save-temps 06_shuffle_vs_lds.hip -o isa_dump/06_test
	mv -f *.s *.ii *.bc *.o *.hipfb *.out 2>/dev/null isa_dump/ || true
	@echo "ISA assembly (.s) dumped into isa_dump/"

run: all
	@for t in $(TARGETS); do \
		echo "----------------------------------------"; \
		./$$t; \
	done

clean:
	rm -rf $(TARGETS) isa_dump *.s *.ii *.bc *.o *.hipfb
