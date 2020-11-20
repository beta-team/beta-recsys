
.PHONY: clean release

# Build  docker images.
# ARGS:
#   WHAT: cpu gpu.
# Examples:
#   make release WHAT=cpu
#
#   # compile all
#   make relase
release:
	hack/release-images.sh $(WHAT)

clean:
	-rm -Rf _output
