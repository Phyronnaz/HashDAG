#!/bin/bash

error() {
	echo "Error: $*" >&2;
	exit 1;
}

# Ray tracing performance overheads:
python3.6 python/benchmark_rt.py epiccitadel 16 || error "RayTrace epiccitadel 16";
python3.6 python/benchmark_rt.py epiccitadel 17 || error "RayTrace epiccitadel 17";
python3.6 python/benchmark_rt.py sanmiguel 16 || error "RayTrace sanmiguel 16";

# Editing performance
python3.6 python/benchmark_edits.py epiccitadel 16 || error "Edits epiccitadel 16";
python3.6 python/benchmark_edits_small.py epiccitadel 16 || error "SmallEdits epiccitadel 16";
python3.6 python/benchmark_edits.py epiccitadel 17 || error "Edits epiccitadel 17";
python3.6 python/benchmark_edits_small.py epiccitadel 17 || error "SmallEdits epiccitadel 17";

# Wee!
echo "All done!"

#EOF
