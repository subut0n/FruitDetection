#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...
cd ./app 

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate fruits

# Re-enable strict mode:
set -euo pipefail