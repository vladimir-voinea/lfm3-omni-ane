#!/bin/sh
# Every Core ML load leaves a compiled .mlmodelc in the system temp dir; a few hundred
# loads is tens of GB. These are caches: Core ML recompiles them on demand.
TMPDIR_REAL=$(getconf DARWIN_USER_TEMP_DIR)
before=$(df -h / | awk 'NR==2 {print $4}')
find "$TMPDIR_REAL" -maxdepth 1 -name '*.mlmodelc' -type d -exec rm -rf {} + 2>/dev/null
rm -rf "$TMPDIR_REAL"/*.mlpackage 2>/dev/null
after=$(df -h / | awk 'NR==2 {print $4}')
echo "freed: $before -> $after"