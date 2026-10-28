#!/bin/bash
# fetch a large HF file with several range requests in parallel; single streams here
# appear to be capped per connection
URL="$1"; OUT="$2"; PARTS=${3:-8}
SIZE=$(curl -sIL "$URL" | tr -d '\r' | awk '/^content-length:/ {print $2}' | tail -1)
CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
rm -rf /tmp/parts; mkdir -p /tmp/parts
for i in $(seq 0 $((PARTS-1))); do
  START=$(( i * CHUNK ))
  END=$(( START + CHUNK - 1 ))
  if [ $END -ge $SIZE ]; then END=$(( SIZE - 1 )); fi
  curl -sL -r $START-$END -o /tmp/parts/part_$i "$URL" &
done
wait
cat $(ls /tmp/parts/part_* | sort -t_ -k2 -n) > "$OUT"
ls -la "$OUT" | awk '{print $5" bytes"}'