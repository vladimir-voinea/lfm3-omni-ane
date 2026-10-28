#!/bin/bash
# resume each range part from its current size
URL="$1"; DIR="$2"; PARTS=${3:-8}
SIZE=$(curl -sIL "$URL" | tr -d '\r' | awk '/^content-length:/ {print $2}' | tail -1)
CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
for i in $(seq 0 $((PARTS-1))); do
  FILE=$DIR/part_$i
  HAVE=$(stat -f%z "$FILE" 2>/dev/null || echo 0)
  START=$(( i * CHUNK + HAVE ))
  END=$(( (i+1) * CHUNK - 1 ))
  if [ $END -ge $SIZE ]; then END=$(( SIZE - 1 )); fi
  if [ $START -ge $END ]; then continue; fi
  curl -sL -r $START-$END "$URL" >> "$FILE" &
  echo "part $i resuming from $HAVE"
done
wait
cat $(ls $DIR/part_* | sort -t_ -k2 -n) > "$DIR/model.safetensors"
ls -la "$DIR/model.safetensors" | awk '{print $5" bytes"}'