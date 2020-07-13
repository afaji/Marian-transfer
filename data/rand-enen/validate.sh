#!/bin/bash
export LC_ALL=C
CPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS=$CPATH/../tools



cat $1 \
    | python3 $TOOLS/sacrebleu.py --score-only $CPATH/dev.trg

