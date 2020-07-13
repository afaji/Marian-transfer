#!/bin/bash
export LC_ALL=C
CPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS=$CPATH/../tools



cat $1 \
    | sed -r 's/(@@ )|(@@ ?$)//g' 2>/dev/null \
    | $TOOLS/detruecase.perl 2>/dev/null \
    | $TOOLS/detokenizer.perl -l en 2>/dev/null \
    | python3 $TOOLS/sacrebleu.py --score-only $CPATH/dev.trg

