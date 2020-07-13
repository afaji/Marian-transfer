MODEL=$1
DATA=$2

shift 2

MARIAN=marian

$MARIAN/build/marian \
        --model $MODEL/model.npz --type transformer \
        --train-sets $DATA/train.bpe.src $DATA/train.bpe.trg \
        --max-length 100 \
        --vocabs $MODEL/vocab.yml $MODEL/vocab.yml \
        --mini-batch-fit -w 8000 --maxi-batch 1000 \
        --early-stopping 10 --cost-type=ce-mean-words \
        --valid-freq 250 --save-freq 250 --disp-freq 50 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-script-path $DATA/validate.sh \
        --valid-sets $DATA/dev.bpe.src $DATA/dev.bpe.trg \
        --overwrite --quiet-translation \
        --valid-mini-batch 8 \
        --beam-size 6 --normalize 0.6 \
        --log $MODEL/train.log --valid-log $MODEL/valid.log \
        --enc-depth 6 --dec-depth 6 \
        --transformer-heads 8 \
        --transformer-postprocess-emb d \
        --transformer-postprocess dan \
        --transformer-dropout 0.1 --label-smoothing 0.1 \
        --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
        --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
        --tied-embeddings-all --sync-sgd \
        --keep-best --seed 1111 --shuffle-in-ram \
        --exponential-smoothing -d 0 $@

