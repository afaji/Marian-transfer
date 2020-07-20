# Marian-transfer
Transfer learning experiment demo with Marian to replicate some of the experiments in https://www.aclweb.org/anthology/2020.acl-main.688/

# Requirement
Compile Marian in this repo:

```
cd marian
mkdir build
cd build
cmake ..
make -j
cd ../..
```

# Train the Baseline

We first train a baseline Indonesian-to-English NMT system with a standard Transformer by the following:

```
mkdir iden-base
./train.sh iden-base data/iden -l 0.0003 --optimizer-delay 2
```

After a few hours on a single GPU, we should expect the model to achieve ~19 BLEU points.

# Transfer Learning from Substitution English Corpus

Start training a random English-English substitution model:

```
mkdir parent
./train.sh parent data/rand-enen -l 0.0003 --after-epochs 80
```

Then, we use that model to initate our Indonesian-to-English MT system.
We start by building the Indonesian-English vocabulary, since we need to match the embedding tokens while performing the transfer learning:

```
mkdir iden-trans-enen
cat data/iden/train.bpe.* | ../marian-dev/build/marian-vocab > iden-trans-enen/vocab.yml
python2 transfer_model.py parent/ iden-trans-rand/
```

Lastly, simply train a new Indonesian-to-English NMT from the transferred model:

```
./train.sh iden-trans-rand/ data/iden -l 0.0003 --optimizer-delay 2
```

We should expect significant BLEU improvement over the baseline.


# Transfer Learning from Substitution Random Corpus

We showed that transfer learning can be implemented from a model that maps random sequences. To demonstrate this, simply follow the previous step, but use `data/rand-sub` when preparing the parent model.

```
mkdir parent
./train.sh parent data/rand-sub -l 0.0003 --after-epochs 45
```
