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
# Transfer Learning from Anther Language

We start by downloading the parent model. In this case, we will use the German to English model

```
wget data.statmt.org/alham/pretrain/pretrain-deen.tar.gz
tar -xf pretrain-deen.tar.gz
```

We will transfer the parent with token-matching method. Start by preparing the child directory and constructing the vocabulary:
```
mkdir iden-trans-deen
cat data/iden/train.bpe.* | ../marian-dev/build/marian-vocab > iden-trans-deen/vocab.yml
```

Then, transfer the model from the parent to the child with the following:

```
python2 transfer_model.py pretrain-deen/ iden-trans-deen/

```

We're done. The last step is to train our Indonesian to English model:
```
./train.sh iden-trans-rand/ data/iden -l 0.0003 --optimizer-delay 2
```

We should expect significant BLEU improvement over the baseline.

# Transfer Learning from Substitution English Corpus

We can prepare our parent model by ourself. In this case, we will show that the transfer learning can be done even with a monolingual data.
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
./train.sh iden-trans-enen/ data/iden -l 0.0003 --optimizer-delay 2
```

We should expect a BLEU improvement over the baseline, but not as high as transferring from another language pair.


# Transfer Learning from Substitution Random Corpus

We showed that transfer learning can be implemented from a model that maps random sequences. To demonstrate this, simply follow the previous step, but use `data/rand-sub` when preparing the parent model.

```
mkdir parent
./train.sh parent data/rand-sub -l 0.0003 --after-epochs 45
```
