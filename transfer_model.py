import numpy as np
import sys
from sys import argv

dir_parent = argv[1]
dir_child = argv[2]


# re-arrange the vocabulary, such that we transfer the the equivalent tokens properly
d_child = []
for l in open(dir_child + "/vocab.yml", "r"):
  w = l.strip().split()[0][:-1]
  d_child.append(w)

vocab_size = len(d_child)

child_new_vocab = open(dir_child + "/vocab.yml", "w")
new_vocab = []
for l in open(dir_parent + "/vocab.yml", "r"):
  w = l.strip().split()[0][:-1]
  if (w in d_child):
    new_vocab.append(w)
    d_child.remove(w)
  else:
    new_vocab.append(None)
  if (len(new_vocab) == vocab_size):
    break

for cnt in range(vocab_size):
  if (len(new_vocab) > cnt) and (new_vocab[cnt] is not None):
    w = new_vocab[cnt]
  else:
    w = d_child[0]
    d_child.remove(w)
  child_new_vocab.write(w + ": "+str(cnt)+"\n")
  


print("RESIZING TO ", vocab_size)

#if parent has less vocab size, double it first
old_model = np.load(dir_parent + "/model.npz.best-translation.npz")
new_model = dict(old_model)

old_size = len(old_model["Wemb"])
new_size = vocab_size

while (new_size > len(new_model["Wemb"])):
    new_model["decoder_ff_logit_out_b"] = np.concatenate((new_model["decoder_ff_logit_out_b"], new_model["decoder_ff_logit_out_b"]), axis=1)
    new_model["Wemb"] = np.concatenate((new_model["Wemb"], new_model["Wemb"]))

# resize the parent's embedding size to match the child's vocab size
print("Before: ", new_model["decoder_ff_logit_out_b"].shape, new_model["Wemb"].shape)
new_model["decoder_ff_logit_out_b"] = new_model["decoder_ff_logit_out_b"][:,:new_size]
new_model["Wemb"] = new_model["Wemb"][:new_size]

print("After: ", new_model["decoder_ff_logit_out_b"].shape, new_model["Wemb"].shape)

# replace the vocab size in yml comfiguration
print("Old yml: ", new_model["special:model.yml"].tostring())

tmp = new_model["special:model.yml"].tostring().decode("utf-8") 
tmp = tmp.replace(str(old_size), str(new_size))
new_model["special:model.yml"] = np.array(bytearray(tmp, 'utf-8'))

print("New yml: ", new_model["special:model.yml"].tostring())

np.savez(dir_child + "/model.npz", **new_model)

