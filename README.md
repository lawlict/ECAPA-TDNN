
## Unofficial implementation of the ([ECAPA-TDNN model](https://arxiv.org/pdf/2005.07143.pdf)).

# Usage:
```shell
from ecapa_tdnn import ECAPA_TDNN

# Input size: batch_size * seq_len * feat_dim
x = torch.zeros(2, 200, 80)
model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192)
out = model(x)
print(model)
print(out.shape)    # should be [2, 192]
```
