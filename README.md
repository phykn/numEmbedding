### numEmbedding
**numEmbedding** is a PyTorch module to embed numerical values into a high-dimensional space. This module finds **NaN** values from the data and replaces them with trainable parameters.

### Requirements
- pytorch
- einops

### Parameters
- `embedding_dim` (*int*) – the size of each embedding vector

### Examples
```python
import torch
from embedding import numEmbedding

# generate data with nan values
shape = (3, 3)
m = torch.empty(*shape).uniform_(0, 1)
m = torch.bernoulli(m)
i = torch.where(m > 0.5)

x = torch.randn(*shape).to("cuda")
x[i] = torch.nan
x
```

```markdown
tensor([[    nan,     nan, -1.5096],
        [ 0.1795,     nan,  1.2659],
        [    nan,     nan,     nan]], device='cuda:0')
```

```python
# embedding with numEmbedding moudle
embedding_dim = 4
embed = numEmbedding(embedding_dim).to("cuda")
embed(x)
```

```markdown
tensor([[[-0.2957, -0.8586,  0.1556, -1.1035],
         [-0.2957, -0.8586,  0.1556, -1.1035],
         [-0.1253,  0.7708,  0.3039, -2.0912]],

        [[-0.5808, -0.3764, -0.5507, -0.4862],
         [-0.2957, -0.8586,  0.1556, -1.1035],
         [-0.8737, -1.1142, -1.1004,  0.5461]],

        [[-0.2957, -0.8586,  0.1556, -1.1035],
         [-0.2957, -0.8586,  0.1556, -1.1035],
         [-0.2957, -0.8586,  0.1556, -1.1035]]], device='cuda:0', grad_fn=<AddBackward0>)
```

In this example, the `nan` value is replaced by

```markdown
tensor([-0.2957, -0.8586,  0.1556, -1.1035], device='cuda:0', requires_grad=True)
```