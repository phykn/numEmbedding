### numEmbedding
**numEmbedding** is a PyTorch module to embed numerical values into a high-dimensional space. This module finds **NaN** values from the data and replaces them with trainable parameters.

### Requirements
- pytorch
- einops

### Parameters
- `embedding_dim` (*int*) – the size of each embedding vector

### Examples
```python
>>> # generate data with nan values
>>> shape = (3, 3)
>>> m = torch.empty(*shape).uniform_(0, 1)
>>> m = torch.bernoulli(m)
>>> i = torch.where(m > 0.5)

>>> x = torch.randn(*shape).to("cuda")
>>> x[i] = torch.nan
>>> x
tensor([[-0.1382,     nan, -0.1591],
        [    nan, -0.1746, -1.5460],
        [    nan,  1.4191,     nan]], device='cuda:0')

>>> # embedding with numEmbedding moudle
>>> embedding_dim = 4
>>> embed = numEmbedding(embedding_dim).to("cuda")
>>> embed(x)
tensor([[[ 0.4167,  0.0142,  0.4243],
         [ 0.1930,  0.7074,  0.2082],
         [-0.1636,  1.7226, -0.1762],
         [-0.8527, -0.5870, -0.8357]],

        [[ 0.0142,  0.4300,  0.9338],
         [ 0.7074,  0.2195,  1.2203],
         [ 1.7226, -0.1856, -1.0145],
         [-0.5870, -0.8231,  0.2934]],

        [[ 0.0142, -0.1554,  0.0142],
         [ 0.7074, -0.9434,  0.7074],
         [ 1.7226,  0.7777,  1.7226],
         [-0.5870, -2.1205, -0.5870]]], device='cuda:0', grad_fn=<AddBackward0>)
```

In this example, the `nan` value is replaced by
```python
tensor([ 0.0142,  0.7074,  1.7226, -0.5870], device='cuda:0', requires_grad=True)
```