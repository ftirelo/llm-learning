# llm-learning

## Intro to LLMs

- Tutorial on how to train an LLM: <https://www.youtube.com/watch?v=kCc8FmEb1nY>

- Colab: <https://colab.research.google.com/drive/1x793tnO5QIf-9JPpHKzpwkwqEyMkMNDj>

## Tokenizers

- Google's SentencePiece: <https://github.com/google/sentencepiece>
- OpenAI's TikToken: <https://github.com/openai/tiktoken>

## Evaluation

- Log-likelihood:
  - Definitions:
    - Likelihood: product of probabilities
    - Log-likelihood: sum of the logs of probabilities
    - Neg-log-likelihood: negative of Log-likelihood
    - In practice, people minimize the average
  - Model smoothing:
    - Zero-probability instances in the eval set, leading to infinity probability
    - Solution: add some bias count to avoid zero probabilities

## Initialization tips

- Over-confident softmax:
  - reduce output of the output layer
- Negligible gradients for activation function:
  - For example, derivative of `tanh(x)` is very close to zero for $x > 1$, so large $x$ basically "kills" a neuron
  - Tip: plot a binary representation of the matrix (white = 1, black = 0), should see very few whites (dead neuron)
- Very high stdev for matrix multiplication:
  - Divide by $\sqrt{n}$, where $n$ in the weight matrix fan-in
- Most common initialization approach: "kaiming normal"

## Batch normalization

- Another approach to avoid initialization issues:
  - Batch normalization after all linear and convolution hidden layers gives more stability
  - Normalize to mini-batch inputs to $N(0,1)$
  - In practice, creates some randomness that is similar to augmentation, which works as regularization and decreases overfitting
  - Note: batch normalization layers have their own bias (the other bias is added and subtracted because we are subtracting the mini-batch's mean)

- Common architecture:
  - Weight layer (multiply by a weight matrix)
  - Normalization layer (e.g. batch normalization)
  - Non-linearity layer (e.g. ReLU, tanh)
  - Good example: Google for `[resnet pytorch]`

## Some interesting diagnosis plots

- Statistics of the activations
- Statistics of the gradients
- Log base 10 of updates versus data over time, should be around 0.001

## Attention

- Attention is a *communication mechanism*
  - See the structure as a direct graph, in which each node $i$ points to $i+1, ...$
  - 