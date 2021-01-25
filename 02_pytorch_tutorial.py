"""
This is a very brief introduction to pytorch. Some examples are adapted from
[here](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) and
[here](https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html).
For a much more comprehensive introduction, see the [official tutorial](https://pytorch.org/tutorials/).

What is pytorch? In [its own words](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html):
"PyTorch is a Python-based scientific computing package serving two broad purposes:
 - A replacement for NumPy to use the power of GPUs and other accelerators.
 - An automatic differentiation library that is useful to implement neural networks."

We will see what these mean in this tutorial.
"""

import torch
from torch import nn
import torch.nn.functional as F


## Tensors

# Tensors are the most basic data type in pytorch.
# They are very similar to numpy arrays in terms of the interface and supported functions.
# For example:
a = torch.tensor([1, 2, 3])  # numpy: a = np.array([1, 2, 3])
b = torch.arange(12).reshape(4, 3)  # numpy: b = np.arange(12).reshape((4, 3))
c = torch.full((2, 2), 7)  # numpy: c = np.full((2, 2), 7)
print(a + b)  # numpy: a + b; note the broadcasting here
print(b.sum(dim=1))  # numpy: b.sum(axis=1)
print(a.type(torch.float))  # equivalently a.float() or a.to(torch.float); numpy: a.astype(np.float)
print(b[1:3, 2])  # numpy: b[1:3, 2]

# pytorch tensors are more powerful than numpy arrays. For example, you can move them to GPUs for
# faster computation (e.g., matrix multiplication).
b.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# They also support gradient tracking, as we will see below.
# You may ask, why did we introduce numpy first then? First of all, numpy is still very widely used
# beyond automatic differentiation. Even for ML/NLP, it is often used for data loading, metric
# calculation, etc. Second, since pytorch tensors have a very similar interface as numpy arrays,
# understanding the latter makes it easy to learn the first.


## Automatic Differentiation

# Most people use pytorch for its automatic differentiation capability.
# That is, for almost all use cases, you only need to write the forward computation and don't need
# to worry about what the gradient looks like in backprop (like we did in A1)! Specifically, we
# define a (forward) "computational graph" with tensors and operations between tensors. We can use
# the `backward` method to calculate the gradient of a scalar tensor w.r.t. all other tensors. To do
# this, we need to set the `requires_grad` flag to indicate the need for gradient tracking.
a = torch.tensor([2, 3], dtype=torch.float, requires_grad=True)
b = torch.tensor([6, 4], dtype=torch.float, requires_grad=True)
Q = 3 * a ** 3 - b ** 2  # ** is the exponential function and takes precedence over *
S = Q.sum()
S.backward()
# The gradients are automatically collected in the .grad field of each leaf tensor in the
# computational graph. The optimizer can then use these gradients to perform optimization. We can
# check if the automatically collected gradients are correct.
print(a.grad == 9 * a ** 2)
print(b.grad == -2 * b)


# Example: Multinomial Logistic Regression Bag-of-Words Classifier

# Now we walk through a toy multinomial logistic regression BoW classifier which is very similar to
# the one that we built in A1. We encourage you to compare this code with your A1 code and see what
# pytorch automatically does for you. Note that, although our model below is general and can handle
# multinomial LR, here we only use two classes.

torch.manual_seed(1)  # to make thing more reproducible

train_data = [
    ("me gusta comer en la cafeteria".split(), "SPANISH"),
    ("Give it to me".split(), "ENGLISH"),
    ("No creo que sea una buena idea".split(), "SPANISH"),
    ("No it is not a good idea to get lost at sea".split(), "ENGLISH"),
]
test_data = [("Yo creo que si".split(), "SPANISH"), ("it is lost on me".split(), "ENGLISH")]
label_to_idx = {"SPANISH": 0, "ENGLISH": 1}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# Now we build the vocabulary that maps each word to a unique index.
# For this toy example only, we look at the test set to build the vocabulary.
# This is generally not a good practice.
word_to_idx = {}
for sent, _ in train_data + test_data:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
print(word_to_idx)

VOCAB_SIZE = len(word_to_idx)
NUM_LABELS = len(label_to_idx)


# Usually, ML/NLP code consists of many models like the one below. You need to inherit from
# nn.Module to take advantage pytorch features such as automatic differentiation (notice that you
# don't need to define the backward function!), parameter tracking, etc.
class MLRBoWClassifier(nn.Module):  # inheriting from nn.Module!
    def __init__(self, num_labels, vocab_size):
        super().__init__()

        # Here we define the model parameters in the computational graph that will be optimized.
        # These are leaf nodes in the computational graph that are not data. Remember, the
        # probability of multinomial logistic regression looks like softmax(Wx + b). pytorch defines
        # nn.Linear() which provides the affine map and encapsulates the W and b parameters. Make
        # sure you understand why the input dimension is vocab_size and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # The softmax nonlinearity does not have parameters, so we don't need to worry about that here

    def forward(self, bow_vec):
        # Here we define the (forward) computational graph. Again, p_{MLR} = softmax(Wx + b).
        # One minor departure from that is that we take the log here rather than in the loss
        # for numerical stability. We can apply the affine transformation with the defined W and b
        # parameters by calling the self.linear module. You define the forward pass in the forward
        # method, but you invoke the forward pass of a module by just calling the object. Calling a
        # module invokes its __call__ method, which in turn calls the forward method but also does
        # some bookkeeping.

        # bow_vec shape: (batch_size, vocab_size)
        # score shape: (batch_size, num_labels)
        score = self.linear(bow_vec)
        # We need to supply the dim=1 argument because pytorch doesn't know which dimension to
        # calculate the softmax against.
        return F.log_softmax(score, dim=1)


def make_feature(sentence, word_to_idx):
    """Calculate the feature for each sentence which is a sum one-hot vectors."""
    vec = torch.zeros(VOCAB_SIZE)
    for word in sentence:
        vec[word_to_idx[word]] += 1
    # unsqueeze makes a leading dimension for batching; shape: (vocab_size,) -> (1, vocab_size)
    return vec.unsqueeze(0)


def make_target(label, label_to_idx):
    return torch.LongTensor([label_to_idx[label]])


model = MLRBoWClassifier(NUM_LABELS, VOCAB_SIZE)

# The model knows its parameters. The first output below is W, the second is b. Whenever you assign
# a component to a class variable in the __init__ function  of a module, which was done with
# self.linear = nn.Linear(...)
# Then through some python magic from the pytorch devs, your module (in this case, MLRBoWClassifier)
# will store knowledge of the nn.Linear's parameters!
for param in model.parameters():
    print(param)


def train(model, data):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Usually you want to pass over the training data several times.
    # 100 is much bigger than on a real data set, but real datasets have more than
    # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
    for epoch in range(100):
        for instance, label in data:
            # Step 1. Make our BOW feature vector and also we must wrap the target in a
            # Tensor as an integer. For example, if the target is SPANISH, then
            # we wrap the integer 0. The loss function then knows that the 0th
            # element of the log probabilities is the log probability
            # corresponding to SPANISH.
            bow_vec = make_feature(instance, word_to_idx)
            target = make_target(label, label_to_idx)

            # Step 2. Run our forward pass.
            log_probs = model(bow_vec)

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step().
            # But first, because loss.backward() accumulates (i.e., adds) gradients on each node
            # to the .grad field, we need to clear them out before computing the gradients.
            model.zero_grad()
            loss = F.nll_loss(log_probs, target)
            loss.backward()
            optimizer.step()


def test(model, data):
    print("testing")
    # Here we don't need to train, so the code is wrapped in torch.no_grad() which disables gradient
    # tracking and makes the code faster and more memory efficient.
    with torch.no_grad():
        for instance, label in data:
            print(instance, label)
            bow_vec = make_feature(instance, word_to_idx)
            log_probs = model(bow_vec)
            print(log_probs)
            predicted = log_probs.argmax(dim=1)
            print(f"Predicted class: {idx_to_label[predicted.item()]}")


# Run on test data before we train, just to see a before-and-after.
test(model, test_data)
print(f'Weight of "creo" before training (0 is {idx_to_label[0]} and 1 is {idx_to_label[1]}):')
print(next(model.parameters())[:, word_to_idx["creo"]])

train(model, test_data)
test(model, test_data)

# Index corresponding to Spanish goes up, English goes down!
print(f'Weight of "creo" after training (0 is {idx_to_label[0]} and 1 is {idx_to_label[1]}):')
print(next(model.parameters())[:, word_to_idx["creo"]])
