"""
cifar-10 dataset, with support for random labels
"""
import numpy as np

import torch
import torchvision.datasets as datasets

class CIFAR10Partial(datasets.CIFAR10):
    def __init__(self, example_per_class=-1, num_classes=10, categories=[], **kwargs):
        super(CIFAR10Partial, self).__init__(**kwargs)
        self.n_classes = num_classes
        self.categories = categories
        self.example_per_class = example_per_class
        if self.example_per_class > 0:
            self.subsample()
    def subsample(self):
        labels = np.array(self.targets)
        indices = []
        if len(self.categories) == 0:
          for i in range(self.n_classes):
              np.random.seed(12345)
              idx = np.random.permutation(np.where(labels == i)[0])[:self.example_per_class]
              indices.append(idx)
        else:
          for i in self.categories:
              np.random.seed(12345)
              idx = np.random.permutation(np.where(labels == i)[0])[:self.example_per_class]
              indices.append(idx)
        indices = np.hstack(indices)
        
        self.data = self.data[indices]
        # self.targets = self.targets[indices]
        labels = [int(labels[i]) for i in indices]
        self.targets = labels
        
# copied from https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
class CIFAR10RandomLabels(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=-1.0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    self.targets = labels
