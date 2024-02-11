import random
import torch


class StereoReservoir(object):
  def __init__(self, max_size: int, min_heap=True):
    """
    A buffer that maintains a uniformly random sample of items that are streamed
    online using Reservoir Sampling.
    
    https://en.wikipedia.org/wiki/Reservoir_sampling.

    Parameters
    ----------
    * `max_size` : The maximum number of items to store in the buffer.
    """
    self.max_size = max_size
    self.buf = []
    self.indices = set()

    # Counts how many items have been streamed so far (tried to add to buffer).
    self.i = 0

  def add(self, img_l: torch.Tensor, img_r: torch.Tensor, value: float, img_index: int) -> bool:
    """
    (Maybe) adds an item to the buffer using Reservoir Sampling (Algorithm R).

    Parameters
    ----------
    * `img_l` : Left image.
    * `img_r` : Right image.
    * `value` : e.g the loss value for this image pair.
    * `img_index` : The index of this image (prevents sorting issues when identical images are added).

    Returns
    -------
    (bool) Whether or not the image pair was added to the buffer.
    """
    self.i += 1

    if img_index in self.indices:
      return False

    # If buffer hasn't reached max size, always add.
    if len(self.buf) < self.max_size:
      self.buf.append([value, img_index, img_l, img_r])
      self.indices.add(img_index)
      return True

    # Replace elements with gradually decreasing probability.
    j = random.randint(1, self.i)
    if j <= self.max_size:
      self.buf[j-1] = [value, img_index, img_l, img_r]
      return True

  def update_value(self, buf_index: int, new_value: float):
    """Update the value of an item in the buffer by its index."""
    self.buf[buf_index][0] = new_value

  def size(self) -> int:
    """The current size of the buffer."""
    return len(self.buf)

  def average_value(self):
    """Computes the average "value" for all items stored in the buffer."""
    total = 0
    for i in range(len(self.buf)):
      total += self.buf[i][0]
    return total / len(self.buf)
