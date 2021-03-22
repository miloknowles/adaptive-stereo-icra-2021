# Copyright 2020 Massachusetts Institute of Technology
#
# @file stereo_buffer.py
# @author Milo Knowles
# @date 2020-07-09 16:58:26 (Thu)

from collections import deque
import heapq

import torch


class StereoPriorityQueue(object):
  def __init__(self, max_size, min_heap=True):
    """
    max_size (int) : The maximum number of items to store in the buffer.
    min (bool) : If True, this is a min heap. If False, a max heap.
    """
    self.max_size = max_size
    self.buf = []
    self.multiplier = 1 if min_heap else -1
    self.min_heap = min_heap
    self.indices = set()

  def add(self, img_l, img_r, value, index):
    if index in self.indices:
      return False

    # If buffer hasn't reached max size, always add.
    if len(self.buf) < self.max_size:
      heapq.heappush(self.buf, [self.multiplier * value, index, img_l, img_r])
      self.indices.add(index)
      return True

    # If buffer is at full size, replace an item only if it would make the heap WORSE.
    # For a MIN heap, add item if its value is SMALLER than that of any existing item.
    # For a MAX heap, add item if its value is LARGER than that of any existing item.
    else:
      largest_item = heapq.nlargest(1, self.buf)[0]
      if (self.multiplier * value) < largest_item[0]:
        # Remove the largest (worst) item.
        self.buf.remove(largest_item)
        self.indices.remove(largest_item[1])

        # Restore heap.
        heapq.heapify(self.buf)

        # Push the new item.
        heapq.heappush(self.buf, [self.multiplier * value, index, img_l, img_r])
        self.indices.add(index)
        return True

    return False

  def size(self):
    return len(self.buf)

  def pop(self):
    return heapq.heappop(self.buf)

  def average_value(self):
    total = 0
    for i in range(len(self.buf)):
      total += self.multiplier * self.buf[i][0]
    return total / len(self.buf)
