# Copyright 2020 Massachusetts Institute of Technology
#
# @file test_stereo_buffer.py
# @author Milo Knowles
# @date 2020-07-09 17:28:31 (Thu)

import unittest

from utils.stereo_priority_queue import StereoPriorityQueue


class StereoPriorityQueueTest(unittest.TestCase):
  def test_01(self):
    q = StereoPriorityQueue(3, min_heap=True)

    self.assertEqual(q.size(), 0)

    for i in range(10):
      q.add("left", "right", i, i)
      self.assertEqual(q.size(), min(3, i + 1))

    min_item = q.pop()
    self.assertEqual(min_item[0], 0)
    self.assertEqual(q.size(), 2)

    q.add("left", "right", 0, 0)
    q.add("left", "right", -1, -1)

    print("Shouldn't see value of 2:")
    print(q.buf)
    min_item = q.pop()
    self.assertEqual(min_item[0], -1)
