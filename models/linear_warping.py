# Copyright 2020 Massachusetts Institute of Technology
#
# @file linear_warping.py
# @author Milo Knowles
# @date 2020-03-16 16:44:21 (Mon)

import torch
import torch.nn as tnn
import torch.nn.functional as F


class LinearWarping(tnn.Module):
  def __init__(self, height, width, device):
    super(LinearWarping, self).__init__()
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width))

    # Make a grid with shape (h, w, 2). Each entry is the (col, row) coordinate.
    # Confusingly, F.grid_sample wants (x, y) coordinates, where x is the col and y is the row.
    self._grid = torch.cat([cols.unsqueeze(-1), rows.unsqueeze(-1)], axis=-1).float().to(device)
    self._grid.requires_grad_(False)
    self._height = height
    self._width = width

  def forward(self, img, positive_disp, mode="bilinear", right_to_left=True):
    """
    If right_to_left == True (synthesize a left image using the right)
      Generates a left image using a right image and the left-centered disparity map. Each positive
      disparity value means that the corresponding right image pixel is "x" pixels to the LEFT.

      L'(x, y) = R(x - positive_disp(x, y), y)

    If right_to_left == False (synthesize a right image using the left)
      Generates a right image using a left image and the right-centered disparity map. Each positive
      disparity value means that the corresponding left image pixel is "x" pixels to the RIGHT.

      R'(x, y) = L(x + positive_disp(x, y), y)

    img (torch.Tensor) : Shape (b, c, h, w), a batch of right images/features.
    positive_disp (torch.Tensor) : Shape (b, 1, h, w) a batch of disparity images.
    right_to_left (bool) : If true, then you should be passing in a left-centered disparity map.
    """
    b, c, h, w = img.shape
    assert(h == self._height)
    assert(w == self._width)

    # Add the extra batch dimension (b, h, w, 2).
    flow = self._grid.expand(b, -1, -1, -1).clone()

    # Add the disparity to the x-dimension (horizontal).
    if right_to_left:
      # NOTE: The (-) means grab pixels to the LEFT.
      flow[:,:,:,0] -= positive_disp.permute(0, 2, 3, 1).squeeze(-1)
    else:
      flow[:,:,:,0] += positive_disp.permute(0, 2, 3, 1).squeeze(-1)

    # Normalize coordinates to be in [-1, +1].
    flow[:,:,:,0] = (2 * flow[:,:,:,0] / w) - 1.0
    flow[:,:,:,1] = (2 * flow[:,:,:,1] / h) - 1.0

    valid_mask = (flow >= -1.0) * (flow <= 1.0)
    valid_mask = valid_mask[:,:,:,0] * valid_mask[:,:,:,1]

    return F.grid_sample(img, flow, mode=mode, padding_mode="border"), valid_mask.unsqueeze(1)


class DispToFlow(tnn.Module):
  def __init__(self, batch_size, height, width):
    super(DispToFlow, self).__init__()
    self.height = height
    self.width = width

    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width))

    # Make a grid with shape (h, w, 2). Each entry is the (col, row) coordinate.
    # Confusingly, F.grid_sample wants (x, y) coordinates, where x is the col and y is the row.
    grid = torch.cat([cols.unsqueeze(-1), rows.unsqueeze(-1)], axis=-1).float()
    grid.requires_grad_(False)
    self.grid = tnn.Parameter(grid.expand(batch_size, -1, -1, -1))

  def forward(self, im_left_disp):
    flow = self.grid.clone()

    # Add the disparity to the x-dimension (horizontal).
    # NOTE: The (-) means grab pixels to the LEFT.
    flow[:,:,:,0] -= im_left_disp.permute(0, 2, 3, 1).squeeze(-1)

    # Normalize coordinates to be in [-1, +1].
    flow[:,:,:,0] = (2 * flow[:,:,:,0] / self.width) - 1.0
    flow[:,:,:,1] = (2 * flow[:,:,:,1] / self.height) - 1.0

    return flow, valid_mask


def convert_disp_to_flow(im_left_disp, height, width):
  """
  Converts a left positive-disparity map to a "flow" image. The positive disparity value means
  that the corresponding right image pixel is "x" pixels to the LEFT.

  F(x, y) = (x - im_left_disp(x, y), y)

  NOTE: This is probably slow due to making the grid every time. Only use for debugging or testing.

  im_left_disp (torch.Tensor) : Shape (b, 1, h, w) a batch of disparity images, centered at the
  left image. Each pixel in im_left_disp is the (positive) offset to the corresponding pixel in
  the right image.
  """
  assert(len(im_left_disp.shape) == 4)
  b = im_left_disp.shape[0]
  rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width))

  # Make a grid with shape (h, w, 2). Each entry is the (col, row) coordinate.
  # Confusingly, F.grid_sample wants (x, y) coordinates, where x is the col and y is the row.
  grid = torch.cat([cols.unsqueeze(-1), rows.unsqueeze(-1)], axis=-1).float().to(im_left_disp.device)
  grid.requires_grad_(False)

  # Add the extra batch dimension (b, h, w, 2).
  flow = grid.expand(b, -1, -1, -1).clone()

  # Add the disparity to the x-dimension (horizontal).
  # NOTE: The (-) means grab pixels to the LEFT.
  flow[:,:,:,0] -= im_left_disp.permute(0, 2, 3, 1).squeeze(-1)

  # Normalize coordinates to be in [-1, +1].
  flow[:,:,:,0] = (2 * flow[:,:,:,0] / width) - 1.0
  flow[:,:,:,1] = (2 * flow[:,:,:,1] / height) - 1.0

  return flow
