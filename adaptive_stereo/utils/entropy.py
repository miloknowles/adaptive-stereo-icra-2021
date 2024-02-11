import torch


def grayscale_shannon_entropy(img):
  """
  Computes the Shannon entropy of the grayscale intensity distribution of an image.

  img (torch.FloatTensor) : A floating-point image with values between 0 and 1.
  NOTE: This function only works for a single image, and is not batched.
  """
  img_256 = (255.0 * img).int()
  bin_counts = torch.histc(img_256, 256, min=0, max=255)

  # Normalize by the total number of pixels (height * width).
  bin_probs = bin_counts.float() / (img.shape[-2]*img.shape[-1])

  log_bin_probs_safe = torch.zeros_like(bin_probs)
  log_bin_probs_safe[bin_probs > 0] = torch.log2(bin_probs[bin_probs > 0])

  # Compute entropy from the discrete distribution.
  entropy = -1 * torch.sum(bin_probs * log_bin_probs_safe)

  return entropy


def gradient_shannon_entropy(img):
  """
  https://stats.stackexchange.com/questions/235270/entropy-of-an-image
  """
  assert(len(img.shape) == 2)
  img_256 = (255.0 * img).int()
  diff_x = img_256[:,1:] - img_256[:,:-1]
  diff_y = img[1:,:] - img[:-1,:]

  bin_counts = torch.histc(diff_x, 256, min=-255, max=255)

  # Normalize by the total number of pixels (height * width).
  bin_probs = bin_counts.float() / (diff_x.shape[-2]*diff_x.shape[-1])

  log_bin_probs_safe = torch.zeros_like(bin_probs)
  log_bin_probs_safe[bin_probs > 0] = torch.log2(bin_probs[bin_probs > 0])

  # Compute entropy from the discrete distribution.
  entropy = -1 * torch.sum(bin_probs * log_bin_probs_safe)

  return entropy
