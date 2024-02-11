import torch

d = torch.load("model.pth")
f, s = {}, {}
for key in d:
  skey = key.replace("feature_extraction.", "")
  if "feature_extraction" in key:
    f[skey] = d[key]
  else:
    s[key] = d[key]

torch.save(f, "feature_net.pth")
torch.save(s, "stereo_net.pth")
