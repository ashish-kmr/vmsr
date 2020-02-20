import cv2
from src import utils
import numpy as np
import matplotlib.pyplot as plt

def extract_images(I, mask):
  # Returns a list of images contained in I.
  x_proj = np.sum(mask > 128, axis=0);
  y_proj = np.sum(mask > 128, axis=1);
  nums = []; szs = []; starts = [];
  
  for proj in [y_proj, x_proj]:
    starts_ = np.logical_and(proj[:-1]==0, proj[1:]>0);
    ends_ = np.logical_and(proj[:-1]>0, proj[1:]==0);
    starts_ = np.where(starts_)[0]+1;
    ends_ = np.where(ends_)[0];
    assert(starts_.shape[0] == ends_.shape[0])
    sz_ = np.max(ends_-starts_)
    num_ = ends_.shape[0]

    starts.append(starts_)
    szs.append(sz_)
    nums.append(num_)
  ims = [[None for _ in range(nums[1])] for _ in range(nums[0])]

  for i in range(nums[0]):
    for j in range(nums[1]):
      ims[i][j] = I[starts[0][i]:starts[0][i]+szs[0]+1, starts[1][j]:starts[1][j]+szs[1]+1,:]

  return ims

def tile_images(out_file_name, file1=None, file2=None, ims1=None, ims2=None):
  if file1 is not None:
    I = cv2.imread(file1, cv2.IMREAD_UNCHANGED)
    ims1 = extract_images(I[:,:,:3][:,:,::-1], I[:,:,3])
  if file2 is not None:
    I = cv2.imread(file2, cv2.IMREAD_UNCHANGED)
    ims2 = extract_images(I[:,:,:3][:,:,::-1], I[:,:,3])
  fig, axes = utils.subplot(plt, (len(ims1)*2, len(ims1[0])), (8, 8))
  axes = axes.ravel()[::-1].tolist()
  for i in range(len(ims1)):
    for j in range(len(ims1[0])):
      ax = axes.pop()
      ax.imshow(ims1[i][j])
      ax = axes.pop()
      ax.imshow(ims2[i][j])
      ax.get_xaxis().set_ticks([])
      ax.get_yaxis().set_ticks([])
  fig.savefig(out_file_name, bbox_inches='tight', transparent=True, pad_inches=0)
  plt.close(fig)
