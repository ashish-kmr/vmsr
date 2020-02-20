from __future__ import print_function
from __future__ import division
import cv2, os, numpy as np
import logging
from src import file_utils 
from src import map_utils as mu 

class CampusTopView():
  def __init__(self, imset):
    self.data_dir = '../data/campus/top_view/v2/'
    # self.data_dir = '/tmp/v2/'
    self.imset = imset

  def get_imset(self):
    ls = []
    with open(os.path.join(self.data_dir, 'splits', self.imset+'.txt'), 'rt') as f:
      for l in f:
        ls.append(l.rstrip())
    return ls

  def load_data(self, name, crop=True, flip=False, map_max_size=None, rng=None, base_resolution=1.0):
    view_file = os.path.join(self.data_dir, '{:s}_20_satellite.png'.format(name))
    assert(os.path.exists(view_file)), '{:s} does not exist.'.format(view_file)
    view = cv2.imread(view_file)[:,:,::-1]
    
    road_file = os.path.join(self.data_dir, 'map_{:s}_lab.png'.format(name))
    if not os.path.exists(road_file):
      logging.error('Road file %s does not exist. Using null roads', road_file)
      road = np.zeros(view.shape, dtype=np.uint8)[:,:,0]
    else:
      assert(os.path.exists(road_file)), '{:s} does not exist.'.format(road_file)
      road = cv2.imread(road_file)[:,:,0]
    road_ = np.zeros(road.shape, dtype=np.bool)
    for id in [1, 2, 3, 6]: road_ = np.logical_or(road_, road==id)

    if base_resolution != 1.0:
      # Resize the road and the view appropriately.
      view = mu.resize_maps(view, [base_resolution], 'antialiasing')[0]
      road_ = mu.resize_maps((road_*255).astype(np.uint8), [base_resolution], 'antialiasing')[0]
      road_ = road_ > 128

    if crop:
      valid = np.linalg.norm(view*1., axis=2) > 0
      xmin = np.argmax(np.any(valid, axis=0))
      ymin = np.argmax(np.any(valid, axis=1))
      valid = valid[:,xmin:]
      valid = valid[ymin:,:]
      xmax = valid.shape[1] - (np.argmax(np.any(valid, axis=0)[::-1])-1)
      ymax = valid.shape[0] - (np.argmax(np.any(valid, axis=1)[::-1])-1)
      if xmax > 0: valid = valid[:,:xmax]
      if ymax > 0: valid = valid[:ymax,:]
      
      view = view[ymin:, xmin:,...]
      road_ = road_[ymin:, xmin:,...]
      if xmax > 0: view = view[:,:xmax,...]
      if ymax > 0: view = view[:ymax,:,...]
      if xmax > 0: road_ = road_[:,:xmax,...]
      if ymax > 0: road_ = road_[:ymax,:,...]
    
    if flip:
      view = view[:,::-1,...]
      road_ = road_[:,::-1,...]
    
    if map_max_size is not None:
      # Crop out the map so that each dimension a maximum map_max_size.
      assert(rng is not None), 'For randomly cropping the map, rng must be specified.'
      x_top = 0; y_top = 0;
      if view.shape[0] > map_max_size:
        y_top = rng.randint(view.shape[0] - map_max_size)
      if view.shape[1] > map_max_size:
        x_top = rng.randint(view.shape[1] - map_max_size)
      view = view[y_top:y_top+map_max_size, x_top:x_top+map_max_size, :]
      road_ = road_[y_top:y_top+map_max_size, x_top:x_top+map_max_size]

    logging.info('Loaded {:s} with flipping: {:d}'.format(name, flip))
    logging.info('%s, %s', str(road_.shape), str(view.shape))
    return view, road_

def test_campus_top_view():
  d = CampusTopView('small')
  names = d.get_imset()
  logging.error(names)
  view, road = d.load_data(name=names[0], crop=True, map_max_size=200, 
    rng=np.random.RandomState(0), base_resolution=0.25)
  logging.error('%s, %s', str(view.shape), str(road.shape))
