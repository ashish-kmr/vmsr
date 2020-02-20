# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Wrapper for selecting the navigation environment that we want to train and
test on.
"""
import numpy as np
import os, glob, logging, yaml
from render import swiftshader_renderer as renderer 
from src import file_utils as fu
from src import utils as utils
from env import mp_env

class Loader():
  def __init__(self, ver, imset):
    self.ver = ver
    self.imset = imset
    self.data_dir = os.path.join('..', 'data', self.ver)
  
  def get_data_dir(self):
    return self.data_dir

  def get_imset(self):
    return self._get_split(self.imset)
  
  def get_split(self, split_name):
    return self._get_split(split_name)
  
  def get_robot(self):
    return self._load_yaml('robots')
  
  def get_env(self):
    return self._load_yaml('envs')

  def _load_yaml(self, yy):
    file_name = os.path.join('env-data', yy, self.ver+'.yaml')
    with open(file_name, 'r') as f:
      tt = yaml.load(f)
    robot = utils.Foo(**tt)
    return robot
  
  def _read_splits(self, file_name):
    ls = []
    with open(file_name, 'r') as f:
      for l in f:
        ls.append(l.rstrip())
    return ls

  def _get_split(self, split_name):
    ss1, ss2 = None, None
    split_file = os.path.join('env-data', 'splits', self.ver, split_name+'.txt')
    if os.path.exists(split_file):
      ss1 = self._read_splits(split_file)
      ss1.sort()

    mesh_dir = os.path.join(self.get_data_dir(), 'mesh', split_name)
    if os.path.exists(mesh_dir):
      ss2 = [split_name]

    assert((ss1 is not None and ss2 is None) or
      (ss2 is not None and ss1 is None))
    ss = ss1 if ss2 is None else ss2
    return ss

  def get_meta_data(self, file_name, data_dir=None):
    if data_dir is None:
      data_dir = self.get_data_dir()
    full_file_name = os.path.join(data_dir, 'meta', file_name)
    assert(fu.exists(full_file_name)), \
      '{:s} does not exist'.format(full_file_name)
    ext = os.path.splitext(full_file_name)[1]
    if ext == '.txt':
      ls = []
      with fu.fopen(full_file_name, 'r') as f:
        for l in f:
          ls.append(l.rstrip())
    elif ext == '.pkl':
      ls = utils.load_variables(full_file_name)
    return ls

  def load_building(self, name, data_dir=None):
    if data_dir is None: data_dir = self.get_data_dir()
    out = {}
    out['name'] = name
    out['data_dir'] = data_dir
    out['room_dimension_file'] = os.path.join(data_dir, 'room-dimension', name+'.pkl')
    out['class_map_folder'] = os.path.join(data_dir, 'class-maps')
    return out

  def load_building_meshes(self, building, materials_scale=1.0):
    dir_name = os.path.join(building['data_dir'], 'mesh', building['name'])
    mesh_file_name = glob.glob1(dir_name, '*.obj')[0]
    mesh_file_name_full = os.path.join(dir_name, mesh_file_name)
    logging.error('Loading building from obj file: %s', mesh_file_name_full)
    shape = renderer.Shape(mesh_file_name_full, load_materials=True, 
      name_prefix=building['name']+'_',  materials_scale=materials_scale)
    return [shape]

  def load_data(self, name, flip=False, map=None):
    robot = self.get_robot()
    env = self.get_env()
    building = mp_env.Building(self, name, robot, env, flip=flip, map=map)
    return building
