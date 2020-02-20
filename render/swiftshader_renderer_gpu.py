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

r"""Implements loading and rendering of meshes. Contains 2 classes:
  Shape: Class that exposes high level functions for loading and manipulating
    shapes. This currently is bound to assimp
    (https://github.com/assimp/assimp). If you want to interface to a different
    library, reimplement this class with bindings to your mesh loading library.

  SwiftshaderRenderer: Class that renders Shapes. Currently this uses python
    bindings to OpenGL (EGL), bindings to an alternate renderer may be implemented
    here. 
"""

import numpy as np, os
import cv2, ctypes, logging, os, numpy as np
import pyassimp as assimp
from six import iteritems
from OpenGL.GLES2 import *
from OpenGL.EGL import *
from src import rotation_utils 
from src import utils
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer('egl_gpu', 0, 'Which gpu to pick for rendering from ones that are available.')

__version__ = 'swiftshader_renderer'

def get_shaders(modalities):
  rgb_shader = 'rgb_flat_color' if 'rgb' in modalities else None
  d_shader = 'depth_rgb_encoded' if 'disparity' in modalities else None
  return rgb_shader, d_shader

def sample_points_on_faces(vs, fs, rng, n_samples_per_face):
  idx = np.repeat(np.arange(fs.shape[0]), n_samples_per_face)
  
  r = rng.rand(idx.size, 2)
  r1 = r[:,:1]; r2 = r[:,1:]; sqrt_r1 = np.sqrt(r1);
  
  v1 = vs[fs[idx, 0], :]; v2 = vs[fs[idx, 1], :]; v3 = vs[fs[idx, 2], :];
  pts = (1-sqrt_r1)*v1 + sqrt_r1*(1-r2)*v2 + sqrt_r1*r2*v3
  
  v1 = vs[fs[:,0], :]; v2 = vs[fs[:, 1], :]; v3 = vs[fs[:, 2], :];
  ar = 0.5*np.sqrt(np.sum(np.cross(v1-v3, v2-v3)**2, 1))
  
  return pts, ar, idx

class Shape():
  def get_pyassimp_load_options(self):
    load_flags = assimp.postprocess.aiProcess_Triangulate;
    load_flags = load_flags | assimp.postprocess.aiProcess_SortByPType;
    load_flags = load_flags | assimp.postprocess.aiProcess_OptimizeGraph;
    load_flags = load_flags | assimp.postprocess.aiProcess_OptimizeMeshes;
    load_flags = load_flags | assimp.postprocess.aiProcess_RemoveRedundantMaterials;
    load_flags = load_flags | assimp.postprocess.aiProcess_FindDegenerates;
    load_flags = load_flags | assimp.postprocess.aiProcess_GenSmoothNormals;
    load_flags = load_flags | assimp.postprocess.aiProcess_JoinIdenticalVertices;
    load_flags = load_flags | assimp.postprocess.aiProcess_ImproveCacheLocality;
    load_flags = load_flags | assimp.postprocess.aiProcess_GenUVCoords;
    load_flags = load_flags | assimp.postprocess.aiProcess_FindInvalidData;
    return load_flags

  def __init__(self, obj_file, material_file=None, load_materials=True,
               name_prefix='', name_suffix='', materials_scale=1.0):
    if material_file is not None:
      logging.error('Ignoring material file input, reading them off obj file.')
    load_flags = self.get_pyassimp_load_options()
    scene = assimp.load(obj_file, processing=load_flags)
    self.scene = scene
    filter_ind = self._filter_triangles(scene.meshes)
    self.meshes = [scene.meshes[i] for i in filter_ind]
    for i, m in enumerate(self.meshes):
      m.name = name_prefix + m.name + '_{:05d}'.format(i) + name_suffix
    logging.error('#Meshes: %d', len(self.meshes))

    dir_name = os.path.dirname(obj_file)
    # Load materials
    materials = None
    if load_materials:
      materials = []
      for m in self.meshes:
        file_name = os.path.join(dir_name, m.material.properties[('file', 1)])
        assert(os.path.exists(file_name)), \
            'Texture file {:s} foes not exist.'.format(file_name)
        img_rgb = cv2.imread(file_name)[::-1,:,::-1]
        if img_rgb.shape[0] != img_rgb.shape[1]:
          logging.warn('Texture image not square.')
          sz = np.maximum(img_rgb.shape[0], img_rgb.shape[1])
          sz = int(np.power(2., np.ceil(np.log2(sz)))) 
          sz = int(sz * materials_scale)
          img_rgb = cv2.resize(img_rgb, (sz,sz), interpolation=cv2.INTER_LINEAR)
        else:
          sz = img_rgb.shape[0]
          sz_ = int(np.power(2., np.ceil(np.log2(sz))))
          if sz != sz_ or materials_scale != 1.:
            # logging.warn('Texture image not square of power of 2 size or ' + 
            #   'materials_scale is not 1.0. Changing size from %d to %d.', 
            #   sz, int(sz_*materials_scale))
            sz = int(sz_*materials_scale)
            img_rgb = cv2.resize(img_rgb, (sz,sz), interpolation=cv2.INTER_LINEAR)
        materials.append(img_rgb)
    self.materials = materials

  def _filter_triangles(self, meshes):
    select = []
    for i in range(len(meshes)):
      if meshes[i].primitivetypes == 4:
        # print(list(meshes[i].material.properties.keys()))
        # if 'file' in meshes[i].material.properties.keys():
        select.append(i)
    return select

  def flip_shape(self):
    for m in self.meshes:
      m.vertices[:,1] = -m.vertices[:,1]
      bb = m.faces*1
      bb[:,1] = m.faces[:,2]
      bb[:,2] = m.faces[:,1]
      m.faces = bb
      # m.vertices[:,[0,1]] = m.vertices[:,[1,0]]

  def get_vertices(self):
    vs = []
    for m in self.meshes:
      vs.append(m.vertices)
    vss = np.concatenate(vs, axis=0)
    return vss, vs

  def get_faces(self):
    vs = []
    for m in self.meshes:
      v = m.faces
      vs.append(v)
    return vs

  def get_number_of_meshes(self):
    return len(self.meshes)

  def scale(self, sx=1., sy=1., sz=1.):
    pass

  def sample_points_on_face_of_shape(self, i, n_samples_per_face, sc):
    v = self.meshes[i].vertices*sc
    f = self.meshes[i].faces
    p, face_areas, face_idx = sample_points_on_faces(
        v, f, np.random.RandomState(0), n_samples_per_face)
    return p, face_areas, face_idx
  
  def __del__(self):
    scene = self.scene
    assimp.release(scene)

class SwiftshaderRenderer():
  def __init__(self):
    self.entities = {}

  def init_display(self, width, height, fov_horizontal, fov_vertical, z_near,
    z_far, rgb_shader, d_shader, im_resize):
    self.init_renderer_egl(width, height)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if d_shader is not None and rgb_shader is not None:
      logging.fatal('Does not support setting both rgb_shader and d_shader.')
    
    if d_shader is not None:
      assert rgb_shader is None
      shader = d_shader
      self.modality = 'disparity'
    
    if rgb_shader is not None:
      assert d_shader is None
      shader = rgb_shader
      self.modality = 'rgb'
    
    self.create_shaders(os.path.join(dir_path, shader+'.vp'),
                        os.path.join(dir_path, shader + '.fp'))
    aspect = width*1./(height*1.)
    self.set_camera(fov_vertical=fov_vertical, fov_horizontal=fov_horizontal,
      z_near=z_near, z_far=z_far, aspect=aspect)
    self.im_resize = im_resize 
    self.width = width # renderer width
    self.height = height # renderer height
    self.fov_horizontal = fov_horizontal
    self.fov_vertical = fov_vertical

  def get_salt_string(self):
    """Returns a string that uniquely identifies the camera properties."""
    str_ = '{:s}-sz{:d}x{:d}-fov{:03d}x{:03d}-r{:04d}'.format(self.modality,
      self.height, self.width, int(self.fov_vertical), int(self.fov_horizontal),
      int(np.round(self.im_resize*1000)))
    return str_
  
  def get_available_gpus(self):
    gpus = []
    for i in range(100):
      try:
        f = open('/dev/nvidia{:d}'.format(i), 'r')
        f.close()
        gpus.append(i)
      except:
        None
    logging.error('Available GPUs: %s', str(gpus))
    return gpus

  def get_display(self, gpu_id):
    from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
    EGLDeviceEXT = _opaque_pointer_cls( 'EGLDeviceEXT' )
    addr = eglGetProcAddress("eglQueryDevicesEXT")
    num_devices = ctypes.c_long()
    devices = (EGLDeviceEXT*100)()
    proto = ctypes.CFUNCTYPE(EGLBoolean, EGLint, EGLDeviceEXT, arrays.GLintArray)
    eglQueryDevicesEXT = proto(addr)
    success = eglQueryDevicesEXT(10, devices, num_devices);
    
    EGLDeviceEXT = _opaque_pointer_cls( 'EGLDeviceEXT' )
    addr = eglGetProcAddress("eglGetPlatformDisplayEXT")
    proto = ctypes.CFUNCTYPE(EGLDisplay, EGLint, EGLDeviceEXT, EGLint)
    eglGetPlatformDisplayEXT = proto(addr)
    eglDpy = eglGetPlatformDisplayEXT(0x313F, devices[gpu_id], 0);
    return eglDpy

  def init_renderer_egl(self, width, height):
    major, minor = ctypes.c_long(), ctypes.c_long()
    # logging.debug('init_renderer_egl: EGL_DEFAULT_DISPLAY: %s', EGL_DEFAULT_DISPLAY)
    # egl_default_display = EGL_DEFAULT_DISPLAY #EGLNativeDisplayType.from_param(0)

    # egl_display = eglGetDisplay(egl_default_display)
    gpus = self.get_available_gpus()
    try:
      egl_gpu = FLAGS.egl_gpu
    except:
      egl_gpu = 0
      logging.error('FLAGS not being used, setting to egl_gpu to 0')
    assert egl_gpu < len(gpus), 'Requested GPU is not available.'
    gpu_id = gpus[egl_gpu]
    egl_display = self.get_display(gpu_id)
    logging.debug('init_renderer_egl: egl_display: %s', egl_display)

    eglInitialize(egl_display, major, minor)
    logging.debug('init_renderer_egl: EGL_OPENGL_API, EGL_OPENGL_ES_API: %s, %s',
                 EGL_OPENGL_API, EGL_OPENGL_ES_API)
    eglBindAPI(EGL_OPENGL_ES_API)

    num_configs = ctypes.c_long()
    configs = (EGLConfig*100)()
    local_attributes = [EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                        EGL_DEPTH_SIZE, 16, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE,]
    logging.debug('init_renderer_egl: local attributes: %s', local_attributes)
    local_attributes = arrays.GLintArray.asArray(local_attributes)
    success = eglChooseConfig(egl_display, local_attributes, configs, 100, num_configs)
    logging.debug('init_renderer_egl: eglChooseConfig success, num_configs: %d, %d', success, num_configs.value)
    egl_config = configs[0]


    context_attributes = [EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE]
    context_attributes = arrays.GLintArray.asArray(context_attributes)
    egl_context = eglCreateContext(egl_display, egl_config, EGL_NO_CONTEXT, context_attributes)

    buffer_attributes = [EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE]
    buffer_attributes = arrays.GLintArray.asArray(buffer_attributes)
    egl_surface = eglCreatePbufferSurface(egl_display, egl_config, buffer_attributes)


    eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)
    logging.debug("init_renderer_egl: egl_display: %s egl_surface: %s, egl_config: %s", egl_display, egl_surface, egl_context)

    glViewport(0, 0, width, height);

    self.egl_display = egl_display
    self.egl_surface = egl_surface
    self.egl_config =  egl_config
    self.egl_mapping = {}
    self.render_timer = utils.Timer(stream='debug')
    self.load_timer = None
    self.height = height
    self.width = width

  def create_shaders(self, v_shader_file, f_shader_file):
    v_shader = glCreateShader(GL_VERTEX_SHADER)
    with open(v_shader_file, 'r') as f:
      ls = ''
      for l in f:
        ls = ls + l
    glShaderSource(v_shader, ls)
    glCompileShader(v_shader);
    assert(glGetShaderiv(v_shader, GL_COMPILE_STATUS) == 1)

    f_shader = glCreateShader(GL_FRAGMENT_SHADER)
    with open(f_shader_file, 'r') as f:
      ls = ''
      for l in f:
        ls = ls + l
    glShaderSource(f_shader, ls)
    glCompileShader(f_shader);
    assert(glGetShaderiv(f_shader, GL_COMPILE_STATUS) == 1)

    egl_program = glCreateProgram();
    assert(egl_program)
    glAttachShader(egl_program, v_shader)
    glAttachShader(egl_program, f_shader)
    glLinkProgram(egl_program);
    assert(glGetProgramiv(egl_program, GL_LINK_STATUS) == 1)
    glUseProgram(egl_program)

    glBindAttribLocation(egl_program, 0, "aPosition")
    glBindAttribLocation(egl_program, 1, "aTextureCoord")

    self.egl_program = egl_program
    self.egl_mapping['vertexs'] = 0
    self.egl_mapping['vertexs_tc'] = 1
    
    glClearColor(0.0, 0.0, 0.0, 1.0);
    # Before enabling culling check if the triangles are oriented consistnetly or not.
    # glEnable(GL_CULL_FACE);  
    # glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

  def set_camera(self, fov_vertical, fov_horizontal, z_near, z_far, aspect):
    width = 2*np.tan(np.deg2rad(fov_horizontal)/2.0)*z_near*aspect;
    height = 2*np.tan(np.deg2rad(fov_vertical)/2.0)*z_near;
    egl_program = self.egl_program
    c = np.eye(4, dtype=np.float32)
    c[3,3] = 0
    c[3,2] = -1
    c[2,2] = -(z_near+z_far)/(z_far-z_near)
    c[2,3] = -2.0*(z_near*z_far)/(z_far-z_near)
    c[0,0] = 2.0*z_near/width
    c[1,1] = 2.0*z_near/height
    c = c.T
    
    projection_matrix_o = glGetUniformLocation(egl_program, 'uProjectionMatrix')
    projection_matrix = np.eye(4, dtype=np.float32)
    projection_matrix[...] = c
    projection_matrix = np.reshape(projection_matrix, (-1))
    glUniformMatrix4fv(projection_matrix_o, 1, GL_FALSE, projection_matrix)

  def load_default_object(self):
    v = np.array([[0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 1.0],
                  [-0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0],
                  [0.5, -0.5, 0.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    v = np.concatenate((v,v+0.1), axis=0)
    v = np.ascontiguousarray(v, dtype=np.float32)

    vbo = glGenBuffers(1)
    glBindBuffer (GL_ARRAY_BUFFER, vbo)
    glBufferData (GL_ARRAY_BUFFER, v.dtype.itemsize*v.size, v, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    self.num_to_render = 6;

  def _actual_render(self):
    for entity_id, entity in iteritems(self.entities):
      if entity['visible']:
        vbo = entity['vbo']
        tbo = entity['tbo']
        num = entity['num']

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glVertexAttribPointer(self.egl_mapping['vertexs'], 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glVertexAttribPointer(self.egl_mapping['vertexs_tc'], 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.egl_mapping['vertexs'])
        glEnableVertexAttribArray(self.egl_mapping['vertexs_tc'])
        
        glBindTexture(GL_TEXTURE_2D, tbo)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glDrawArrays(GL_TRIANGLES, 0, num)

  def render(self, take_screenshot=False, output_type=0):
    with self.render_timer.record():
      self._actual_render()
    self.render_timer.display(log_at=100, log_str='render timer: ', type='time')

    np_rgb_img = None
    np_d_img = None
    c = 1000.
    if take_screenshot:
      if self.modality == 'rgb':
        # Even though we dont want the alpha channel, opengl crashes if you
        # dont read it. Bad OpenGL.
        screenshot_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, screenshot_rgba)
        np_rgb_img = screenshot_rgba[::-1,:,:3]
        
        # Resize here if necessary.
        if self.im_resize < 1.:
          np_rgb_img = cv2.resize(np_rgb_img, None, None, fx=self.im_resize,
            fy=self.im_resize, interpolation=cv2.INTER_LINEAR)
        elif self.im_resize > 1.:
          np_rgb_img = cv2.resize(np_rgb_img, None, None, fx=self.im_resize, 
            fy=self.im_resize, interpolation=cv2.INTER_AREA)

      if self.modality == 'disparity': 
        screenshot_d = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, screenshot_d)
        np_d_img = screenshot_d[::-1,:,:3];
        np_d_img = np_d_img[:,:,2]*(255.*255./c) + np_d_img[:,:,1]*(255./c) + np_d_img[:,:,0]*(1./c)
        np_d_img = np_d_img.astype(np.float32)
        # np_d_img[np_d_img == 0] = np.NaN
        np_d_img = np_d_img[:,:,np.newaxis]
        d = np_d_img
        d[d < 0.01] = np.NaN; isnan = np.isnan(d);
        d = 100./d; d[isnan] = 0.;
        d = np.concatenate((d, isnan), axis=2)
        np_d_img = d
        
        # Resize here if necessary.
        if self.im_resize < 1.:
          np_d_img_0 = cv2.resize(np_d_img[...,0], None, None,
            fx=self.im_resize, fy=self.im_resize, interpolation=cv2.INTER_AREA)
        elif self.im_resize > 1.:
          np_d_img_0 = cv2.resize(np_d_img[...,0], None, None, fx=self.im_resize,
            fy=self.im_resize, interpolation=cv2.INTER_AREA)
        if self.im_resize != 1.:
          np_d_img_1 = cv2.resize(np_d_img[...,1], None, None,
            fx=self.im_resize, fy=self.im_resize,
            interpolation=cv2.INTER_NEAREST)
          np_d_img = np.concatenate((np_d_img_0[:,:,np.newaxis],
            np_d_img_1[:,:,np.newaxis]), axis=2)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    return np_rgb_img, np_d_img

  def _load_mesh_into_gl(self, mesh, material, trans):
    vvt = np.concatenate((mesh.vertices + trans, mesh.texturecoords[0,:,:2]), axis=1)
    vvt = np.ascontiguousarray(vvt[mesh.faces.reshape((-1)),:], dtype=np.float32)
    num = vvt.shape[0]
    vvt = np.reshape(vvt, (-1))

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vvt.dtype.itemsize*vvt.size, vvt, GL_STATIC_DRAW)
    glVertexAttribPointer(self.egl_mapping['vertexs'], 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
    glVertexAttribPointer(self.egl_mapping['vertexs_tc'], 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
    glEnableVertexAttribArray(self.egl_mapping['vertexs']);
    glEnableVertexAttribArray(self.egl_mapping['vertexs_tc']);
    assert(glGetError() == GL_NO_ERROR)
    
    tbo = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tbo)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, material.shape[1],
                 material.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE,
                 np.reshape(material, (-1)))
    # glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    # m = np.zeros([material.shape[0], material.shape[1], 4], dtype=material.dtype)
    # m[...,:3] = material
    # m[...,3] = 255
    # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m.shape[1],
    #              m.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE,
    #              np.reshape(m, (-1)))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    assert(glGetError() == GL_NO_ERROR)
    
    return num, vbo, tbo

  def load_shapes(self, shapes, name_suffix=None, trans=None):
    if trans is None: trans = [np.zeros((1,3), dtype=np.float32) for s in shapes]
    if name_suffix is None: name_suffix = ['' for s in shapes]
    if type(name_suffix) != list: 
      assert(type(name_suffix) == str)
      name_suffix = [name_suffix for s in shapes]
    entities = self.entities
    entity_ids = []
    for i, shape in enumerate(shapes):
      for j in range(len(shape.meshes)):
        name = shape.meshes[j].name + name_suffix[i]
        assert name not in entities, '{:s} entity already exists.'.format(name)
        num, vbo, tbo = self._load_mesh_into_gl(shape.meshes[j], 
          shape.materials[j], trans=trans[i])
        entities[name] = {'num': num, 'vbo': vbo, 'tbo': tbo, 'visible': False}
        entity_ids.append(name)
    return entity_ids

  def del_shapes(self, entity_ids):
    for i, entity_id in enumerate(entity_ids):
      # Delete things
      entity = self.entities.pop(entity_id, None)
      vbo = entity['vbo']
      tbo = entity['tbo']
      num = entity['num']
      glDeleteBuffers(1, [vbo])
      glDeleteTextures(1, [tbo])

  def set_entity_visible(self, entity_ids, visibility):
    for entity_id in entity_ids:
      self.entities[entity_id]['visible'] = visibility

  def position_camera(self, camera_xyz, lookat_xyz, up):
    camera_xyz = np.array(camera_xyz)
    lookat_xyz = np.array(lookat_xyz)
    up = np.array(up)
    lookat_to = lookat_xyz - camera_xyz
    lookat_from = np.array([0, 1., 0.])
    up_from = np.array([0, 0., 1.])
    up_to = up * 1.
    # np.set_printoptions(precision=2, suppress=True)
    # print up_from, lookat_from, up_to, lookat_to
    r = rotation_utils.rotate_camera_to_point_at(up_from, lookat_from, up_to, lookat_to)
    R = np.eye(4, dtype=np.float32)
    R[:3,:3] = r

    t = np.eye(4, dtype=np.float32)
    t[:3,3] = -camera_xyz

    view_matrix = np.dot(R.T, t)
    flip_yz = np.eye(4, dtype=np.float32)
    flip_yz[1,1] = 0; flip_yz[2,2] = 0; flip_yz[1,2] = 1; flip_yz[2,1] = -1;
    view_matrix = np.dot(flip_yz, view_matrix)
    view_matrix = view_matrix.T
    # print np.concatenate((R, t, view_matrix), axis=1)
    view_matrix = np.reshape(view_matrix, (-1))
    view_matrix_o = glGetUniformLocation(self.egl_program, 'uViewMatrix')
    glUniformMatrix4fv(view_matrix_o, 1, GL_FALSE, view_matrix)
    return None, None #camera_xyz, q

  def clear_scene(self):
    keys = list(self.entities.keys())
    for entity_id in keys:
      entity = self.entities.pop(entity_id, None)
      vbo = entity['vbo']
      tbo = entity['tbo']
      num = entity['num']
      glDeleteBuffers(1, [vbo])
      glDeleteTextures(1, [tbo])

  def __del__(self):
    self.clear_scene()
    eglMakeCurrent(self.egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
    eglDestroySurface(self.egl_display, self.egl_surface)
    eglTerminate(self.egl_display)

def get_r_obj(camera_param):
  cp = camera_param
  rgb_shader, d_shader = get_shaders(cp.modalities)
  r_obj = SwiftshaderRenderer()
  fov_vertical = cp.fov_vertical
  r_obj.init_display(width=cp.width, height=cp.height,
    fov_vertical=fov_vertical, fov_horizontal=cp.fov_horizontal,
    z_near=cp.z_near, z_far=cp.z_far,
    rgb_shader=rgb_shader, d_shader=d_shader, im_resize=cp.im_resize)
  r_obj.clear_scene()
  return r_obj

def _test_renderer(modality, N=16):
  dir_name = os.path.dirname(os.path.realpath(__file__))
  cube = Shape('{:s}/cube/cube.obj'.format(dir_name), load_materials=True, 
    name_prefix='cube')
  camera_param = utils.Foo(width=225, height=225, z_near=0.01, z_far=20.0,
    fov_horizontal=60., fov_vertical=60., modalities=[modality], img_channels=3,
    im_resize=1.)
  r_obj = get_r_obj(camera_param)
  entities = r_obj.load_shapes([cube])
  r_obj.set_entity_visible(entities, True)
  
  r = 3;
  out_dir = os.path.join('tmp', 'test-renderer', modality + '-' + utils.get_time_str())
  utils.mkdir_if_missing(out_dir)
  logging.error('Logging to directory: %s', out_dir)
  for i in range(N):
    angle = i*(2.*np.pi)/N
    camera_xyz = [r*np.sin(angle), r*np.cos(angle), 1.]
    lookat_xyz = [0,0,0]
    up = [0, 0, 1];
    r_obj.position_camera(camera_xyz, lookat_xyz, up)
    screenshot_rgb, _ = r_obj.render(take_screenshot=True)
    if 'rgb' in camera_param.modalities:
      if i < 100:
        _ = cv2.imwrite('{:s}/b_{:04d}.png'.format(out_dir, i), screenshot_rgb[:,:,::-1])
        assert(_), 'Failed to write output file.'
      assert(np.all(np.mean(np.reshape(screenshot_rgb, [-1,1]), axis=0) > 20))
    if 'disparity' in camera_param.modalities:
      if i < 100:
        cv2.imwrite('{:s}/mask_{:04d}.png'.format(out_dir, i), (_[...,1]*255).astype(np.uint8))
        cv2.imwrite('{:s}/depth_{:04d}.png'.format(out_dir, i), (_[...,0]).astype(np.uint8))
      assert(np.mean(_[...,0]) > 4 and np.mean(_[...,0]) < 4.6)
      assert(np.mean(_[...,1]) > 0.85 and np.mean(_[...,1]) < 0.91)
  logging.error('Finished logging to directory: %s', out_dir)
  print(r_obj.render_timer.display(log_at=1, log_str='render timer', type='calls'))

def test_rgb():
  _test_renderer('rgb')

def test_d():
  _test_renderer('disparity')

def _load_shapenet():
  dir_name = '/data0/sgupta/shapenet/03001627'
  ls = os.listdir(dir_name)
  ls.sort()
  rng = np.random.RandomState(0)
  ind = rng.permutation(len(ls))
  ls = [ls[_] for _ in ind]
  # ls = ['1b4071814d1c1ae6e2367b9e27f16a71']
  i = 0
  for l in ls:
    try:
      cube = Shape(os.path.join(dir_name, l, 'model.obj'), load_materials=True, 
        name_prefix='chair')
    except:
      continue
      # print("{:s} does not work".format(l))
    print(l)
    i = i + 1
    if i == 100:
      break

def get_chair_list(imset):
  with open('render/chairs_{:s}.txt'.format(imset), 'rt') as f:
    content = f.readlines()
    content = [x.strip() for x in content] 
  return content

def load_chairs(imset, sz=0.6):
  tt = get_chair_list(imset)
  dir_name = '/data0/sgupta/shapenet/03001627'
  chairs = []
  for name in tt:
    chair = Shape('{:s}/{:s}/model.obj'.format(dir_name, name), 
    load_materials=True, name_prefix=name)
    vs = []
    for m in chair.meshes:
      vs.append(m.vertices)
    vs = np.concatenate(vs, 0)
    min_vs = np.min(vs, 0)
    max_vs = np.max(vs, 0)
    sc = max_vs - min_vs
    sc = sc / sz
    sc[1] = np.sqrt(sc[0]*sc[2])
    min_z = min_vs[1] * sc[1]
    for m in chair.meshes:
      m.vertices = m.vertices * sc[np.newaxis, :]
      m.vertices = m.vertices[:,[2,0,1]]
      m.vertices[:,2] = m.vertices[:,2] - min_z
    chairs.append(chair)
  return chairs

def _test_shapenet(name, modality, N=16):
  dir_name = '/data0/sgupta/shapenet/03001627'
  cube = Shape('{:s}/{:s}/model.obj'.format(dir_name, name), 
    load_materials=True, name_prefix='chair')
  vs = []
  for m in cube.meshes:
    vs.append(m.vertices)
  vs = np.concatenate(vs, 0)
  min_vs = np.min(vs, 0)
  max_vs = np.max(vs, 0)
  sc = max_vs - min_vs
  sc = sc / 0.6
  sc[1] = np.sqrt(sc[0]*sc[2])
  print(sc)
  min_z = min_vs[1] * sc[1]
  for m in cube.meshes:
    m.vertices = m.vertices * sc[np.newaxis, :]
    m.vertices = m.vertices[:,[2,0,1]]
    m.vertices[:,2] = m.vertices[:,2] - min_z
  # import pdb; pdb.set_trace()
  camera_param = utils.Foo(width=225, height=225, z_near=0.01, z_far=20.0,
    fov_horizontal=60., fov_vertical=60., modalities=[modality], img_channels=3,
    im_resize=1.)
  r_obj = get_r_obj(camera_param)
  entities = r_obj.load_shapes([cube])
  r_obj.set_entity_visible(entities, True)
  
  r = 3;
  out_dir = os.path.join('tmp', 'test-renderer', 'chair', modality + '-' + utils.get_time_str())
  utils.mkdir_if_missing(out_dir)
  logging.error('Logging to directory: %s', out_dir)
  for i in range(N):
    angle = i*(2.*np.pi)/N
    camera_xyz = [r*np.sin(angle), r*np.cos(angle), 1.]
    lookat_xyz = [0,0,0]
    up = [0, 0, 1];
    r_obj.position_camera(camera_xyz, lookat_xyz, up)
    screenshot_rgb, _ = r_obj.render(take_screenshot=True)
    if 'rgb' in camera_param.modalities:
      if i < 100:
        _ = cv2.imwrite('{:s}/b_{:04d}.png'.format(out_dir, i), screenshot_rgb[:,:,::-1])
        assert(_), 'Failed to write output file.'
    if 'disparity' in camera_param.modalities:
      if i < 100:
        cv2.imwrite('{:s}/mask_{:04d}.png'.format(out_dir, i), (_[...,1]*255).astype(np.uint8))
        cv2.imwrite('{:s}/depth_{:04d}.png'.format(out_dir, i), (_[...,0]).astype(np.uint8))
  logging.error('Finished logging to directory: %s', out_dir)
  print(r_obj.render_timer.display(log_at=1, log_str='render timer', type='calls'))

def main(_):
  _test_renderer('rgb', N=160)
  _test_renderer('disparity', N=160)

if __name__ == '__main__':
  app.run(main)
  # tt = get_chair_list()
  # for t in tt[:2]:
  #   _test_shapenet(t, 'rgb')
  # _load_shapenet()
