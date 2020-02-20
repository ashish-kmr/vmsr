from src import graph_utils as gu
from src import utils
import matplotlib.pyplot as plt
import graph_tool as gt
import numpy as np

def random_ring_graph(x_radii, y_radii, num_nodes, rng):
  # Sample nodes on the circle.
  dtheta0 = np.pi*2./num_nodes
  thetas0 = np.arange(num_nodes) * dtheta0
  n0 = np.array([x_radii[0]*np.cos(thetas0), y_radii[0]*np.sin(thetas0)]).T
  thetas1 = thetas0 + dtheta0/3.
  n1 = np.array([x_radii[1]*np.cos(thetas1), y_radii[1]*np.sin(thetas1)]).T
  thetas2 = thetas0 - dtheta0/3.
  n2 = np.array([x_radii[2]*np.cos(thetas2), y_radii[2]*np.sin(thetas2)]).T

  e0 = np.array([np.arange(num_nodes), np.mod(np.arange(num_nodes)+1, num_nodes)]).T
  e1 = np.array([np.arange(num_nodes), num_nodes + np.arange(num_nodes)]).T
  e2 = np.array([np.arange(num_nodes), 2*num_nodes + np.arange(num_nodes)]).T

  g0 = np.zeros((num_nodes), dtype=np.uint32)
  g1 = g0+1; g2 = g1+1;
  
  ns = np.concatenate((n0, n1, n2), axis=0)
  es = np.concatenate((e0, e1, e2), axis=0)
  gs = np.concatenate((g0, g1, g2), axis=0)

  return es, ns, gs

def vis_ring_graph(es, ns, gs, file_name):
  fig, ax = utils.subplot(plt, (1,1), (8,8))
  colstr = 'rgb'
  for i, col in enumerate(colstr):
    print i, col
    ax.plot(ns[gs==i,0], ns[gs==i,1], col+'.')

  for e in es:
    ax.plot(ns[e,0], ns[e,1], 'k')
  
  if file_name is not None:
    fig.savefig(file_name, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close(fig)

def convert_to_gt(es, ns, gs):
  g = gt.Graph(directed=False)
  g.add_vertex(n=ns.shape[0])
  g.add_edge_list(es)
  w = np.linalg.norm(ns[es[:,0],:]-ns[es[:,1],:], axis=1)
  wt = g.new_edge_property('float')
  g.edge_properties['weight'] = wt
  for i in range(es.shape[0]):
    e = g.edge(s=es[i,0], t=es[i,1])
    wt[e] = w[i]
  v_group = g.new_vertex_property('int')
  g.vertex_properties['vertex_id'] = v_group
  v_group.a = gs
  return g

def _tmp():
  from env import toy_graph_env as tge
  import graph_tool as gt
  from graph_tool import draw
  import numpy as np
  task_params = tge.get_graph_factory_task_params(4, 'train')
  e = tge.GraphFactory(task_params)
  e.reset()
  g = e.task.graphs[1]
  wt_str = g.new_edge_property('string')
  wt_inv = g.new_edge_property('float')
  wt = g.ep['weight']
  wt_max = np.max(wt.a)*1.1
  for e in g.edges(): wt_str[e] = '{:0.1f}'.format(wt[e]); wt_inv[e] = wt_max-wt[e]
  pos = gt.draw.sfdp_layout(g, eweight=wt_inv)
  gt.draw.graph_draw(g, pos=pos, output='a.pdf', edge_pen_width=wt_inv, edge_text=wt_str, vertex_size=10)

def get_graph_factory_task_params(batch_size=4, mode='train', num_sources=20,
  num_targets=20, randomize_graph=False, seed=0):
  task_params = utils.Foo(batch_size=batch_size, mode=mode,
    num_sources=num_sources, num_targets=num_targets,
    randomize_graph=randomize_graph, seed=seed)
  return task_params

class GraphFactory():
  def __init__(self, task_params):
    # Samples batch_size different graphs. Subsequent calls to get_data return
    # path queries on these batch size different graphs.
    self.task_params = task_params
    self.reset()
    self.iteration = 0
    print(task_params)

  def reset(self):
    self.task = utils.Foo(graphs=[], all_dists=[], nodes=[], groups=[])
    rng = np.random.RandomState(self.task_params.seed)
    for i in range(self.task_params.batch_size):
      x_radii = 10*np.sort(rng.rand(3)) 
      y_radii = 10*np.sort(rng.rand(3))
      num_nodes = 8 #rng.choice(19) + 2 
      rng_ = np.random.RandomState(rng.randint(np.iinfo(np.uint32).max))
      es, ns, gs = random_ring_graph(x_radii, y_radii, num_nodes, rng_)
      g = convert_to_gt(es, ns, gs)
      tt = gt.topology.shortest_distance(g, source=None, target=None, 
        weights=g.edge_properties['weight'])
      all_dist = np.array([tt[g.vertex(i)] for i in range(g.num_vertices())])
      groups = [np.where(np.logical_or(gs==0, gs==1))[0],
        np.where(np.logical_or(gs==0, gs==2))[0], 
        np.where(np.logical_or(gs==1, gs==2))[0]]
      
      self.task.graphs.append(g)
      self.task.nodes.append(ns)
      self.task.all_dists.append(all_dist)
      self.task.groups.append(groups)
  
  def set_mode(self, mode):
    self.task_params.mode = mode

  def gen_data(self, rng):
    if self.task_params.randomize_graph and np.mod(self.iteration, 1000) == 0:
      self.task_params.seed = self.task_params.seed + 1
      self.reset()
    self.iteration = self.iteration + 1
    source_nodes = []; target_nodes = []; distances = [];
    for all_dist, nodes, group in zip(self.task.all_dists, self.task.nodes, self.task.groups):
      id = rng.randint(2) if self.task_params.mode == 'train' else (rng.randint(1)+2)
      g = group[id]
      source_ind = rng.choice(g, size=self.task_params.num_sources)
      source_node = nodes[source_ind, :]
      target_ind = rng.choice(g, size=self.task_params.num_targets)
      target_node = nodes[target_ind, :]
      distance_ = all_dist[source_ind, :]
      distance_ = distance_[:,target_ind]

      source_nodes.append(source_node)
      target_nodes.append(target_node)
      distances.append(distance_)
    source_nodes = np.array(source_nodes)
    target_nodes = np.array(target_nodes)
    distances = np.expand_dims(np.array(distances), -1)

    max_num_nodes = [ns.shape[0] for ns in self.task.nodes] 
    max_num_nodes = np.max(max_num_nodes)
    all_nodes = np.zeros((self.task_params.batch_size, max_num_nodes, 2), dtype=np.float32)
    all_nodes[:] = np.NaN
    for i, ns in enumerate(self.task.nodes):
      all_nodes[i,:ns.shape[0],:] = ns

    out = {'distances': distances, 
           'source_states': source_nodes,
           'target_states': target_nodes,
           'all_nodes': all_nodes}
    return out

# if __name__ == '__main__':
#   es, ns, gs = random_ring_graph([1, 2.2, 2.8], [2, 2.5, 3.5], 18, None)
#   vis_ring_graph(es, ns, gs, 'ring.png')
