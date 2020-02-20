"""
===========
MovieWriter
===========

This example uses a MovieWriter directly to grab individual frames and write
them to a file. This avoids any event loop integration, but has the advantage
of working with even the Agg backend. This is not recommended for use in an
interactive setting.

"""
# -*- noplot -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)# :, extra_args={'facecolor':'black'})


plt.xlim(-5, 5)
plt.ylim(-5, 5)

x0, y0 = 0, 0

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(3,4)
gs.update(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.05, hspace=0.05)
ax_view = plt.subplot(gs[:3,:3]) # First person view
ax_teacher = plt.subplot(gs[0,-1:]) # Reference image
ax_synth = plt.subplot(gs[1,-1:]) # Reference image
ax_map = plt.subplot(gs[2,-1:])


with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(2):
        ax_map.imshow(np.random.randn(10,10))
        ax_view.imshow(np.random.randn(10,10))
        ax_view.text(0, 0, 't=0', fontdict={'fontsize': 20, 'color': 'red'},
          bbox=dict(facecolor='white', alpha=0.5, lw=0))
        
        ax_view.text(.5, 1., 'center', verticalalignment='top', 
          horizontalalignment='center', transform=ax_view.transAxes,
          fontdict={'fontsize': 20, 'color': 'red'},
          bbox=dict(facecolor='white', alpha=0.5, lw=0))

        ax_view.text(1., 1., '$\eta_t=0$', verticalalignment='top', 
          horizontalalignment='right', transform=ax_view.transAxes,
          fontdict={'fontsize': 20, 'color': 'red'},
          bbox=dict(facecolor='white', alpha=0.5, lw=0))
        # ax_map.axis('off')
        # ax_view.axis('off')
        ax_teacher.imshow(np.random.randn(10,10))
        # ax_teacher.axis('off')
        ax_synth.imshow(np.random.randn(10,10))
        # ax_synth.axis('off')
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        ax_map.plot(x0, y0)
        writer.grab_frame(**{'facecolor':'black'})
