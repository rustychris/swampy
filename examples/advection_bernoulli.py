"""
Non-notebook version of the advection test
"""

import stompy.grid.unstructured_grid as ugrid
from stompy import utils
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Rather than adding swampy to sys.path, assume this script is run
# from this folder and add the relative path
utils.path("../")
from swampy import swampy
import numpy as np
import six
six.moves.reload_module(swampy)

##

class SwampyBump1D(swampy.SwampyCore):
    use_contract_factor = True
    downstream_eta = 0.33
    upstream_flow = 0.05 # m2/s

    W=1 
    L=20 # channel length
    nx=int(20/0.1)  # number of nodes in along-channel direction
    ny=2 # >=2, number of nodes in cross-channel direction

    theta=0.55
    dt=None # set by caller
     
    last_plot=-1000000
    def step_output(self,n,ei,**kwargs):
        plot_interval=1.0
        if self.t-self.last_plot<plot_interval:
            return
        self.last_plot=self.t

    def z_bed(self,xy):
        """ 
        Define bed elevation.  Positive up.
        """
        # parabolic bump, 0.2m high
        xb=10
        return (0.2 - 0.05 * (xy[:,0] - xb)**2).clip(0,np.inf)
    
    def set_grid(self):
        g=ugrid.UnstructuredGrid(max_sides=4)
        g.add_rectilinear([0,0],[self.L,self.W],
                          self.nx,self.ny)
        g.orient_edges()

        g.add_node_field('node_z_bed',self.z_bed(g.nodes['x']))
        g.add_cell_field('cell_z_bed',self.z_bed(g.cells_center()))

        super(SwampyBump1D,self).set_grid(g)
    def set_initial_conditions(self):
        # allocate:
        super(SwampyBump1D,self).set_initial_conditions()
        
        self.ic_zi[:]=-self.grd.cells['cell_z_bed']
        self.ic_ei[:]=np.maximum(self.downstream_eta,-self.ic_zi)

    snap=None
    stop=False
    def snapshot_figure(self,**kwargs):
        """
        Plot current model state with solution
        """
        def stop(event,self=self):
            self.stop=True
        
        if self.snap is None:
            self.snap={}
            fig=self.snap['fig']=plt.figure(3)
            fig.clf()
            self.snap['ax']=fig.add_subplot(1,1,1)
            axstop         =fig.add_axes([0.8, 0.93, 0.09, 0.06])
            self.snap['btn'] = Button(axstop, 'Stop')
            self.snap['btn'].on_clicked(stop)
            
            self.snap['ax'].plot(self.grd.cells_center()[:,0], -self.zi, 'b-o',ms=3)

        ax=self.snap['ax']
        del ax.lines[1:]
        ax.plot(self.grd.cells_center()[:,0], self.ei, 'g-' )
        # add analytical or measured.

    def set_bcs(self):
        self.add_bc( swampy.StageBC( geom=[[self.L,0],
                                           [self.L,self.W]],
                                     h=self.downstream_eta ) )
        self.add_bc( swampy.FlowBC( geom=[[0,0],
                                          [0,self.W]],
                                    Q=self.W*self.upstream_flow ) )


def calc_Fr_CFL_Bern(sim):
    # Just make sure we actually got a hydraulic jump
    e2c=sim.grd.edge_to_cells()
    u=sim.uj[sim.intern]
    x=sim.grd.edges_center()[sim.intern][:,0]

    ui=sim.get_center_vel(sim.uj)
    xi=sim.grd.cells_center()[:,0]
    
    i_up=np.where(sim.uj>0,
                  sim.grd.edges['cells'][:,0],
                  sim.grd.edges['cells'][:,1])
    h=sim.hi[ i_up[sim.intern] ] # or mean
    Fr=u/np.sqrt(9.8*h)
    CFL=u*sim.aj[sim.intern]*sim.dt/sim.vi[i_up[sim.intern]]
    phi=u**2/(2*9.8) + sim.ei[i_up[sim.intern]]

    return dict(x=x,xi=xi,Fr=Fr,CFL=CFL,phi=phi,u=u,
                h=h,ui=ui[:,0])

def plot_longitudinal(sim):
    V=calc_Fr_CFL_Bern(sim)

    plt.figure(4).clf()
    fig,axs=plt.subplots(4,1,sharex=True,num=4)
    axs[0].plot(V['x'],V['u'],label='u')
    axs[0].plot(V['xi'], V['ui'],label='ui')

    axs[1].plot(V['x'],V['Fr'],label='Fr')
    axs[1].plot(V['x'],V['CFL'],label='CFL')
    axs[2].plot(V['xi'],V['ui']*sim.hi,label='Qi')
    axs[2].plot(V['x'],sim.uj[sim.intern]*V['h'],label='Qij')
    axs[2].plot(V['x'],(sim.uj*sim.hjstar)[sim.intern],label='Q=uj*hjstar')

    axs[3].plot(V['xi'], -sim.zi, 'b-o',ms=3,label='bed')
    axs[3].plot(V['xi'], sim.ei, 'g-', label='eta')
    axs[3].plot(V['x'], V['phi'], label='Bernoulli' )

    axs[0].set_title(f"t={sim.t:.2f}")
    for ax in axs:
        ax.legend(loc='upper right')


if __name__=='__main__':        
    # upstream_flow=0.10 will run, but any higher
    # and there is some transient drying that causes an
    # issue.
    # this must have worked at a point, though.
    sim=SwampyBump1D(cg_tol=1e-10,dt=0.1,
                     upstream_flow = 0.05)
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.prepare_to_run()

##

sim.bcs[1].ramp_time=100.
while sim.t<400 and not sim.stop:
    (hi, uj, tvol, ei) = sim.run_until(nsteps=10)
    plot_longitudinal(sim)
    plt.draw()
    plt.pause(0.01)

## 
# with dx=0.1, by sim.t==400, this is reasonably settled in.
# bernoulli error is:
assert V['phi'].max() - V['phi'].min() < 0.002,"Bernoulli function is too variable"
assert np.all( np.abs(sim.upstream_flow - (sim.uj*sim.hjstar)[sim.intern]) < 0.002),"Flow not constant"

# This looks fine by t=500, with dx=0.1
#  Minor remaining issues:
#   ui drops at boundaries. Would be nice to fix that by setting a uj
#   at least for the outflow boundary
#   Make it clear in the code that uj*hjstar is the
#   "correct" flux (add a comment so that as we move into
#   output and metadata, we get the right Q).

##

if 0:
    plt.figure(5).clf()
    ecoll=sim.grd.plot_edges(values=sim.uj,cmap='jet')

    ecoll.set_lw(3)
    ecoll.set_clim([-0.4,0.4])
    plt.colorbar(ecoll)
    plt.axis('equal')


