"""
Test flow over a bump. 
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
    upstream_flow = 0.12 # m2/s

    W=1 
    L=20 # channel length
    nx=int(20/0.05)  # number of nodes in along-channel direction
    ny=2 # >=2, number of nodes in cross-channel direction

    theta=0.55
    dt=1.0 # fix
     
    def __init__(self,**kw):
        utils.set_keywords(self,kw)

        super(SwampyBump1D,self).__init__(**kw)

    last_plot=-1000000
    def step_output(self,n,ei,**kwargs):
        nondims=calc_Fr_CFL_Bern(self)
        if nondims['CFL'].max()>=0.95:
            self.snapshot_figure()
            self.stop=True
            raise Exception("CFL too high: %.3f"%( nondims['CFL'].max() ))
        
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
        super(SwampyBump1D,self).set_grid(g)
        
    def set_initial_conditions(self):
        # allocate:
        super(SwampyBump1D,self).set_initial_conditions()
        
        self.ic_ei[:]=self.downstream_eta

        for i in range(self.grd.Ncells()):
            node_x=self.grd.nodes['x'][self.grd.cell_to_nodes(i),0]
            alpha=(np.arange(self.max_subcells)+0.5)/self.max_subcells
            x_sub = (1-alpha)*node_x.min() + alpha*node_x.max()
            y_sub = self.grd.cells_center()[i,1]*np.ones(self.max_subcells)

            self.ic_zi_sub['z'][i,:]=self.z_bed(np.c_[x_sub,y_sub])
            self.ic_zi_sub['A'][i,:]=self.grd.cells_area()[i]/self.max_subcells

        # So edges can pull agg values
        self.prepare_cell_subgrid(self.ic_zi_sub)

        e2c=self.grd.edge_to_cells().copy()
        missing=e2c.min(axis=1)<0
        e2c[missing,:] = e2c[missing,:].max()
        
        z_min=self.zi_agg['min'][e2c].max(axis=1) 
        self.ic_zj_sub['z'][:,0]=z_min
        self.ic_zj_sub['l'][:,0]=self.grd.edges_length()
        
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
            self.snap['ax']=fig.add_subplot(2,1,1)
            self.snap['ax2']=fig.add_subplot(2,1,2)
            axstop         =fig.add_axes([0.8, 0.93, 0.09, 0.06])
            self.snap['btn'] = Button(axstop, 'Stop')
            self.snap['btn'].on_clicked(stop)
            
            self.snap['ax'].plot(self.grd.cells_center()[:,0], self.zi_agg['min'], 'b-o',ms=3)
            self.snap['t_label']=self.snap['ax'].text(0.05,0.85,'time',
                                                      transform=self.snap['ax'].transAxes)

        ax=self.snap['ax']
        del ax.lines[1:]
        ax.plot(self.grd.cells_center()[:,0], self.ei, 'g-' )
        self.snap['t_label'].set_text("%.4f"%self.t)
        # add analytical or measured.

        ax2=self.snap['ax2']
        ax2.cla()
        nondims=calc_Fr_CFL_Bern(sim)
        ax2.plot(nondims['x'],
                 nondims['Fr'],
                 label='Fr')
        ax2.plot(nondims['x'],
                 nondims['CFL'],
                 label='CFL')
        ax2.legend(loc='upper right')

    def set_bcs(self):
        self.add_bc( swampy.StageBC( geom=[[self.L,0],
                                           [self.L,self.W]],
                                     h=self.downstream_eta ) )
        self.add_bc( swampy.FlowBC( geom=[[0,0],
                                          [0,self.W]],
                                    Q=self.W*self.upstream_flow ) )

def calc_Fr_CFL_Bern(sim):
    # To make sure we actually got a hydraulic jump
    e2c=sim.grd.edge_to_cells()
    u=sim.uj[sim.intern]
    x=sim.grd.edges_center()[sim.intern][:,0]

    ui=sim.get_center_vel(sim.uj)
    xi=sim.grd.cells_center()[:,0]
    
    i_up=np.where(sim.uj>0,
                  sim.grd.edges['cells'][:,0],
                  sim.grd.edges['cells'][:,1])
    i_up=i_up[sim.intern]

    ej=sim.ei[i_up]
    h=ej - sim.zj_agg['min'][sim.intern] 
    Fr=u/np.sqrt(9.8*h)
    CFL=u*sim.aj[sim.intern]*sim.dt/sim.vi[i_up]
    phi=u**2/(2*9.8) + sim.ei[i_up]

    return dict(x=x,xi=xi,Fr=Fr,CFL=CFL,phi=phi,u=u,
                h=h,ui=ui[:,0])


def test_low_flow():
    """
    bump test case, all subcritical.
    This also tests ramp time for flow bc, conservation of Bernoulli
    """
    sim=SwampyBump1D(cg_tol=1e-10,dt=0.05,
                     nx=int(20/0.1),
                     upstream_flow = 0.05)
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.bcs[1].ramp_time=100.0
    sim.run(t_end=250)
    
    V=calc_Fr_CFL_Bern(sim)
    assert V['Fr'].max() < 1.0,"Should be entirely subcritical"
    assert V['phi'].max() - V['phi'].min() < 0.002,"Bernoulli function is too variable"
    Q=(sim.uj*sim.hjstar)[sim.intern]
    assert np.all( np.abs(sim.W*sim.upstream_flow - Q) < 0.002),"Flow not constant"

def test_med_flow():
    """
    Bump test case, small hydraulic jump
    """
    # used to be dt=0.04, but that violates CFL
    # condition.
    # 0.02 is stable, if slow.
    sim=SwampyBump1D(cg_tol=1e-10,dt=0.02,
                     upstream_flow = 0.10)
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.run(t_end=50)

    Fr=calc_Fr_CFL_Bern(sim)['Fr']
    assert Fr[0] < 1.0,"Inflow is not subcritical"
    assert Fr[-1] < 1.0,"Outflow is not subcritical"
    assert Fr.max() > 1.0,"Did not reach supercritical"
    
def test_high_flow():
    """
    Bump test case, hydraulic jump.
    With a longer timestep this violates CFL, goes unstable,
    has some drying, and crashes.
    Not so interesting, but stable, now with the longer time
    step.
    """
    sim=SwampyBump1D(cg_tol=1e-10,dt=0.01,
                     upstream_flow = 0.15)
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.run(t_end=30)
        
if 1: # __name__=='__main__':        
    # upstream_flow=0.10 will run, but any higher
    # and there is some transient drying that causes an
    # issue.
    # this must have worked at a point, though.
    # with upstream flow=0.15, CFL breaks 0.95
    # before things really blow up.
    # same with dt=0.025
    # dt=0.01 is okay.  Just takes forever.
    sim=SwampyBump1D(cg_tol=1e-10,dt=0.01,
                     upstream_flow = 0.05)
    sim.get_fu=sim.get_fu_no_adv
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.prepare_to_run()
    sim.snapshot_figure()
    plt.draw()
    plt.pause(0.01)

    while sim.t<30 and not sim.stop:
        (hi, uj, tvol, ei) = sim.run_until(sim.t+1.0)
        sim.snapshot_figure()
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.01)

