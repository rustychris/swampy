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

from test_common import calc_Fr_CFL_Bern
six.moves.reload_module(swampy)
#

class SwampyBump1D(swampy.SwampyCore):
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

        # record timeseries
        self.history=np.zeros( 0, dtype=[('t',np.float64),
                                         ('eta_var',np.float64),
                                         ('Q_max_error',np.float64),
                                         ('phi_range',np.float64)])

    last_plot=-1000000
    last_uj=None
    last_ei=None
    def step_output(self,n,ei,**kwargs):
        nondims=calc_Fr_CFL_Bern(self)
        if nondims['CFL'].max()>=0.95:
            self.snapshot_figure()
            self.stop=True
            raise Exception("CFL too high: %.3f"%( nondims['CFL'].max() ))

        self.max_d_uj=self.max_d_eta=0.0
        if self.last_uj is not None:
            self.max_d_uj=np.abs((self.uj - self.last_uj)/self.dt).max()
        if self.last_ei is not None:
            self.max_d_eta=np.abs( (self.ei - self.last_ei)/self.dt).max()
        self.last_uj=self.uj.copy()
        self.last_ei=self.ei.copy()

        self.history=utils.array_append(self.history)
        self.history['eta_var'][-1]=np.var(self.ei)
        phi=nondims['phi'][ (nondims['x']>1) & (nondims['x']<19)]
        self.history['phi_range'][-1]=phi.max() - phi.min()
        self.history['t'][-1]=self.t

        Qbc=sim.W*sim.upstream_flow
        Qcalc=(sim.uj*sim.aj)[sim.intern]
        self.history['Q_max_error'][-1]=np.abs(Qbc-Qcalc).max()
        
        plot_interval=1.0
        if self.t-self.last_plot<plot_interval:
            return
        self.last_plot=self.t

    def report_steadiness(self):
        print(f"Max du/dt: {self.max_d_uj:0.5f}  Max deta/dt {self.max_d_eta:0.5f}")
        
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
        e2c[missing,:] = e2c[missing,:].max(axis=1)[:,None]
        
        z_min=self.zi_agg['min'][e2c].max(axis=1)
        self.ic_zj_sub['z'][:,0]=z_min
        self.ic_zj_sub['l'][:,0]=self.grd.edges_length()
        # So we can calculate flows

        flow_A=self.W * (self.ic_ei-self.zi_agg['mean'])
        ui_tmp=np.array([1,0])[None,:] * (self.upstream_flow / flow_A)[:,None]
        self.uj[:] = (self.cell_to_edge_interp(ui_tmp) * self.en).sum(axis=1)
        
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
            fig=self.snap['fig']=plt.figure(4)
            fig.clf()
            self.snap['ax']=fig.add_subplot(3,1,1)
            self.snap['ax2']=fig.add_subplot(3,1,2)
            axstop         =fig.add_axes([0.8, 0.93, 0.09, 0.06])
            self.snap['btn'] = Button(axstop, 'Stop')
            self.snap['btn'].on_clicked(stop)
            
            self.snap['ax'].plot(self.grd.cells_center()[:,0], self.zi_agg['min'], 'b-o',ms=3)
            self.snap['t_label']=self.snap['ax'].text(0.05,0.85,'time',
                                                      transform=self.snap['ax'].transAxes)

            self.snap['ax_t']=fig.add_subplot(3,1,3)
            
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
        ax.plot(nondims['x'],
                nondims['phi'], # / nondims['phi'].mean(),
                'r-',
                label='phi')
        ax2.axis(ymin=0,ymax=1.05)
        ax2.legend(loc='upper right')

        ax=self.snap['ax_t']
        ax.cla()
        ax.plot(self.history['t'],self.history['eta_var'],label='var(eta)')
        ax.plot(self.history['t'],self.history['Q_max_error'],label='Q nonuniform')
        ax.plot(self.history['t'],self.history['phi_range'],label='max(phi)-min(phi)')
        ax.legend(loc='upper right')
        return self.snap['fig']    

    def set_bcs(self):
        self.add_bc( swampy.StageBC( geom=[[self.L,0],
                                           [self.L,self.W]],
                                     z=self.downstream_eta ) )
        self.add_bc( swampy.FlowBC( geom=[[0,0],
                                          [0,self.W]],
                                    Q=self.W*self.upstream_flow ) )

def calc_Fr_CFL_Bern(sim):
    # To make sure we actually got a hydraulic jump
    e2c=sim.grd.edge_to_cells()
    u=sim.uj[sim.intern]
    x=sim.grd.edges_center()[sim.intern][:,0]

    ui=sim.get_center_vel(uj=sim.uj)
    xi=sim.grd.cells_center()[:,0]

    # TODO: more rigorous choice of ej
    # ej=sim.cell_to_edge_upwind(sim.ei,sim.uj)[sim.intern]
    # interp gets a 'better' answer.
    ej=sim.cell_to_edge_interp(sim.ei)[sim.intern]
    vj=sim.cell_to_edge_upwind(sim.vi,sim.uj)[sim.intern]
    
    hj=ej - sim.zj_agg['min'][sim.intern]
    
    Fr=u/np.sqrt(9.8*hj)
    CFL=u*sim.aj[sim.intern]*sim.dt/vj
    phi=u**2/(2*9.8) + ej

    return dict(x=x,xi=xi,Fr=Fr,CFL=CFL,phi=phi,u=u,
                h=hj,ui=ui[:,0])


def test_low_flow_no_adv():
    run_low_flow(advection='no_adv')

def test_low_flow_orig():
    run_low_flow(advection='orig')
    
def run_low_flow(advection='no_adv'):
    """
    bump test case, all subcritical.
    """
    sim=SwampyBump1D(cg_tol=1e-10,
                     # nx=int(20/0.5),dt=0.5,
                     nx=int(20/0.10),dt=0.10,
                     upstream_flow = 0.05)
    if advection=='no_adv':
        sim.get_fu=sim.get_fu_no_adv
    elif advection=='orig':
        sim.get_fu=sim.get_fu_orig
    else:
        raise Exception("Bad advection: "+advection)
    
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.prepare_to_run()
    
    c,j_Q,Q=sim.bcs[1].cell_edge_flow()
    
    assert np.all(sim.aj[j_Q]>0.0)

    sim.run_until(t_end=2000)

    assert np.all(sim.aj[j_Q]>0.0)
    
    V=calc_Fr_CFL_Bern(sim)
    assert V['Fr'].max() < 1.0,"Should be entirely subcritical"
    # coarse grid, no advection, I get delta of 0.0061 here
    # so 0.002 is reasonable, though not a really strong check
    # on advection.
    # Discard the ends -- maybe once momentum is advected at
    # boundaries this won't be a problem
    
    phi=V['phi'][ (V['x']>1) & (V['x']<19)]
    
    adv_passes=(phi.max() - phi.min()) < 0.002
    print("phi max-min",V['phi'].max()-V['phi'].min())
    if advection=='no_adv':
        assert not adv_passes,"Bernoulli function should be variable w/o advection"
    else:
        assert adv_passes,"Bernoulli function is too variable"

    Q=(sim.uj*sim.aj)[sim.intern]
    assert np.all( np.abs(sim.W*sim.upstream_flow - Q) < 0.002),"Flow not uniform"

    # basic test that center_vel is not crazy
    # approximate comparison between edge velocity and the cell-centered
    # x-velocity averaged from the adjacent cells
    ui_at_j=V['ui'][ sim.grd.edges['cells'][sim.intern,:] ].mean(axis=1)
    delta=sim.uj[sim.intern] - ui_at_j
    ui_rel_difference = np.mean(np.abs(delta)) / np.mean(abs(sim.uj[sim.intern]))
    # was 0.014 in a quick test
    assert ui_rel_difference<0.05


def test_med_flow():
    """
    Bump test case, small hydraulic jump.
    Just tests for stability, and that flow has 
    sub - super - sub critical transitions
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
        
if 0: # __name__=='__main__':        
    # upstream_flow=0.10 will run, but any higher
    # and there is some transient drying that causes an
    # issue.
    sim=SwampyBump1D(cg_tol=1e-10,dt=0.02,
                     nx=int(20/0.10),
                     upstream_flow = 0.15)
    sim.get_fu=sim.get_fu_orig
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.prepare_to_run()
    fig=sim.snapshot_figure()
    plt.draw()
    plt.pause(0.01)

    while sim.t<30 and not sim.stop:
        (hi, uj, tvol, ei) = sim.run_until(sim.t+2.0)
        sim.report_steadiness()
        sim.snapshot_figure()
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.01)
# else:
#     # Oddly, the Q nonuniformity actually grows over time
#     # in this run. Maybe some residual in cg?
#     test_low_flow_no_adv() # phi diff 0.00648
#     test_low_flow_orig()   #          0.0010


