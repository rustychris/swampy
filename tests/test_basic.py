"""
Test flow though two cells.  Super basic to spot fundamental errors.
"""

import stompy.grid.unstructured_grid as ugrid
from stompy import utils
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from test_common import calc_Fr_CFL_Bern

# Rather than adding swampy to sys.path, assume this script is run
# from this folder and add the relative path
utils.path("../")
from swampy import swampy
import numpy as np

##

class TwoCell(swampy.SwampyCore):
    use_contract_factor = True
    downstream_eta = 0.30
    upstream_flow = 0.10 # m2/s

    W=1 
    L=20 # channel length
    nx=3  # number of nodes in along-channel direction
    ny=2 # >=2, number of nodes in cross-channel direction

    theta=0.55
    dt=1.0 
     
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        super(TwoCell,self).__init__(**kw)

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
        return -1*np.ones_like(xy[0])
    
    def set_grid(self):
        g=ugrid.UnstructuredGrid(max_sides=4)
        g.add_rectilinear([0,0],[self.L,self.W],
                          self.nx,self.ny)
        g.orient_edges()
        g.add_cell_field('cell_z_bed',self.z_bed(g.cells_center()))

        super(TwoCell,self).set_grid(g)
    def set_initial_conditions(self):
        self.max_subcells=1
        self.max_subedges=1

        # allocate:
        super(TwoCell,self).set_initial_conditions()
        self.ic_ei[:]=self.downstream_eta

        for i in range(self.grd.Ncells()):
            self.ic_zi_sub['z'][i,:]=-1.0
            self.ic_zi_sub['A'][i,:]=self.grd.cells_area()[i]/self.max_subcells
        for j in range(self.grd.Nedges()):
            self.ic_zj_sub['z'][j,:]=-1.0
            self.ic_zj_sub['l'][j,:]=self.grd.edges_length()[j]/self.max_subedges

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
            self.snap['ax']=fig.add_subplot(3,1,1)
            self.snap['ax_u']=fig.add_subplot(3,1,2)
            self.snap['ax_nondim']=fig.add_subplot(3,1,3)
            axstop         =fig.add_axes([0.8, 0.93, 0.09, 0.06])
            self.snap['btn'] = Button(axstop, 'Stop')
            self.snap['btn'].on_clicked(stop)
            
            self.snap['ax'].plot(self.grd.cells_center()[:,0], self.zi_agg['mean'],
                                 'b-o',ms=3)

            ui=self.get_center_vel(self.uj)
            self.snap['ax_u'].plot(self.grd.cells_center()[:,0],
                                   ui,'b-o')
            
            self.snap['t_label']=self.snap['ax'].text(0.05,0.85,'time',
                                                      transform=self.snap['ax'].transAxes)

        ax=self.snap['ax']
        del ax.lines[1:]
        ax.plot(self.grd.cells_center()[:,0], self.ei, 'g-' )
        self.snap['t_label'].set_text("%.4f"%self.t)
        # add analytical or measured.

        del self.snap['ax_u'].lines[0:]
        ui=self.get_center_vel(self.uj)
        self.snap['ax_u'].plot(self.grd.cells_center()[:,0],
                               ui[:,0],'b-o')
        
        ax2=self.snap['ax_nondim']
        ax2.cla()
        nondims=calc_Fr_CFL_Bern(sim)
        ax2.plot(nondims['x'],
                 nondims['Fr'],'o-',
                 label='Fr')
        ax2.plot(nondims['x'],
                 nondims['CFL'],'o-',
                 label='CFL')
        ax2.legend(loc='upper right')
        return self.snap['fig']

    def set_bcs(self):
        self.add_bc( swampy.StageBC( geom=[[self.L,0],
                                           [self.L,self.W]],
                                     z=self.downstream_eta ) )
        self.add_bc( swampy.FlowBC( geom=[[0,0],
                                          [0,self.W]],
                                    Q=self.W*self.upstream_flow ) )



def test_basic():
    sim=TwoCell(cg_tol=1e-10,dt=1.0)
    sim.get_fu = sim.get_fu_no_adv
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.bcs[1].ramp_time=10.0
    sim.run(t_end=250)
        
if False: # __name__=='__main__':
    sim=TwoCell(cg_tol=1e-10,dt=1.0)
    sim.set_grid()
    sim.set_initial_conditions()
    sim.set_bcs()
    sim.bcs[1].ramp_time=40.0

    sim.prepare_to_run()
    
    sim.snapshot_figure()
    plt.draw()
    plt.pause(0.01)

    while sim.t<120 and not sim.stop:
        (hi, uj, tvol, ei) = sim.run_until(sim.t+1.0)
        fig=sim.snapshot_figure()
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.01)

