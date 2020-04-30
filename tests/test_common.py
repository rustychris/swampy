import numpy as np

def calc_Fr_CFL_Bern(sim):
    # To make sure we actually got a hydraulic jump
    e2c=sim.grd.edge_to_cells()
    u=sim.uj[sim.intern]
    x=sim.grd.edges_center()[sim.intern][:,0]

    ui=sim.get_center_vel(sim.uj)
    xi=sim.grd.cells_center()[:,0]
    
    ej=sim.cell_to_edge_upwind(sim.ei,sim.uj)[sim.intern]
    vj=sim.cell_to_edge_upwind(sim.vi,sim.uj)[sim.intern]
    
    hj=ej - sim.zj_agg['min'][sim.intern]
    
    Fr=u/np.sqrt(9.8*hj)
    CFL=u*sim.aj[sim.intern]*sim.dt/vj
    phi=u**2/(2*9.8) + ej

    return dict(x=x,xi=xi,Fr=Fr,CFL=CFL,phi=phi,u=u,
                h=hj,ui=ui[:,0])

