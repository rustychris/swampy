"""
Created on Fri 24 Mar 2018

@author: Ed Gross, Steve Andrews
@organization: Resource Management Associates
@contact: ed@rmanet.com, steve@rmanet.com
@note: Highly simplified river model. Current assumptions:
    - unsteady and barotropic terms only
    - no subgrid
    - circumcenters inside cell
"""

import time

import h5py
from matplotlib.collections import PolyCollection
from scipy.spatial.distance import euclidean
from stompy.grid import unstructured_grid as ugrid

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

# global variables
dc_min = 1.0
max_sides = 8
dzmin = 0.001
# m2ft = 3.28083
m2ft = 1.0
g = 9.8 * m2ft  # gravity


class SWAMPpy(object):

    def __init__(self, dx, dt, theta, ManN):
        self.dx = dx
        self.dt = dt
        self.theta = theta
        self.ManN = ManN

    def center_vel_perot(self, i, cell_uj):
        """
        Reconstructed cell center velocity
        """
        acu = np.zeros(2, np.float64)
        uside = np.zeros_like(acu)
        ucen = np.zeros_like(acu)
        nsides = len(cell_uj)
        for l in range(nsides):
            j = self.grd.cells[i]['edges'][l]
            for dim in range(2):  # x and y
                uside[dim] = cell_uj[l] * self.en[j, dim]  # u vel component of normal vel
                acu[dim] += uside[dim] * self.len[j] * self.dist[i, l]
        for dim in range(2):  # x and y
            ucen[dim] = acu[dim] / self.grd.cells['_area'][i]

        return ucen

    def get_center_vel(self, uj):
        """
        Get center velocities for every cell in the grid
        """
        ui = np.zeros((self.ncells, 2), np.float64)
        for i in range(self.ncells):
            nsides = self.ncsides[i]
            cell_uj = np.zeros(nsides, np.float64)
            for l in range(nsides):
                j = self.grd.cells[i]['edges'][l]
                cell_uj[l] = uj[j]
            ui[i] = self.center_vel_perot(i, cell_uj)

        return ui

    def get_fu_Perot(self, uj, alpha, sil, hi, hjstar, hjbar, hjtilde):
        """
        Return fu term with Perot method
        """
        fu = np.zeros_like(uj)
        ui = self.get_center_vel(uj)
#         self.write_ui(ui)
        gDtThetaCompl = g * self.dt * (1.0 - self.theta)
        for j in self.intern:
            ii = self.grd.edges[j]['cells']
            iL = ii[0]
            iR = ii[1]

            fu[j] = uj[j]
            if hjbar[j] < dzmin:
                fu[j] -= gDtThetaCompl * (hi[iR] - hi[iL]) / self.dc[j]
            else:
                fu[j] -= gDtThetaCompl * hjtilde[j] / hjbar[j] * (hi[iR] - hi[iL]) / self.dc[j]

            if hjbar[j] < dzmin:  # no advection term if no water at a cell edge
                continue

            # left cell
            sum1 = 0.
            nsides = self.ncsides[iL]
            for l in range(nsides):
                k = self.grd.cells[iL]['edges'][l]
                Q = sil[iL, l] * self.len[k] * hjstar[k] * uj[k]
                if Q >= 0.:  # ignore fluxes out of the cell
                    continue
                iitmp = self.grd.edges[k]['cells']
                i2 = iitmp[np.nonzero(iitmp - iL)[0][0]]  # get neighbor
                ui_norm = ui[i2, 0] * self.en[j, 0] + ui[i2, 1] * self.en[j, 1]
                sum1 += Q * (ui_norm - uj[j])
            fu[j] -= self.dt * alpha[j, 0] * sum1 / (self.grd.cells['_area'][iL] * hjbar[j])
            # right cell
            sum1 = 0.
            nsides = self.ncsides[iR]
            for l in range(nsides):
                k = self.grd.cells[iR]['edges'][l]
                Q = sil[iR, l] * self.len[k] * hjstar[k] * uj[k]
                if Q >= 0.:  # ignore fluxes out of the cell
                    continue
                iitmp = self.grd.edges[k]['cells']
                i2 = iitmp[np.nonzero(iitmp - iR)[0][0]]  # get neighbor
                ui_norm = ui[i2, 0] * self.en[j, 0] + ui[i2, 1] * self.en[j, 1]
                sum1 += Q * (ui_norm - uj[j])
            fu[j] -= self.dt * alpha[j, 1] * sum1 / (self.grd.cells['_area'][iR] * hjbar[j])

        return fu

    def calc_hjstar(self, ei, zi, uj):
        """
        Calculate hjstar from Kramer and Stelling
        """
        hjstar = np.zeros(self.nedges)
        for j in self.intern:
            ii = self.grd.edges[j]['cells']
            if uj[j] > 0:
                e_up = ei[ii[0]]
            elif uj[j] < 0:
                e_up = ei[ii[1]]
            else:
                e_up = max(ei[ii[0]], ei[ii[1]])
            hjstar[j] = min(zi[ii[0]], zi[ii[1]]) + e_up

        return hjstar

    def calc_hjbar(self, ei, zi):
        """
        Calculate hjbar from Kramer and Stelling
        """
        hjbar = np.zeros(self.nedges)
        for j in self.intern:  # loop over internal edges
            ii = self.grd.edges[j]['cells']  # 2 cells
            # esg - could use hi directly instead of recalculating
            hL = ei[ii[0]] + zi[ii[0]]
            hR = ei[ii[1]] + zi[ii[1]]
            hjbar[j] = self.alpha[j][0] * hL + self.alpha[j][1] * hR

        return hjbar

    def calc_hjtilde(self, ei, zi):
        """
        Calculate hjtilde from Kramer and Stelling
        """
        hjtilde = np.zeros(self.nedges)
        for j in self.intern:  # loop over internal edges
            ii = self.grd.edges[j]['cells']  # 2 cells
            eavg = 0.5 * (ei[ii[0]] + ei[ii[1]])
            bL = zi[ii[0]]
            bR = zi[ii[1]]
            hjtilde[j] = eavg + self.alpha[j][0] * bL + self.alpha[j][1] * bR

        return hjtilde

    def calc_volume(self, hi):
        """
        Calculate cell volume give cell total water depth
        """
        vol = np.zeros(self.ncells)
        for i in range(self.ncells):
            vol[i] = self.grd.cells['_area'][i] * hi[i]
        return vol

    def calc_wetarea(self, hi):
        """
        Calculate cell wetted area give cell total water depth
        """
        wa = np.zeros(self.ncells)
        for i in range(self.ncells):
#             if hi[i] > 0.000001:
#                 wa[i] = self.grd.cells['_area'][i]
#             else:
#                 wa[i] = 0.
            wa[i] = self.grd.cells['_area'][i]
        return wa

    def calc_edge_wetarea(self, hj):
        """
        Calculate wetted area for the cell faces
        """
        wa = np.zeros(self.nedges)
        for j in self.intern:
            wa[j] = self.len[j] * hj[j]
        return wa

    def calc_edge_friction(self, uj, aj, hj):
        """
        Calculate friction coef at edge
        Cf = n^2*g*|u|/Rh^(4/3)
        """
        cf = np.zeros(self.nedges)
        n = self.ManN
        for j in self.intern:
            rh = hj[j]  # assume no side wall friction
            if rh < dzmin:
                cf[j] = 0.
            else:
                cf[j] = n * n * g * np.abs(uj[j]) / rh ** (4. / 3.)
        return cf

    def make_1D_grid_Cartesian(self, L, show_grid=False):
        """
        Setup unstructured grid for channel 1 cell wide
        Grid is Cartesian with edge length = self.dx
        """
        ncells = int(L / self.dx)
        npoints = 2 * ncells + 2
        nedges = 3 * ncells + 1
        points = np.zeros((npoints, 2), np.float64)
        # in future if only 3 edges, node4 will have value -1
        cells = -1 * np.ones((ncells, 4), np.int32)  # cell nodes
        edges = -1 * np.ones((nedges, 2), np.int32)  # edge nodes
        for i in range(ncells + 1):  # set node x,y etc.
            points[2 * i, 0] = self.dx * i
            points[2 * i, 1] = 0.0
            points[2 * i + 1, 0] = self.dx * i
            points[2 * i + 1, 1] = self.dx
            if i < ncells:
                cells[i, :] = [2 * (i + 1), 2 * (i + 1) + 1, 2 * i + 1, 2 * i]
                # bottom of cell
                edges[3 * i, :] = [2 * (i + 1), 2 * i]
                # left of cell
                edges[3 * i + 1, :] = [2 * i, 2 * i + 1]
                # top of cell
                edges[3 * i + 2, :] = [2 * i + 1, 2 * (i + 1) + 1]
        # far right hand edge
        edges[3 * ncells, :] = [2 * ncells, 2 * ncells + 1]
        self.grd = ugrid.UnstructuredGrid(edges=edges, points=points,
                                          cells=cells, max_sides=4)
        if show_grid:
            self.grd.plot_edges()
            plt.show()

        return

    def make_2D_grid_Cartesian(self, L, nwide=3, show_grid=False):
        """
        Setup unstructured grid for channel n cells wide
        Grid is Cartesian with edge length = self.dx
        """
        nlong = int(L / self.dx)
        ncells = nlong * nwide
        npoints = (nlong + 1) * (nwide + 1)
        nedges = nwide * (nlong + 1) + (nwide + 1) * nlong
        points = np.zeros((npoints, 2), np.float64)
        cells = -1 * np.ones((ncells, 4), np.int32)  # cell nodes
        edges = -1 * np.ones((nedges, 2), np.int32)  # edge nodes
        ipt = 0
        for i in range(nlong + 1):
            x = self.dx * i
            for j in range(nwide + 1):
                y = self.dx * j
                points[ipt, 0] = x
                points[ipt, 1] = y
                ipt += 1
        icell = 0
        for i in range(nlong):
            for j in range(nwide):
                i1 = icell + i
                i2 = i1 + 1
                i3 = icell + nwide + i + 1
                i4 = i3 + 1
                cells[icell, :] = [i3, i4, i2, i1]
                icell += 1
        iedge = 0
        for i in range(nlong):
            for j in range(nwide):
                icell = i * nwide + j
                i1 = icell + i
                i2 = i1 + 1
                i3 = icell + nwide + i + 1
                i4 = i3 + 1
                if j == 0:
                    edges[iedge, :] = [i1, i3]  # bottom
                    iedge += 1
                edges[iedge, :] = [i1, i2]  # left
                iedge += 1
                edges[iedge, :] = [i2, i4]  # top
                iedge += 1
        i = nlong - 1  # last column right edges
        for j in range(nwide):
            icell = i * nwide + j
            i3 = icell + nwide + i + 1
            i4 = i3 + 1
            edges[iedge, :] = [i3, i4]  # top
            iedge += 1
        self.grd = ugrid.UnstructuredGrid(edges=edges, points=points, cells=cells, max_sides=4)
        if show_grid:
            self.grd.plot_edges()
            plt.show()

        return

    def import_ras_geometry(self, hdf_fname, twod_area_name, max_cell_faces, show_grid=False):

        h = h5py.File(hdf_fname, 'r')

        points_xy = h['Geometry/2D Flow Areas/' + twod_area_name + '/FacePoints Coordinate']
        npoints = len(points_xy)
        points = np.zeros((npoints, 2), np.float64)
        for n in range(npoints):
            points[n, 0] = points_xy[n][0]
            points[n, 1] = points_xy[n][1]

        edge_nodes = h['Geometry/2D Flow Areas/' + twod_area_name + '/Faces FacePoint Indexes']
        nedges = len(edge_nodes)
        edges = -1 * np.ones((nedges, 2), dtype=int)
        for j in range(nedges):
            edges[j][0] = edge_nodes[j][0]
            edges[j][1] = edge_nodes[j][1]

        cell_nodes = h['Geometry/2D Flow Areas/' + twod_area_name + '/Cells FacePoint Indexes']
        for i in range(len(cell_nodes)):
            if cell_nodes[i][2] < 0:  # first ghost cell (which are sorted to end of list)
                break
        ncells = i  # don't count ghost cells
        cells = -1 * np.ones((ncells, max_cell_faces), dtype=int)
        for i in range(ncells):
            for k in range(max_cell_faces):
                cells[i][k] = cell_nodes[i][k]

        cell_center_xy = h['Geometry/2D Flow Areas/' + twod_area_name + '/Cells Center Coordinate']
        self.ras_xf = np.array([cell_center_xy[i][0] for i in range(ncells)])
        self.ras_yf = np.array([cell_center_xy[i][1] for i in range(ncells)])

        self.grd = ugrid.UnstructuredGrid(edges=edges, points=points,
                                          cells=cells, max_sides=max_cell_faces)

        if show_grid:
            self.grd.plot_edges()
            ax = plt.gca()
            ax.plot(self.ras_xf, self.ras_yf, 'rs', ms=3, mfc='none')
            plt.show()

        return

    def set_topology(self, grid_type):
        """
        Use functions of unstructured grid class for remaining topology
        """
        self.nedges = len(self.grd.edges)
        self.ncells = len(self.grd.cells)
        self.nnodes = len(self.grd.nodes)
        self.grd.update_cell_edges()
        self.grd.update_cell_nodes()
        self.grd.edge_to_cells()
        self.grd.cells_area()

#         if grid_type == 'RAS':
#             centers = np.zeros((self.ncells, 2), 'f8') * np.nan
#             for i in range(self.ncells):
#                 centers[i] = [self.ras_xf[i], self.ras_yf[i]]
#             self.grd.cells['_center'] = centers
#         else:
        self.grd.cells['_center'] = self.grd.cells_centroid()

        self.grd.edges['mark'] = 0  # default is internal cell
        self.extern = np.where(np.min(self.grd.edges['cells'], axis=1) < 0)[0]
        self.grd.edges['mark'][self.extern] = 1  # boundary edge
        self.intern = np.where(self.grd.edges['mark'] == 0)[0]
        self.nedges_intern = len(self.intern)  # number of internal edges
        self.exy = self.grd.edges_center()
        self.en = self.grd.edges_normals()
        self.len = self.grd.edges_length()
        # number of valid sides for each cell
        self.ncsides = np.asarray([sum(jj >= 0) for jj in self.grd.cells['edges']])
        self.dist = self.edge_to_cen_dist()

        return

    def get_cell_center_spacings(self):
        """
        Return cell center spacings
        Spacings for external edges are set to dc_min (should not be used)
        """
        dc = dc_min * np.ones(self.nedges, np.float64)
        for j in self.intern:
            ii = self.grd.edges[j]['cells']  # 2 cells
            xy = self.grd.cells[ii]['_center']  # 2x2 array
            dc[j] = euclidean(xy[0], xy[1])  # from scipy
        self.dc = dc

        return dc

    def edge_to_cen_dist(self):
        """
        Return cell edge to center distance for interpolations
        """
        dist = np.zeros((self.ncells, max_sides), np.float64)
        for i in range(self.ncells):
            cen_xy = self.grd.cells['_center'][i]  # centroid now
            nsides = self.ncsides[i]
            for l in range(nsides):
                j = self.grd.cells[i]['edges'][l]
#                 side_xy = self.exy[j]
#                 dist[i, l] = euclidean(side_xy, cen_xy)  # from scipy
                # Find intersection between line connecting centers and edge
                nodes = self.grd.edges['nodes'][j]
                p1x = self.grd.nodes['x'][nodes[0]][0]
                p1y = self.grd.nodes['x'][nodes[0]][1]
                p2x = self.grd.nodes['x'][nodes[1]][0]
                p2y = self.grd.nodes['x'][nodes[1]][1]
                p3x = cen_xy[0]
                p3y = cen_xy[1]
                cs = self.grd.edges['cells'][j]
                if cs[0] == i:
                    neigh = cs[1]
                else:
                    neigh = cs[0]
                if neigh < 0:
                    p4x = self.exy[j][0]
                    p4y = self.exy[j][1]
                else:
                    neigh_xy = self.grd.cells['_center'][neigh]
                    p4x = neigh_xy[0]
                    p4y = neigh_xy[1]
                intersect = self.line_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)
                dist[i, l] = euclidean(intersect, cen_xy)  # from scipy

        return dist

    def line_intersect(self, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
        s1x = p2x - p1x
        s1y = p2y - p1y
        s2x = p4x - p3x
        s2y = p4y - p3y
        s = (-s1y * (p1x - p3x) + s1x * (p1y - p3y)) / (-s2x * s1y + s1x * s2y)
        t = (s2x * (p1y - p3y) - s2y * (p1x - p3x)) / (-s2x * s1y + s1x * s2y)
        # IF (s>=0D0 .AND. s<=1D0 .AND. t>=0D0 .AND. t<=1D0) THEN
        # success=.TRUE.
        intersect = np.zeros(2)
        intersect[0] = p1x + (t * s1x)
        intersect[1] = p1y + (t * s1y)
        return intersect

    def get_alphas_Perot(self):
        """
        Return Perot weighting coefficients
        Coefficients for external edges are set to zero (should not be used)
        """
        alpha = np.zeros((self.nedges, 2), np.float64)
        for j in self.intern:
            ii = self.grd.edges[j]['cells']
            cen_xyL = self.grd.cells['_center'][ii[0]]
            cen_xyR = self.grd.cells['_center'][ii[1]]
            nodes = self.grd.edges['nodes'][j]
            p1x = self.grd.nodes['x'][nodes[0]][0]
            p1y = self.grd.nodes['x'][nodes[0]][1]
            p2x = self.grd.nodes['x'][nodes[1]][0]
            p2y = self.grd.nodes['x'][nodes[1]][1]
            p3x = cen_xyL[0]
            p3y = cen_xyL[1]
            p4x = cen_xyR[0]
            p4y = cen_xyR[1]
            intersect = self.line_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)
#             side_xy = self.exy[j]
#             alpha[j, 0] = euclidean(side_xy, cen_xyL) / self.dc[j]
#             alpha[j, 1] = euclidean(side_xy, cen_xyR) / self.dc[j]
            alpha[j, 0] = euclidean(intersect, cen_xyL) / self.dc[j]
            alpha[j, 1] = euclidean(intersect, cen_xyR) / self.dc[j]
        self.alpha = alpha

        return alpha

    def get_sign_array(self):
        """
        Sign array for all cells, for all edges
        Positive if face normal is pointing out of cell
        """
        sil = np.zeros((self.ncells, max_sides), np.float64)
        for i in range(self.ncells):
            for l in range(self.ncsides[i]):
                j = self.grd.cells[i]['edges'][l]
                ii = self.grd.edges[j]['cells']  # 2 cells
                sil[i, l] = (ii[1] - 2 * i + ii[0]) / (ii[1] - ii[0])

        return sil

    def get_side_num_of_cell(self):
        """
        Sparse matrix of local side number (0,1,2,or 3) for each cell, for each edge
        """
        lij = -1 * np.ones((self.ncells, self.nedges), np.int32)
        for i in range(self.ncells):
            for l in range(self.ncsides[i]):
                j = self.grd.cells[i]['edges'][l]
                lij[i, j] = l

        return lij

    def set_grid(self, case, domain_length=100., grid_type='1D_Cartesian', show_grid=False):
        """
        Set up grid for simulation
        """
        self.domain_length = domain_length
        self.grid_type = grid_type
        if 'dam_break' in case:
            if grid_type == '1D_Cartesian':
                self.make_1D_grid_Cartesian(domain_length, show_grid=show_grid)
            elif grid_type == '2D_Cartesian':
                self.make_2D_grid_Cartesian(domain_length, show_grid=show_grid)
            else:
                raise NotImplementedError('Grid type ' + grid_type + ' not supported for this test case')
        elif 'bump' in case:
            if grid_type == '1D_Cartesian':
                self.make_1D_grid_Cartesian(domain_length, show_grid=show_grid)
            elif grid_type == '2D_Cartesian':
                self.make_2D_grid_Cartesian(domain_length, show_grid=show_grid)
            else:
                raise NotImplementedError('Grid type ' + grid_type + ' not supported for this test case')
        elif 'U-chan' in case:
            if grid_type == 'RAS':
                self.import_ras_geometry(r'J:\work_2018\RAS-2D\final_alt_solver_test_cases\VV-Lab-180DegreeBend\Bend180.g06.hdf',
                                         'Bend 2D area', 5, show_grid=False)
            else:
                raise NotImplementedError('Grid type ' + grid_type + ' supported for this test case')

        # compute topology using unstructured_grid class methods
        self.set_topology(grid_type)
        print 'ncells', self.ncells

        return

    def set_initial_conditions(self, case, us_eta=0., ds_eta=0., us_q=0.):
        """
        Set up initial conditions for simulation
        """
        # set initial conditions
        self.ic_ei = np.zeros(self.ncells, np.float64)
        self.ic_zi = np.zeros_like(self.ic_ei)

        # set initial conditions - dam break
        if case == 'dam_break':
            self.ds_eta = ds_eta
            self.ic_ei[:] = ds_eta  # downstream wse
            # Dam break propagating left to right
            ileft = np.where(self.grd.cells['_center'][:, 0] < self.domain_length / 2.)[0]
            self.ic_ei[ileft] = us_eta
            # Dam break propagating right to left
#             iright = np.where(self.grd.cells['_center'][:, 0] > domain_length / 2.)[0]
#             self.ic_ei[iright] = us_eta

        # set initial conditions - bump
        elif 'bump' in case:
            self.ds_eta = ds_eta
            self.qinflow = us_q
            self.ic_ei[:] = ds_eta
            self.ds_i = np.where(self.grd.cells['_center'][:, 0] > (self.domain_length - self.dx))[0]
            self.us_i = np.where(self.grd.cells['_center'][:, 0] < (0. + self.dx))[0]
            self.us_j = np.where(self.exy[:, 0] < (0. + self.dx / 4.))[0]
            self.us_jlen = self.len[self.us_j]
            for i in range(self.ncells):
                xtmp = self.grd.cells['_center'][i, 0]
                if xtmp >= 8.*m2ft and xtmp <= 12.*m2ft:
                    self.ic_zi[i] = -(0.2 - 0.05 * (xtmp - 10.) ** 2) * m2ft

        # set initial conditions - U channel
        elif 'U-chan' in case:
            self.ds_eta = ds_eta
            self.qinflow = us_q
            self.ic_ei[:] = ds_eta
            self.ic_zi[:] = 0.
            self.ds_i = np.where(self.grd.cells['_center'][:, 0] < -1.98)[0]
            self.us_i = np.where((self.grd.cells['_center'][:, 0] < 0.0275) & (self.grd.cells['_center'][:, 1] < 1.))[0]
            self.us_j = np.where((self.exy[:, 0] < 0.01) & (self.exy[:, 1] < 1.))[0]
            self.us_jlen = self.len[self.us_j]

        return

    def set_uj_Uchan(self):
        uj = np.zeros(self.nedges, np.float64)
        vmag = 0.25
        avec = np.array([0., -1.])
        for j in self.intern:
            edgex = self.exy[j][0]
            edgey = self.exy[j][1]
            if edgex <= 1.:
                if edgey < 1.2:
                    # velocity in positive x direction
                    unit_v = np.array([1., 0])
                else:
                    # velocity in negative x direction
                    unit_v = np.array([-1., 0.])
            else:
                bvec = self.exy[j] - np.array([1., 1.2])
                cos_theta = np.dot(avec, bvec) / (np.linalg.norm(avec) * np.linalg.norm(bvec))
                th = np.arccos(cos_theta)
                unit_v = np.array([np.cos(th), np.sin(th)])
            uj[j] = vmag * np.dot(self.en[j], unit_v)

#         f = open('uj_python.txt', 'w')
#         for j in range(len(uj)):
#             outstr = str(self.exy[j][0]) + ',' + str(self.exy[j][1]) + ',' + str(uj[j]) + '\n'
#             f.write(outstr)
#         f.close()
        return uj

    def run(self, case, tend, is_animate):
        """
        Run simulation
        """
        dt = self.dt
        self.tend = tend
        nsteps = np.int(np.round(tend / dt))
        xy = self.get_grid_poly_collection()

        # precompute constants
        gdt2theta2 = g * dt * dt * self.theta * self.theta
        dc = self.get_cell_center_spacings()
        alpha = self.get_alphas_Perot()
        sil = self.get_sign_array()
        # len_dc_ratio = np.divide(self.len, dc)  # edge length divided by center spacing

        # cell center values
        ncells = self.ncells
        hi = np.zeros(ncells, np.float64)  # total depth
        zi = np.zeros_like(hi)  # bed elevation, measured positive down
        ei = np.zeros_like(hi)  # water surface elevation
        vi = np.zeros_like(hi)  # cell volumes
        pi = np.zeros_like(hi)  # cell wetted areas
        # edge values
        uj = np.zeros(self.nedges, np.float64)  # normal velocity at side
#         uj = self.set_uj_Uchan()
        aj = np.zeros(self.nedges, np.float64)  # edge wet areas
        cf = np.zeros(self.nedges, np.float64)  # edge friction coefs
        cfterm = np.zeros(self.nedges, np.float64)  # edge friction coefs - term for matrices
        # Matrices and rhs
        Ai = np.zeros((ncells, ncells), np.float64)  # inner iterations
        bi = np.zeros(ncells, np.float64)
        Ao = np.zeros((ncells, ncells), np.float64)  # outer iterations
        bo = np.zeros(ncells, np.float64)
        x0 = np.zeros_like(hi)

        # set initial conditions
        ei[:] = self.ic_ei[:]
        zi[:] = self.ic_zi[:]

        hi = zi + ei  # update total depth
        hjstar = self.calc_hjstar(ei, zi, uj)
        hjbar = self.calc_hjbar(ei, zi)
        hjtilde = self.calc_hjtilde(ei, zi)
        vi = self.calc_volume(hi)
        pi = self.calc_wetarea(hi)
        aj = self.calc_edge_wetarea(hjbar)
        cf = self.calc_edge_friction(uj, aj, hjbar)
        cfterm = 1. / (1. + self.dt * cf[:])

        # time stepping loop
        for n in range(nsteps):
            print 'step', n + 1, 'of', nsteps

            # calculate advection term
            fu = self.get_fu_Perot(uj, alpha, sil, hi, hjstar, hjbar, hjtilde)
#             self.write_fu(fu)
#             self.write_facedist()
#             self.write_centroid()
#             self.write_len()

            # set matrix coefficients
            # first calculate *usual* coefficient matrix and rhs
            Ai[:, :] = 0.
            bi[:] = 0.
            for i in range(ncells):
                sum1 = 0.0
                for l in range(self.ncsides[i]):
                    j = self.grd.cells[i]['edges'][l]
                    if self.grd.edges[j]['mark'] == 0:  # if internal
                        sum1 += sil[i, l] * aj[j] * (self.theta * cfterm[j] * fu[j] +
                                                     (1. - self.theta) * uj[j])
                        if hjbar[j] > dzmin:
                            hterm = hjtilde[j] / hjbar[j]
                        else:
                            hterm = 1.0
#                         hterm = 1.0
                        coeff = gdt2theta2 * aj[j] * cfterm[j] / dc[j] * hterm
                        Ai[i, i] += coeff
                        ii = self.grd.edges[j]['cells']  # 2 cells
                        i2 = ii[np.nonzero(ii - i)[0][0]]  # i2 = index of neighbor
                        Ai[i, i2] = -coeff
                bi[i] = vi[i] - dt * sum1

            # Make additions to form outer (correction) matrix
            for iter in range(10):

                Ao[:, :] = Ai[:, :]
                for i in range(ncells):
                    Ao[i, i] += pi[i]
                bo[:] = 0.
                for i in range(ncells):
                    bo[i] = vi[i] + np.dot(Ai[i, :], ei[:]) - bi[i]

                if 'bump' in case or 'U-chan' in case:
                    for idx in range(len(self.ds_i)):
                        i = self.ds_i[idx]
                        Ao[i, :] = 0.
                        Ao[i, i] = 1.
                        bo[i] = 0.  # no correction
                    for idx in range(len(self.us_i)):
                        i = self.us_i[idx]
                        bo[i] = bo[i] - dt * self.qinflow * self.us_jlen[idx]

                # invert matrix
                start = time.time()
                ei_corr, success = sparse.linalg.cg(Ao, bo, x0=x0, tol=1.e-16)  # Increase tolerance for faster speed
                end = time.time()
                print 'matrix solve took', '{0:0.4f}'.format(end - start), 'sec'
                if success != 0:
                    raise RuntimeError('Error in convergence of conj grad solver')

                ei = ei - ei_corr
                hi = zi + ei
                pi = self.calc_wetarea(hi)
                vi = self.calc_volume(hi)
                rms = np.sqrt(np.sum(np.multiply(ei_corr, ei_corr)) / self.ncells)
                if rms < 0.01:
                    break

            # substitute back into u solution
            for j in self.intern:  # loop over internal cells
                ii = self.grd.edges[j]['cells']  # 2 cells
#                 if hjbar[j] > dzmin:
#                     hterm = hjtilde[j] / hjbar[j]
#                 else:
#                     hterm = 1.0
                hterm = 1.0
                term = g * dt * self.theta * (ei[ii[1]] - ei[ii[0]]) / dc[j]
                uj[j] = cfterm[j] * (fu[j] - term * hterm)

            # Update elevations
            hjstar = self.calc_hjstar(ei, zi, uj)
            hjbar = self.calc_hjbar(ei, zi)
            hjtilde = self.calc_hjtilde(ei, zi)
            vi = self.calc_volume(hi)
            pi = self.calc_wetarea(hi)
            aj = self.calc_edge_wetarea(hjstar)
            cf = self.calc_edge_friction(uj, aj, hjbar)
            cfterm = 1. / (1. + self.dt * cf[:])

            # Animation frames
            if is_animate:
                if 'bump' in case and n % 10 == 0:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(self.grd.cells['_center'][:, 0], ei, 'r.', ms=2, label='model')
                    ax.plot(self.grd.cells['_center'][:, 0], -self.ic_zi, '-k', label='bed')
                    ax.set_ylabel('Water Surface Elevation, m')
                    ax.legend(loc='upper right')
                    plt.savefig(case + '_' + str(n) + '.png', bbox_inches='tight')
                    plt.close('all')
                elif 'U-chan' in case and n % 10 == 0:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    collection = PolyCollection(xy, cmap=plt.get_cmap('jet'))  # 'Blues' 'jet' 'ocean' mcm
                    ui = self.get_center_vel(uj)
                    uimag = np.sqrt(np.multiply(ui[:, 0], ui[:, 0]) + np.multiply(ui[:, 1], ui[:, 1]))
                    collection.set_array(uimag)
                    collection.set_linewidths(0.2)
                    collection.set_edgecolors('none')
                    collection.set_clim(vmin=0.15, vmax=0.45)
                    ax.add_collection(collection)
                    ax.set_xlim([-2.2, 2.5])
                    ax.set_ylim([-0.1, 2.6])
                    ax.set_aspect('equal')
                    plt.savefig(case + '_' + str(n) + '.png', bbox_inches='tight')
                    plt.close('all')

        return hi, uj, ei

    def get_edge_x_vel(self, uj):
        """
        Return x direction velocity and location at cell edge midpoints
        """
        xe = np.zeros(self.nedges_intern)
        uxe = np.zeros_like(xe)
        for idx, j in enumerate(self.intern):
            xe[idx] = self.exy[j][0]  # x coordinate
            uxe[idx] = uj[j] * self.en[j][0]

        return xe, uxe

    def get_xsect_avg_val(self, eta_i):
        """
        Return cross-sectionally averaged x velocities for non-uniform grids
        """
        xc = []
        uxc = []
        nx = self.domain_length / self.dx
        for i in range(int(nx)):
            startx = float(i) * self.dx
            endx = float(i + 1) * self.dx
            midx = (float(i) + 0.5) * self.dx
            d = (self.grd.cells['_center'][:, 0] > startx) & (self.grd.cells['_center'][:, 0] < endx)
            if np.sum(d) > 0:
                utmp = np.mean(eta_i[d])
                uxc.append(utmp)
                xc.append(midx)
        return (xc, uxc)

    def get_grid_poly_collection(self):
        """
        Return list of polygon points useful for 2D plotting
        """
        xy = []
        for i in range(self.ncells):
            tmpxy = []
            for l in range(self.ncsides[i]):
                n = self.grd.cells[i]['nodes'][l]
                tmp = self.grd.nodes[n]['x']
                tmpxy.append([tmp[0], tmp[1]])
            n = self.grd.cells[i]['nodes'][0]
            tmp = self.grd.nodes[n]['x']
            tmpxy.append([tmp[0], tmp[1]])
            xy.append(tmpxy)

        return xy

    def write_fu(self, fu):
        f = open('Fu_python.txt', 'w')
        for j in range(len(fu)):
            f.write(str(fu[j]) + '\n')
        f.close()

    def write_ui(self, ui):
        f = open('ui_python.txt', 'w')
        for celluv in ui:
            f.write(str(celluv[0]) + ',' + str(celluv[1]) + '\n')
        f.close()

    def write_facedist(self):
        f = open('fd_python.txt', 'w')
        for i in range(self.ncells):
            nsides = self.ncsides[i]
            outstr = ''
            for l in range(nsides):
                outstr += str(self.dist[i, l]) + ','
            outstr = outstr[:-1] + '\n'
            f.write(outstr)
        f.close()

    def write_centroid(self):
        f = open('centroid_python.txt', 'w')
        for centroid in self.grd.cells['_center']:
            f.write(str(centroid[0]) + ',' + str(centroid[1]) + '\n')
        f.close()

    def write_len(self):
        f = open('len_python.txt', 'w')
        for j in range(len(self.len)):
            f.write(str(self.len[j]) + '\n')
        f.close()


def get_dry_dam_break_analytical_soln(domain_length, upstream_height, tend):
    """
    Return x, h, u for analytical dry dam break solution
    Solution from Delestre et al, 2012 4.1.2
    dx here just applies to resolution of analytical solution
    """
    dx = 1.0
    x0 = domain_length / 2.0
    t = tend
    sgh = np.sqrt(g * upstream_height)
    xA = x0 - t * sgh
    xB = x0 + 2.0 * t * sgh
    imx = np.int(domain_length / dx)
    xj = np.asarray([dx * float(i) for i in range(imx + 1)])
    xi = 0.5 * (xj[:-1] + xj[1:])
    han = np.zeros(imx, np.float64)
    uan = np.zeros_like(han)

    for i in range(imx):
        if xi[i] < xA:
            han[i] = upstream_height
            uan[i] = 0.0
        elif xi[i] > xB:
            han[i] = 0.0
            uan[i] = 0.0
        else:
            xt = xi[i] - x0
            han[i] = (4.0 / (9.0 * g)) * (sgh - xt / (2.0 * t)) ** 2
            uan[i] = (2.0 / 3.0) * (xt / t + sgh)

    return (xi, han, uan)


def get_bump_observed():
    """
    Read in observed data from lab experiment
    """
    fname = r"\\raid01\data9\RAS_2D\Observed_Data\bump_analytical_soln.csv"
    x = []
    y = []
    count = 0
    for line in open(fname, 'r'):
        if not line:
            break
        count += 1
        if count < 2:  # header line
            continue
        sline = line.split(',')
        x.append(float(sline[0]))
        y.append(float(sline[1]))
    return (np.array(x), np.array(y))


if __name__ == '__main__':

    case = 'U-chan'  # dam_break , bump , U-chan

    # Flow over rounded bump test case
    if case == 'bump':
        dx = 0.05 * m2ft
        dt = 0.04  # stable with 0.1 for dx = 0.1 for 1D cartesian, 0.05 for 2D cartesian
        tend = 100.
        theta = 1.0
        ManN = 0.000001
        grid_type = '2D_Cartesian'
        domain_length = 20. * m2ft
        upstream_flow = 0.18 * m2ft * m2ft  # m2/s
        downstream_eta = 0.33 * m2ft
    # Dry dam break case
    elif case == 'dam_break':
        dx = 5.0
        dt = 0.1  # stable with 0.1 for dx = 5, 0.05 with dx = 2
        tend = 30.
        theta = 1.0
        ManN = 0.0
        domain_length = 1200.0
        grid_type = '2D_Cartesian'  # 1D_Cartesian, 2D_Cartesian
        upstream_height = 10.0
        downstream_height = 0.
    elif case == 'U-chan':
        dx = 0.05
        dt = 0.01
        tend = 60.
        theta = 1.0
        ManN = 0.01
        grid_type = 'RAS'
        domain_length = 0.  # Not used
        upstream_flow = 0.0123 / 0.8  # m2/s
        downstream_eta = 0.057

    # Initialize, set grid, set ICs
    swampy = SWAMPpy(dx, dt, theta, ManN)
    swampy.set_grid(case, domain_length=domain_length, grid_type=grid_type, show_grid=False)
    if 'bump' in case or 'U-chan' in case:
        swampy.set_initial_conditions(case, ds_eta=downstream_eta, us_q=upstream_flow)
    elif case == 'dam_break':
        swampy.set_initial_conditions(case, us_eta=upstream_height, ds_eta=downstream_height)

    # Run
    (hi, uj, ei) = swampy.run(case, tend, is_animate=True)  # returns final water surface elevation, velocity

    # Extract results and plot
    ui = swampy.get_center_vel(uj)
    (xe, uxe) = swampy.get_edge_x_vel(uj)
    (xc, uxc) = swampy.get_xsect_avg_val(ui[:, 0])
    (xc, ec) = swampy.get_xsect_avg_val(ei)

    if case == 'dam_break':

        # dry dam break analytical solution
        (xan, han, uan) = get_dry_dam_break_analytical_soln(domain_length, upstream_height, tend)

        # Analytical comparison plot
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(xan, han, '-b', label='analytical', linewidth=1.5)
        ax1.plot(xc, ec, 'r.', ms=2, label='model')
        ax1.set_ylabel('Depth (m)')
        ax1.legend(loc='upper right')
        ax2 = fig.add_subplot(212)
        ax2.plot(xan, uan, '-b', label='analytical', linewidth=1.5)
        ax2.plot(xc, uxc, 'r.', ms=2, label='model')
        ax2.set_ylabel('Current Velocity (m/s)')
        ax2.set_xlabel('Distance (m)')
        ax2.legend(loc='upper left')

    elif 'bump' in case:

        (xan, han) = get_bump_observed()

        # Comparison plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xan * m2ft, han * m2ft, '-b', label='observed', linewidth=1.5, zorder=5)
        ax.plot(swampy.grd.cells['_center'][:, 0], ei, 'r.', ms=4, label='model', zorder=5)  # , linewidth=1.5)
        ax.fill(swampy.grd.cells['_center'][:, 0], -swampy.ic_zi, '0.7', label='bed', zorder=5)
        ax.plot(swampy.grd.cells['_center'][:, 0], -swampy.ic_zi, 'k-', zorder=5)
        ax.grid(color='0.8', linestyle='-', zorder=3)
        ax.set_ylabel('Water Surface Elevation, m')
#         ax.set_ylim([-1.2, 1.2])
        ax.legend(loc='center left')

    elif 'U-chan' in case:

        print 'Complete'

    plt.show()
    print 'Complete!'
