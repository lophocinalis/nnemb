import ase
import ase.io
import numpy as np
import gc

class EXTPOT:
    def __init__(self, filename):
        self.filename = filename

    def parse(self):
        filename = self.filename        
        with open(filename) as fp:
            lines = fp.readlines()
        shape = np.array(lines[0].split(), dtype=np.int64)
        elements = np.loadtxt(lines[1:])
        elements = elements.reshape(shape[::-1])
        elements = elements.swapaxes(0, 2)
        #print(elements.shape)
        gc.collect()
        return shape, elements

class Grid:
    def __init__(self, filename, gridshape):
        self.filename = filename
        self.gridshape = gridshape
    
    def setup_grid(self):
        atoms = ase.io.read(self.filename)
        cell = atoms.get_cell()
        gridshape = self.gridshape
        acompgrid = np.linspace(0, 1 * (gridshape[0] - 1) / gridshape[0], gridshape[0])
        bcompgrid = np.linspace(0, 1 * (gridshape[1] - 1) / gridshape[1], gridshape[1])
        ccompgrid = np.linspace(0, 1 * (gridshape[2] - 1) / gridshape[2], gridshape[2])
        acomp, bcomp, ccomp = np.meshgrid(acompgrid, bcompgrid, ccompgrid, indexing='ij')
        agrid = acomp.reshape((gridshape[0], gridshape[1], gridshape[2], 1)) * cell[0].reshape((1,1,1,3))
        bgrid = bcomp.reshape((gridshape[0], gridshape[1], gridshape[2], 1)) * cell[1].reshape((1,1,1,3))
        cgrid = ccomp.reshape((gridshape[0], gridshape[1], gridshape[2], 1)) * cell[2].reshape((1,1,1,3))
        grid = agrid + bgrid + cgrid
        self.grid = grid
        

if __name__ == '__main__':
        extpot = EXTPOT('EXTPOT.final')
        shape, elements = extpot.parse()
        grid = Grid('POSCAR.all', shape)
        grid.setup_grid()
