import ase
import copy
import numpy as np
from tqdm import tqdm

class GridDescriptor:
    def __init__(self, grids, acsf, cluster=None, environ=None):
        self.grids = grids
        self.acsf = acsf
        self.cluster = cluster
        self.environ = environ
    
    def get_descriptor(self, atoms, coord, pseudo_atom_index):
        acsf = self.acsf

        atoms[pseudo_atom_index].position = coord
        acsf_single = acsf.create(atoms, centers=[pseudo_atom_index])

        return acsf_single

    def get_nn_snn(self, atoms, coord, pseudo_atom_index):

        atoms[pseudo_atom_index].position = coord
        indices = list(range(len(atoms)))
        dists = atoms.get_distances(pseudo_atom_index, indices, mic=True, vector=False)
        vecs = atoms.get_distances(pseudo_atom_index, indices, mic=True, vector=True)
        sorted_incdices = np.argsort(dists)

        return dists[sorted_incdices[1:3]], vecs[sorted_incdices[1:3]]

    def get_nn_snns(self):
        cluster_with_pseudo = copy.deepcopy(self.cluster)
        environ_with_pseudo = copy.deepcopy(self.environ)
        # create the pseudo atoms
        symbol = cluster_with_pseudo[0].symbol
        pseudo_atom = ase.Atom(symbol, [0,0,0])
        cluster_with_pseudo.append(pseudo_atom)

        symbol = environ_with_pseudo[0].symbol
        pseudo_atom = ase.Atom(symbol, [0,0,0])
        environ_with_pseudo.append(pseudo_atom)
        cluster_index = len(cluster_with_pseudo) - 1
        environ_index = len(environ_with_pseudo) - 1
        all_dists, all_vecs = [], []
        for grid in tqdm(self.grids.reshape((-1, 3))):
            cluster_dists, cluster_vecs = self.get_nn_snn(cluster_with_pseudo, grid, pseudo_atom_index=cluster_index)
            environ_dists, environ_vecs = self.get_nn_snn(environ_with_pseudo, grid, pseudo_atom_index=environ_index)
            all_dists.append( np.hstack((cluster_dists, environ_dists),) )
        all_dists = np.array(all_dists)
        return all_dists



    def get_descriptors(self):
        cluster_with_pseudo = copy.deepcopy(self.cluster)
        environ_with_pseudo = copy.deepcopy(self.environ)
        # create the pseudo atoms
        symbol = cluster_with_pseudo[0].symbol
        pseudo_atom = ase.Atom(symbol, [0,0,0])
        cluster_with_pseudo.append(pseudo_atom)

        symbol = environ_with_pseudo[0].symbol
        pseudo_atom = ase.Atom(symbol, [0,0,0])
        environ_with_pseudo.append(pseudo_atom)
        cluster_index = len(cluster_with_pseudo) - 1
        environ_index = len(environ_with_pseudo) - 1
        
        all_descriptors = []
        for grid in tqdm(self.grids.reshape((-1, 3))):
            cluster_descriptor = self.get_descriptor(cluster_with_pseudo, grid, pseudo_atom_index=cluster_index).reshape((-1,))
            environ_descriptor = self.get_descriptor(environ_with_pseudo, grid, pseudo_atom_index=environ_index).reshape((-1,))
            all_descriptor = np.hstack((cluster_descriptor, environ_descriptor))
            all_descriptors.append(all_descriptor)
        all_descriptors = np.array(all_descriptors)

        return all_descriptors
            
