
from ase.spacegroup import crystal
import numpy as np
import ase.io 


# Dictionary containing atomic number with respective element

atomic_name = {"1":"H", "2":"He", "3":"Li", "4":"Be", "5":"B", "6":"C", "7":"N", "8":"O", "9":"F", "10":"Ne",\
               "11":"Na", "12":"Mg", "13":"Al", "14":"Si", "15":"P", "16":"S", "17":"Cl", "18":"Ar", "19": "K",\
               "20":"Ca", "21":"Sc", "22":"Ti", "23":"V", "24":"Cr", "25":"Mn", "26":"Fe", "27":"Co", "28":"Ni",\
               "29":"Cu", "30":"Zn", "31":"Ga", "32":"Ge", "33":"As","34":"Se", "35":"Br","36":"Kr", "37":"Rb",\
               "38":"Sr", "39":"Y", "40":"Zr", "41":"Nb", "42":"Mo", "43":"Tc", "44":"Ru", "45":"Rh", "46":"Pd",\
               "47":"Ag", "48":"Cd", "49":"In", "50":"Sn", "51":"Sb", "52":"Te", "53":"I", "54":"Xe", "55":"Cs", "56":"Ba",\
               "57":"La", "58":"Ce", "59":"Pr", "60":"Nd", "61":"Pm", "62":"Sm", "63":"Eu", "64":"Gd", "65":"Tb", "66":"Dy",\
               "67":"Ho", "68":"Er", "69":"Tm", "70":"Yb", "71":"Lu", "72":"Hf", "73":"Ta", "74":"W", "75":"Re", "76":"Os",\
               "77":"Ir", "78":"Pt", "79":"Au", "80":"Hg", "81":"Tl", "82":"Pb", "83":"Bi", "84":"Po", "85":"At", "86":"Rn",\
               "87":"Fr", "88":"Ra", "89":"Ac", "90":"Th", "91":"Pa", "92":"U", "93":"Np", "94":"Pu"}
               

class InvertPCR :
    
    """ Here we will extract necessary informations from point clound reps (PCR) in order to construct cif file."""
    
    def __init__(self,PCR,z_max,n_sites) :
        self.PCR = PCR
        self.z_max = z_max
        self.n_sites = n_sites
        
    def get_element_matrix(self):
        ele_mat = self.PCR[0:self.z_max,:]
        ele_mat[ele_mat < 0.5] = 0
        return ele_mat
    
    def get_lattice_matrix(self):
        lat_mat = self.PCR[self.z_max:self.z_max+2,:]
        return lat_mat
    
    def get_lattice_parameters(self):
        lattice_matrix = self.get_lattice_matrix()
        lattice_parameters = list(lattice_matrix.flatten())
        a = lattice_parameters[0]
        b = lattice_parameters[1]
        c = lattice_parameters[2]
        alpha = lattice_parameters[3]
        beta = lattice_parameters[4]
        gamma = lattice_parameters[5]
        scaled_par = [a,b,c,alpha,beta,gamma]
        return scaled_par
    
    def get_basis_matrix(self):
        basis_mat = self.PCR[self.z_max+2:self.z_max+2+self.n_sites,:]
        return basis_mat
    
    def get_site_matrix(self):
        site_mat = self.PCR[self.z_max+2+self.n_sites:self.z_max+2+2*self.n_sites,:]
        site_mat[site_mat < 0.5] = 0
        return site_mat

    def get_property_matrix(self):
        prop_mat = self.PCR[self.z_max+2+2*self.n_sites:self.z_max+2+2*self.n_sites+8,:]
        return prop_mat
    
    def get_unique_atomic_numbers(self):
        element_matrix = self.get_element_matrix()
        z_unique = []
        for i in range(3):
            col = list(element_matrix[:,i])
            max_value = max(col)
            if max_value == 0 :
                z_unique.append(0)
            if max_value != 0 :
                z_unique.append(col.index(max_value)+1)
        return z_unique
    
    def get_unique_elements(self):
        z_unique = self.get_unique_atomic_numbers()
        unique_element = []
        for z in z_unique :
            if z == 0 :
                unique_element.append("-")
            else :
                unique_element.append(atomic_name[str(z)])
        return unique_element

    
    def get_atomic_numbers(self):
        element_matrix = self.get_element_matrix()
        site_matrix = self.get_site_matrix()
        z_unique = self.get_unique_atomic_numbers()
        z_total = []
        for i in range(self.n_sites):
            row = list(site_matrix[i,:])
            val = max(row)
            if val == 0 :
                z_total.append(0)
            if val != 0 :
                z_total.append(z_unique[row.index(val)])
        return z_total
    
    def get_elements_basis(self):
        basis = self.get_basis_matrix()
        z_total = self.get_atomic_numbers()
        final_atoms = []
        final_atoms_index = []
        final_elements = []
        for i in range(len(z_total)):
            z = z_total[i]
            if z != 0 :
                final_atoms_index.append(i)
                final_atoms.append(z)
                final_elements.append(atomic_name[str(z)])
        final_basis = basis[final_atoms_index,:]
   
        return final_elements, final_basis

    def get_formula(self) :
        a,_ = self.get_elements_basis()
        elements = [a[0]]
        numbers = []

        for i in range(len(a)) :
            ele = a[i]
            if ele != elements[len(elements)-1] :
                elements.append(ele)
                numbers.append(i)

        numbers.append(len(a))

        final_numbers = []
        final_numbers.append(numbers[0])
        for i in range(1,len(numbers)):
            final_numbers.append(numbers[i]-numbers[i-1])

        formula = elements[0] + str(final_numbers[0])
        for i in range(1,len(elements)) :
            formula += elements[i] + str(final_numbers[i])

        return formula
    
    def get_atoms_object(self):
        symbol = self.get_formula()
        _, basis = self.get_elements_basis()
        lattice = self.get_lattice_parameters()
        atoms = crystal(symbols=symbol,
                        basis=basis,
                        cellpar=lattice)
        return atoms

    def get_distances(self):
        atoms = self.get_atoms_object()
        positions = atoms.get_positions() 
        nele = positions.shape[0]
        distances = []
        for i in range(nele):
            for j in range(i+1,nele):
                pos1 = positions[i,:]
                pos2 = positions[j,:]
                vec = pos1 - pos2
                dist = np.linalg.norm(vec)
                distances.append(np.around(dist,4))
        return distances    

