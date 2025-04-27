import numpy as np
from ase.io import read
from mace.calculators.mace import MACECalculator
from ase.build import make_supercell


def get_lattice_energy(model,sol,mol,nmol):
# read the geometry of the solid and the gas phase 
# and return the lattice energy computed with 'model'
# as energy_solid/nmol - energy_gas
    sol.calc=model
    mol.calc=model
    energy_solid=sol.get_potential_energy()
    energy_molecule=mol.get_potential_energy()
    return energy_solid/nmol-energy_molecule

for system in ["01_cyclohexanedione",
 "02_acetic_acid",
 "03_adamantane",
 "04_ammonia",
 "05_anthracene",
 "06_benzene",
 "07_co2",
 "08_cyanamide",
 "09_cytosine",
 "10_ethylcarbamate",
 "11_formamide",
 "12_imidazole",
 "13_naphthalene",
 "14_oxalic_acid_alpha",
 "15_oxalic_acid_beta",
 "16_pyrazine",
 "17_pyrazole",
 "18_triazine",
 "19_trioxane",
 "20_uracil",
 "21_urea",
 "22_hexamine",
 "23_succinic_acid"]:

    mol=read(str(system)+'/POSCAR_molecule','0')
    sol=read(str(system)+'/POSCAR_solid','0')
    ref=np.loadtxt(str(system)+'/lattice_energy_DMC')
    nmol=np.loadtxt(str(system)+'/nmol')

    model_paths=[]
    model_paths.append('/scratch/snx3000/fdellapi/MolCrys/X23_VASP/models/full_qne/models/REPULSION-2/'+str(system)+'/MACE_model_swa.model')

    models=[]
    models.append(MACECalculator(model_paths[0], default_dtype='float64',device='cuda'))


    ev_to_kjmol=96.485
    for i in range(len(models)):
        lat=get_lattice_energy(models[i],sol,mol,nmol)
        lat *= ev_to_kjmol # convert to kJ/mol
        print(i, lat, lat-ref[0])  # print the name of the system, the lattice energy according to the MACE model and the difference with DMC
