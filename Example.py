import shutil
import joblib
import GA
import numpy as np#
import selfies as sf
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
import os
import json
import math
import rdkit
import h5py

def mock_fitness_ID(speciman_list, paths, Dummy_File):
    """
    Mock fitness function: Returns a fitness function according to the negative value of the ID of each ligand. Pure testing purposes.

    Parameters
    ----------
    speciman_list : LIST
        List of speciman to be evaluated.
    paths : LIST
        Contains the paths for work,final storage etc. Mandatory for all fitness functions.
    Dummy_File : STRING
        Path to a dummy xyz file that is copied into the hdf5 file to test functionality.

    """
    charges=[2]*len(speciman_list)
    multiplicities=[4]*len(speciman_list)
    D_values=[-int(pop.ID) for pop in speciman_list]
    if os.path.isfile(Dummy_File):
        for pop in speciman_list:
            ID=str(pop.ID).zfill(4)
            compound="".join(pop.genome)
            shutil.copy(Dummy_File, "{}/{}_{}.xyz".format(storage_path, ID,compound))
    return(charges, multiplicities,D_values)

def mock_fitness_Carbon(speciman_list, paths, Dummy_File):
    """
    Mock fitness function: Returns a fitness function according to the negative value of the number of carbon atoms in the decoded smile string based on the first ligand. Pure testing purposes.

    Parameters
    ----------
    speciman_list : LIST
        List of speciman to be evaluated.
    paths : LIST
        Contains the paths for work,final storage etc. Mandatory for all fitness functions.
    Dummy_File : STRING
        Path to a dummy xyz file that is copied into the hdf5 file to test functionality.

    """
    charges=[2]*len(speciman_list)
    multiplicities=[4]*len(speciman_list)
    D_values=[-sf.decoder(speciman.genome[1]).count("C") for speciman in speciman_list]
    if os.path.isfile(Dummy_File):
        for pop in speciman_list:
            ID=str(pop.ID).zfill(4)
            compound="".join(pop.genome)
            shutil.copy(Dummy_File, "{}/{}_{}.xyz".format(storage_path, ID,compound))
    return(charges, multiplicities,D_values)

def mock_fitness_database(speciman_list, paths, parameters):
    """
    Benchmark fitness function: Returns a fitness function according to some database, that does not need to contain all possible combinations of ligands.

    Parameters
    ----------
    speciman_list : LIST
        List of speciman to be evaluated.
    paths : LIST
        Contains the paths for work,final storage etc. Mandatory for all fitness functions.
    parameters : LIST [database, Dummy_File]
        Contains a database of values with keys equivalent to the corresponding ligands, and a path to a dummy xyz file that is copied into the hdf5 file to test functionality.

    """
    database,Dummy_File=parameters
    charges=[2]*len(speciman_list)
    multiplicities=[4]*len(speciman_list)
    keys=[pop.genome[0]+pop.genome[1] for pop in speciman_list]
    D_values=[]
    for key in keys:
        try:
            D_values.append(database[key])
        except:
            D_values.append(None)
    if os.path.isfile(Dummy_File):
        for pop in speciman_list:
            ID=str(pop.ID).zfill(4)
            compound="".join(pop.genome)
            shutil.copy(Dummy_File, "{}/{}_{}.xyz".format(storage_path, ID,compound))
    return(charges, multiplicities,D_values)

   
if __name__ == "__main__":
    #General Parameters
    popsize=64 # Number of total specimen in a generation
    gene_length=2 # Number of unique ligands in each compound
    mutation_rate=1 # Rate of mutation at each child process (<=1)
    max_generation=5 # Maximum number of planned generations, required for space management of hdf5 file
    
    
    
    #Database and model file location
    ligand_list_file="./" # Location of file containing all ligands used for static encoding
    Example_file="./Example_Geometry.xyz" # Example xyz file to test hdf5 functionalities
    if os.path.isfile(ligand_list_file):
        with open(ligand_list_file, "r") as jsonfile:
            ligand_list=json.load(jsonfile)
    else:
        ligand_list=["L"+str(i) for i in range(64)]
    
    
    
    #Semi Dynamic ligand information and model
    static_ligand="CAFYAK.Co_ligand0" # Name of static ligand
    static_ligand_smile="[O-][Si](C)(C)C" # Smile of static ligand
    RFmodel=joblib.load("./GA/model.joblib") # Loading provided RF model for pre-filtering
    minl=4 # Minimum number of tokens in Selfie string
    maxl=11 # Maximum number of tokens in Selfie string
    
    
    
    #Test Static Algorithm    
    ligand_path="./" 
    work_path="./Work_Static/"
    storage_path="./Storage_Static/"
    hdf5_file_name="{}/Test_Static.hdf5".format(storage_path)
    
    opt=GA.genetic_optimizer(ligand_list, popsize, max_generation, mutation_rate, mock_fitness_ID, gene_length, ligand_path, work_path, storage_path)
    opt.run_simulation(Example_file, hdf5_file_name, max_gen=3)
    opt2=GA.recover_static_optimizer(hdf5_file_name, mock_fitness_ID, ligand_list, work_path, ligand_path, storage_path)
    opt2.continue_simulation(Example_file,hdf5_file_name, max_gen=2)
    print(opt2.info)
    
    #Test Dynamic Algorithm
    ligand_path="./"
    ligand_storage_path="./Ligand_Storage/"
    work_path="./Work_Semi/"
    storage_path="./Storage_Semi/"
    hdf5_file_name="{}/Test_Semi.hdf5".format(storage_path)
    
    
    opt=GA.genetic_optimizer_semi_dynamic(ligand_list,popsize, max_generation, mutation_rate, mock_fitness_Carbon, gene_length, ligand_path, ligand_storage_path, work_path, storage_path, minl, maxl, static_ligand, static_ligand_smile, RFmodel=RFmodel)
    opt.run_simulation(Example_file, hdf5_file_name, max_gen=3)
    opt2=GA.recover_dynamic_optimizer(hdf5_file_name, mock_fitness_Carbon, work_path, ligand_path, storage_path, ligand_storage_path)
    opt2.continue_simulation(Example_file, hdf5_file_name, max_gen=2)
    print(opt2.info)