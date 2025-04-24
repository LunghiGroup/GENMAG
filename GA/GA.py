import numpy as np
import math
from copy import deepcopy
from operator import itemgetter
import random
import selfies as sf
from rdkit import RDLogger  
from .ECFP_RFClassifier import query_RF
import os
import shutil
import h5py

RDLogger.DisableLog('rdApp.*')
Selfie_alphabet=sf.get_semantic_robust_alphabet()
removal = ["[H]"]

for letter in Selfie_alphabet:
    if "+" in letter or "-" in letter or "Ring3" in letter or "Branch3" in letter:
        removal.append(letter)
for letter in removal:
    try:
        Selfie_alphabet.remove(letter)
    except:
        pass


Cdt= np.dtype([("Atomtype", "S4"),("X", "f4"),("Y", "f4"), ("Z", "f4")])
Idt=np.dtype([("Ligands", "S200"),("Charge", "i4"),("multiplicity", "f4"), ("Anisotropy", "f4")])

#------------------------------------------------------------GA---------------------------------------------------------
def get_Idt(gene_length):
    return np.dtype([("Ligand {}".format(i), "S200") for i in range(gene_length)]+[("Charge", "i4"),("multiplicity", "f4"), ("Anisotropy", "f4")])


def recover_static_optimizer(hdf5_file_name, fitness_function, ligand_list, work_path, ligand_path, storage_path):
    """
    Recovers a dynamic optimizer from and hdf5 file to continue claculation.

    Parameters
    ----------
    hdf5_file_name : STRING
        full path to HDF5 file.
    fitness_function : PYTHON FUNCTION
        Python fitness function that takes a list of specimen, a dictionary of paths, and a list of parameters as input, and returns charges, multiplicities and anisotropies. It should store a final xyz file called ID_compound.xyz in the storage location if storage in HDF5 is wanted.
    ligand_list : list
        list of ligands stored in ligand_path to be used for genetic optimization.
    work_path : STRING
        Location for fitness function to perform calculations in.
    ligand_path : STRING
        Location to find the ligands.
    storage_path : STRING
        Final storage location for output files and xyz files.


    Returns
    -------
    opt : genetic_optimizer
        Optimizer object with all information stored to continue calculation

    """
    
    if not os.path.isfile(hdf5_file_name):
        print("no file found")
        return None
    f=h5py.File(hdf5_file_name,"r")
    dset=f["Compounds"]
    max_generation=dset.attrs["max_generation"]
    mutation_rate=dset.attrs["mutation_rate"]
    popsize=dset.attrs["popsize"]
    gene_length=dset.attrs["gene_length"]
    current_ID=dset.attrs["current_ID"]
    placeholder=dset.attrs["placeholder"]
    current_population=dset.attrs["current_population"]
    generations=dset.attrs["generations"]
    
    opt=genetic_optimizer(ligand_list, popsize, max_generation, mutation_rate, fitness_function, gene_length, ligand_path, work_path, storage_path)
    for ID in current_population:
        data=list(dset[ID])
        genome=[dset[ID][j].decode() for j in range(gene_length)]
        pop=opt.speciman_type(genome, opt.gen_info,gene_length)
        pop.ID=ID
        pop.fitness=data[-1]
        if pop.fitness==None or math.isnan(pop.fitness):
            pop.fitness=placeholder
        opt.population.append(pop)
    opt.generations=generations
    opt.node_counter=current_ID
    f.close()
    return opt

def recover_dynamic_optimizer(hdf5_file_name, fitness_function, work_path, ligand_path, storage_path, ligand_storage_path, RFmodel=None):
    """
    Recovers a dynamic optimizer from and hdf5 file to continue claculation.

    Parameters
    ----------
    hdf5_file_name : STRING
        full path to HDF5 file.
    fitness_function : PYTHON FUNCTION
        Python fitness function that takes a list of specimen, a dictionary of paths, and a list of parameters as input, and returns charges, multiplicities and anisotropies. It should store a final xyz file called ID_compound.xyz in the storage location if storage in HDF5 is wanted.
    work_path : STRING
        Location for fitness function to perform calculations in.
    ligand_path : STRING
        Location to find the ligands.
    storage_path : STRING
        Final storage location for output files and xyz files.
    ligand_storage_path : STRING
        location to store ligands that were optimised during calculations.
    RFmodel : model, optional
        Optional RF model to be queried for pre-filtering of compounds.

    Returns
    -------
    opt : genetic_optimizer_semi_dynamic
        Optimizer object with all information stored to continue calculation

    """
    if not os.path.isfile(hdf5_file_name):
        print("no file found")
        return None
    f=h5py.File(hdf5_file_name,"r")
    dset=f["Compounds"]
    max_generation=dset.attrs["max_generation"]
    mutation_rate=dset.attrs["mutation_rate"]
    popsize=dset.attrs["popsize"]
    gene_length=dset.attrs["gene_length"]
    current_ID=dset.attrs["current_ID"]
    placeholder=dset.attrs["placeholder"]
    current_population=dset.attrs["current_population"]
    generations=dset.attrs["generations"]
    minl=dset.attrs["minl"]
    maxl=dset.attrs["maxl"]
    static_ligand=dset.attrs["static_ligand"]
    static_ligand_smile=dset.attrs["static_ligand_smile"]
    
    opt=genetic_optimizer_semi_dynamic([], popsize, max_generation, mutation_rate, fitness_function, gene_length, ligand_path, ligand_storage_path, work_path, storage_path,minl,maxl,static_ligand,static_ligand_smile,RFmodel=RFmodel)
    for dat in dset[0:current_ID]:
        smile=sf.decoder(dat[1].decode())
        opt.known_compounds.append(smile)
    for ID in current_population:
        data=list(dset[ID])
        genome=[dset[ID][j].decode() for j in range(gene_length)]
        pop=opt.speciman_type(genome, opt.gen_info,gene_length)
        pop.ID=ID
        pop.fitness=data[-1]
        if pop.fitness==None or math.isnan(pop.fitness):
            pop.fitness=placeholder
        opt.population.append(pop)
    opt.generations=generations
    opt.node_counter=current_ID
    f.close()
    return opt
    
 
    
def random_Selfie(minl,maxl,binding=True):
    """
    Parameters
    ----------
    minl : Int
        Minimum number of tokens
    maxl : Int
        maximum number of tokens
    binding : BOOLEAN, optional
        whether or not to start with a connecting atom. Used for recursive calls.
    Returns
    -------
    rnd_selfie : STRING
        randomly generated selfie of required length with connecting atom in the beginning.

    """
    binders=['[O]','[N]']#,'[C-1]']
    atoms=['[=C]', '[Cl]', '[I]', '[Br]', '[=O]', '[=P]', '[#C]', '[C]', '[#N]', '[O]', '[B]', '[=S]', '[=N]', '[F]', '[=B]', '[#P]', '[N]', '[#B]', '[P]', '[S]']
    operators=['[Branch1]','[=Branch1]','[#Branch1]','[Branch2]','[=Branch2]','[#Branch2]', '[Ring1]', '[=Ring1]', '[Ring2]', '[=Ring2]']
    operators=['[Branch1]','[Ring1]']
    numbers=["[C]", "[Ring1]", "[Ring2]","[Branch1]", "[=Branch1]", "[#Branch1]","[Branch2]", "[=Branch2]", "[#Branch2]","[O]", "[N]", "[=N]", "[=C]", "[#C]", "[S]", "[P]"]
    carbons=['[C]']*30+['[=C]']*10
    atoms3=['[=C]','[N]']
    atoms2=['[O]','[=N]','[#C]',]
    Tokens=[]
    if binding:
        start=np.random.choice(binders)
        Tokens.append(start)
    operator_threshhold=0.3
    length=np.random.randint(minl,maxl)
    i=0
    while i <length:
        if np.random.rand()<operator_threshhold and length-i>2:
            #add operator and number
            operator=random.choice(operators)
            number=np.random.randint(1,length-i-1)
            number=np.min([number,8])
            if "Branch" in operator:
                Tokens.append(operator)
                Tokens.append(numbers[number])
            for j in range(number):
                Tokens.append(list(carbons+atoms3)[np.random.randint(0,len(carbons+atoms3))])
            if "Ring" in operator:
                Tokens.append(operator)
                Tokens.append(numbers[number])
            i+=2+number
        else:
            #add atom
            if i==length-1:
                Tokens.append(random.choice(atoms))
            else:
                Token=list(carbons+atoms3+atoms2)[np.random.randint(0,len(carbons+atoms3+atoms2))]
                Tokens.append(Token)
            i+=1
    rnd_selfie="".join(Tokens)
    return rnd_selfie


def compare_genomes(pop1, pops, gene_type):
    """
    Parameters
    ----------
    pop1 : speciman
        speciman to be searched for
    pops : speciman
        list of population to check for duplicates in
        
    Returns
    -------
    bool
        whether pop1 is already found in pops
    """
    for pop2 in pops:
        if gene_type=="list_index":
            if np.all(pop1.genome == pop2.genome, axis=0):
                return True
        else:
            Smile1=sf.decoder(pop1.genome[1])
            Smile2=sf.decoder(pop2.genome[1])
            if Smile1==Smile2:
                return True
    return False

class speciman:
    #speciman being a single individual, capable of mutation and crossover on it's own.
    def __init__(self,genome,gen_info,gene_length):
        """
        Parameters
        ----------
        genome : list or string
            Either list of genes (ligands) or string of genes. If set to "random", automatically creates random genome.
        gen_info : list
            Contains ligand list to sample from.
        gene_length : Int
            Number of genes.

        Returns
        -------
        None.

        """
        self.gen_info=gen_info
        self.fitness=None
        self.ligand_list=gen_info
        self.gene_length=gene_length
        self.ID=None
        self.parents=[]
        if type(genome)==str:
            if genome=="random":
                self.genome=sorted(random.sample(gen_info,gene_length))
            else:
                self.genome=sorted(genome)
        else:
            self.genome=sorted(genome)
    
    def mutate(self,rate):
        """
        Parameters
        ----------
        rate: float
            chance of mutation happening
        
        
        Returns:
        --------
            child: speciman
                speciman with up to 1 gene being randomized
        """
        if np.random.rand()<rate:
            
            #mutate random gene
            new_genome=self.genome
            mut_pos=np.random.randint(0,self.gene_length)
            pos=random.randint(0,len(self.ligand_list)-1)
            new_genome[mut_pos]=self.ligand_list[pos]
            new_genome=sorted(new_genome)
            child=speciman(new_genome,self.ligand_list,self.gene_length)
            return child
        else:
            #don't change anything
            return self
        
    def crossover_mutate(self,mutation_rate,speciman2):
        """
        Parameters
        ----------
        mutation_rate: float
            chance of mutation happening
        speciman2: speciman
            second parent
        
        Returns:
        --------
            child: speciman
                speciman bred by the two parents, with up to 1 gene being randomized
        """
        
        father_gene_number=np.random.randint(0,self.gene_length-1)
        father_gene=random.sample(list(self.genome),father_gene_number)
        mother_gene_number=self.gene_length-father_gene_number
        mother_gene=random.sample(list(speciman2.genome),mother_gene_number)
        new_genome=sorted(np.concatenate((father_gene,mother_gene)))
        child=speciman(np.array(new_genome),self.ligand_list,self.gene_length)
        return([child.mutate(mutation_rate)])
    
class speciman_dynamic(speciman):
    #Subclass of speciman, which uses selfies to describe its ligands instead of a fixed ligand list. Capable of mutation and crossover on its own.
    def __init__(self,genome,gen_info,gene_length):
        """
        Parameters
        ----------
        genome: np.array or STRING
            contains the individual ligand IDs, stored sorted numerically, set to "random" for first gen
        gen_info: LIST [minl,maxl]
            contains minimum and maximum token number
        gene_length: INT
            number of genes in genome
        """
        self.fitness=None
        self.gene_length=gene_length
        self.ID=None
        self.parents=[]
        self.minl=gen_info[0]
        self.maxl=gen_info[1]
        self.speciman_type=speciman_dynamic
        if type(genome)==str:
            if genome=="random":
                self.genome=[random_Selfie(self.minl, self.maxl)for i in range(gene_length)]
            else:
                self.genome=genome
        else:
            self.genome=genome

    def mutate(self, rate):
        """
        Parameters
        ----------
        rate : float
            Rate of Mutation, equally split between extension, modification or reduction of the tokens.

        Returns
        -------
        Offspring speciman_molecule
            The possibly mutated clone of the original speciman.

        """
        new_genome=deepcopy(self.genome)
        for i,gene in enumerate(self.genome):
            Selfietokens=list(sf.split_selfies(gene))
            if np.random.rand()<rate:
                mutat_type=random.sample([0,1,2],1)[0]
                if mutat_type==0 and len(Selfietokens)>2:#remove Token
                    Selfietokens.pop(np.random.randint(1,len(Selfietokens)))
                elif mutat_type==1 and len(Selfietokens)<self.maxl:#add Token
                    pos=np.random.randint(1,len(Selfietokens))
                    Selfietokens.insert(pos,random.sample(Selfie_alphabet,1)[0])
                elif mutat_type==2:
                    new_gene=random.sample(Selfie_alphabet,1)[0]
                    Selfietokens[np.random.randint(1,len(Selfietokens))]=new_gene
                new_Selfie="".join(Selfietokens)
                new_genome[i]=new_Selfie
        return(speciman_dynamic(new_genome,[self.minl, self.maxl],self.gene_length))
    
    def crossover_mutate(self, rate, speciman2):
        """
        Parameters
        ----------
        rate : float
            Rate of Mutation, equally split between extension, modification or reduction of the tokens.

        speciman2 : speciman_molecule
            second parent whos genome will be mixed to produce two offsprings

        Returns
        -------
        children : list of speciman_molecule
            List containing the two offsprings, possibly mutated.

        """


        genome1=[]
        genome2=[]
        for i in range(self.gene_length):
            
            alpha=np.random.rand()*0.8+0.1
            father_tokens=list(sf.split_selfies(self.genome[i]))
            mother_tokens=list(sf.split_selfies(speciman2.genome[i]))
            
            father_gene_number=round(alpha*len(father_tokens))
            mother_gene_number=round((1-alpha)*len(mother_tokens))
            new_tokens1=father_tokens[:father_gene_number]+mother_tokens[-mother_gene_number:]
            new_gene1="".join(new_tokens1)
            genome1.append(new_gene1)
            
            new_tokens2=mother_tokens[:mother_gene_number]+father_tokens[-father_gene_number:]
            new_gene2="".join(new_tokens2)
            genome2.append(new_gene2)
        child1=speciman_dynamic(genome1,[self.minl, self.maxl],self.gene_length).mutate(rate)
        child2=speciman_dynamic(genome2,[self.minl, self.maxl],self.gene_length).mutate(rate)
        return([child1, child2])
        
class speciman_semi_dynamic(speciman):
    #Subclass of speciman_dynamic. Has a single fixed ligand, and a number of selfie based ones. Capable of mutation and crossover on its own.
    def __init__(self,genome,gen_info,gene_length):
        """
        Parameters
        ----------
        genome: np.array or STRING
            Contains the individual ligand IDs, stored sorted numerically, set to "random" for first gen.
        gen_info: LIST [minl, maxl, static_ligand]
            Contains information on minimum and maximum token number, as well as the fixed ligands name.
        gene_length: INT
            number of ligands in genome
        """
        self.fitness=None
        self.gene_length=gene_length
        self.ID=None
        self.parents=[]
        self.gen_info=gen_info
        self.minl=gen_info[0]
        self.maxl=gen_info[1]
        self.speciman_type=speciman_semi_dynamic
        if type(genome)==str:
            if genome=="random":
                rest_genome=[random_Selfie(self.minl, self.maxl)for i in range(gene_length-1)]
                self.genome=[gen_info[2]]+rest_genome
            else:
                self.genome=genome
        else:
            self.genome=genome
    def get_compound(self):
        """
        

        Returns
        -------
        compound : STRING
            contains decoded smiles and fixed ligand to compare compounds quickly.
        """
        fixed=[self.genome[0]]
        smiles=[sf.decoder(selfie) for selfie in self.genome[1:]]
        compound="".join(fixed+smiles)
        return(compound)
    def mutate(self, rate):
        """
        Parameters
        ----------
        rate : float
            Rate of Mutation, equally split between extension, modification or reduction of the tokens.

        Returns
        -------
        Offspring speciman_molecule
            The possibly mutated clone of the original speciman.

        """
        new_genome=deepcopy(self.genome)
        for i,gene in enumerate(self.genome[1:]):
            Selfietokens=list(sf.split_selfies(gene))
            if np.random.rand()<rate:
                mutat_type=random.sample([0,1,2],1)[0]
                if mutat_type==0 and len(Selfietokens)>2:#remove Token
                    Selfietokens.pop(np.random.randint(1,len(Selfietokens)))
                elif mutat_type==1 and len(Selfietokens)<self.maxl:#add Token
                    pos=np.random.randint(1,len(Selfietokens))
                    Selfietokens.insert(pos,random.sample(Selfie_alphabet,1)[0])
                elif mutat_type==2:
                    new_gene=random.sample(Selfie_alphabet,1)[0]
                    Selfietokens[np.random.randint(1,len(Selfietokens))]=new_gene
                new_Selfie="".join(Selfietokens)
                new_genome[i+1]=new_Selfie
        return(speciman_semi_dynamic(new_genome,self.gen_info,self.gene_length))
    
    def crossover_mutate(self, rate, speciman2):
        """
        Parameters
        ----------
        rate : float
            Rate of Mutation, equally split between extension, modification or reduction of the tokens.
        speciman2 : speciman_molecule
            second parent whos genome will be mixed to produce two offsprings

        Returns
        -------
        children : list of speciman_molecule
            List containing the two offsprings, possibly mutated.
        """
        genome1=[self.genome[0]]
        genome2=[self.genome[0]]
        for i in range(self.gene_length-1):
            alpha=np.random.rand()*0.8+0.1
            father_tokens=list(sf.split_selfies(self.genome[i+1]))
            mother_tokens=list(sf.split_selfies(speciman2.genome[i+1]))
            
            father_gene_number=round(alpha*len(father_tokens))
            mother_gene_number=round((1-alpha)*len(mother_tokens))
            new_tokens1=father_tokens[:father_gene_number]+mother_tokens[-mother_gene_number:]
            new_gene1="".join(new_tokens1)
            genome1.append(new_gene1)
            
            new_tokens2=mother_tokens[:mother_gene_number]+father_tokens[-father_gene_number:]
            new_gene2="".join(new_tokens2)
            genome2.append(new_gene2)
        child1=speciman_semi_dynamic(genome1, self.gen_info, self.gene_length).mutate(rate)
        child2=speciman_semi_dynamic(genome2, self.gen_info, self.gene_length).mutate(rate)
        return([child1, child2])

class genetic_optimizer:
    def __init__(self,ligand_list,popsize,max_generation,mutation_rate,fitness_function,gene_length,ligand_path, work_path, storage_path):
        """
        Main Class of the genetic algorithm. It performs the loop of optimization based on a given 
        fitness function, and contains information on fitness growth, current population etc.
        
        Parameters
        ----------
        ligand_list : LIST
            List of all available ligands, must be stored as xyz files in ligand_path.
        popsize : INT
            Number of total population before culling. Number of new compounds each generation is half that.
        max_generation : INT
            Maximum number of expected generations to be performed to prepare adequate space in hdf5 file. Generation loop automatically ends at this number/
        mutation_rate : FLOAT
            Rate of mutation for specimen.
        fitness_function : Python function
            Python fitness function that takes a list of specimen, a dictionary of paths, and a list of parameters as input, and returns charges, multiplicities and anisotropies. It should store a final xyz file called ID_compound.xyz in the storage location if storage in HDF5 is wanted.
        gene_length : INT
            Number of unique ligands in each compound.
        ligand_path : STRING
            Location of ligand xyz files.
        work_path : STRING
            Location for work of fitness function to be performed in.
        storage_path : STRING
            Location for final storage of output files and compound xyz data.
        """
        #Parameters to be save in HDF5 file:
        self.max_generation=max_generation
        self.mutation_rate=mutation_rate
        self.popsize=popsize
        self.gene_length=gene_length
        self.current_ID=0
        self.placeholder_cost=100
        
        self.generations=0
        
        #Parameters not saved in HDF5 file:
        self.speciman_type=speciman
        self.known_compounds=[]
        self.max_breed=10000
        self.ligand_list=ligand_list
        self.fitness_function=fitness_function
        self.population=[]
        self.info=[]
        self.gen_info=ligand_list
        self.evaluation_counter=0
        self.node_counter=0
        self.real_counter=0
        #Paths for calculations:
        self.ligand_path=ligand_path
        self.work_path=work_path
        self.storage_path=storage_path
        self.paths={
        "ligand_path":ligand_path,
        "work_path":work_path,
        "storage_path":storage_path,
        }
        
       
    def create_HDF5(self, hdf5_file_name):
        """
        Creates a new hdf5 file and stores all information needed to continue the calculation alter.
        """
        file=h5py.File(hdf5_file_name,"w")
        number_of_specimen=int(self.popsize*(1+self.max_generation/2))
        dset = file.create_dataset("Compounds", (number_of_specimen), dtype=get_Idt(self.gene_length))
        dset.attrs["current_population"]=list(range(self.popsize))
        dset.attrs["current_ID"]=0
        dset.attrs["popsize"]=self.popsize
        dset.attrs["gene_length"]=self.gene_length
        dset.attrs["mutation_rate"]=self.mutation_rate
        dset.attrs["max_generation"]=self.max_generation
        dset.attrs["number_ligands"]=len(self.ligand_list)
        dset.attrs["placeholder"]=self.placeholder_cost
        dset.attrs["generations"]=0
       
    def evaluate_fitness(self,population,fitness_parameters):
        """
        Calculates fitness for all speciman in current population, that haven't been set already. 
        If fitness function returns None, forced mutation is attempted
        """
        charges, multiplicities, D_values=self.fitness_function(population, self.paths, fitness_parameters)
        for i,pop in enumerate(population):
            fitness=D_values[i]
            if fitness==None:
                fitness=self.placeholder_cost
            pop.fitness=fitness
        compounds=["".join(pop.genome) for pop in population]
        return(charges, multiplicities, D_values, compounds)
        
    def init_first_gen(self):
        """
        Generates first generation of randomly created speciman
        """
        for i in range(self.popsize):
            pop=self.speciman_type("random",self.gen_info,self.gene_length)
            pop.ID=self.node_counter
            self.node_counter+=1
            self.population.append(pop)
            
    
    def parent_select(self):
        """
        Selects upper half of current population for next gen breeding.
        """
        fitness_values=[speciman.fitness for speciman in self.population]
        for i in range(len(fitness_values)):
            if fitness_values[i]==None:
                fitness_values[i]=self.placeholder_cost
            if math.isnan(fitness_values[i]):
                fitness_values[i]=self.placeholder_cost
        fitness_values,sorted_parents = (list(t) for t in zip(*sorted(zip(fitness_values, self.population),key=itemgetter(0))))
        elites=sorted_parents[:math.floor(len(sorted_parents)/2)]
        return(elites)
    
    def update_population(self):
        """
        Creates a new batch of specimen based on the current population to create a new generation after the elite selection
        """
        parents=self.parent_select()
        new_pop=deepcopy(parents)
        breed_counter=0
        while len(new_pop)<self.popsize and breed_counter<self.max_breed:
            parent1=parents[np.random.randint(0,len(parents))]
            parent2=parents[np.random.randint(0,len(parents))]
            children=parent1.crossover_mutate(self.mutation_rate,parent2)
            for child in children:
                if not "".join(child.genome) in self.known_compounds:
                    if not "".join(child.genome) in self.known_compounds:
                        self.known_compounds.append("".join(child.genome))
                    child.ID=self.node_counter
                    self.node_counter+=1
                    new_pop.append(child)
                    
                if len(new_pop)==self.popsize:
                    break
            breed_counter+=1
            
        while len(new_pop)<self.popsize and breed_counter<self.max_breed:
            parent1=parents[np.random.randint(0,len(parents))]
            parent2=parents[np.random.randint(0,len(parents))]
            children=parent1.crossover_mutate(self.mutation_rate,parent2)
            for child in children:
                if not compare_genomes(child,new_pop, "list_index"):
                #if not "".join(child.genome) in self.known_compounds:
                    if not "".join(child.genome) in self.known_compounds:
                        self.known_compounds.append("".join(child.genome))
                    child.ID=self.node_counter
                    self.node_counter+=1
                    new_pop.append(child)
                    
                if len(new_pop)==self.popsize:
                    break
            breed_counter+=1
        self.population=new_pop
            
    def get_information(self):
        """
        Documents max, min and average fitness of current population in self.info
        """
        fitnesses=[]
        for speciman in self.parent_select():
            if speciman.fitness!=self.placeholder_cost and not math.isnan(speciman.fitness):
                fitnesses.append(speciman.fitness)
        fitnesses.sort()
        maxf=fitnesses[-1]
        averagef=np.average(fitnesses)
        minf=fitnesses[0]
        self.info.append([minf,averagef,maxf])
        return([minf,averagef,maxf])
        
        
    def continue_simulation(self, fitness_parameters, hdf5_file_name, max_gen=-1):
        """
        Continues simulation with current population. Used to cut calculation into pieces on HPC clusters. 
        Paths are taken from self.paths, while fitness_parameters is changebable at each call.

        Parameters
        ----------
        fitness_parameters : LIST
            List of parameters that are handed to fitness function. Usually contains computational details, database for benchmarks etc.
        hdf5_file_name : STRING
            Full path to hdf5 file to be written into.
        max_gen : INT, optional
            Optional: Number of generations to be performed in this run. Does not overwrite maximum number of generations. Can be set to -1 to continue until maximum number is reached.
        """
        if not os.path.isdir(self.work_path):
            os.mkdir(self.work_path)
        if not os.path.isdir(self.storage_path):
            os.mkdir(self.storage_path)
        gen_done=0
        for gen in range(self.generations,self.max_generation):
            gen_done+=1
            if gen_done>max_gen and max_gen!=-1:
                break
            self.update_population()
            charges, multiplicity, D_values, compounds=self.evaluate_fitness(self.population[int(self.popsize/2):],fitness_parameters)
            f=h5py.File(hdf5_file_name,"r+")
            self.generations+=1
            self.save_coordinates(self.population[int(self.popsize/2):], hdf5_file_name, charges, multiplicity, D_values, compounds)
            self.get_information()
            dset=f["Growth"]
            dset[gen+1]=self.info[-1]
            f["Compounds"].attrs["generations"]=self.generations
            f.close()

    
    def save_coordinates(self, population, hdf5_file_name, charges, multiplicity, D_values, compounds):
        """
        Stores xyz coordinates inside hdf5 file, sorted by generation folders. Requires the final geometry to be saved as ID_compound.xyz inside final storage location.
        """
        f=h5py.File(hdf5_file_name,"r+")
        f["Compounds"].attrs["current_ID"]+=len(population)
        f["Compounds"].attrs["current_population"]=[pop.ID for pop in self.population]
        for i, pop in enumerate(population):
            ID=str(pop.ID).zfill(4)
            dset=f["Compounds"]
            values=tuple(pop.genome+[charges[i], multiplicity[i], D_values[i]])
            dset[pop.ID]=values
            try:
                name=ID+"_"+pop.genome[0]+"_"+pop.genome[1]
                with open("{}/{}_{}.xyz".format(self.storage_path,ID,compounds[i]),"r") as datafile:
                    lines=datafile.readlines() 
                    dset=f.create_dataset("generation_{}/{}".format(self.generations,name),(len(lines)-2,),dtype=Cdt)
                    dset.attrs["D_values"]=pop.fitness
                    dset.attrs["atomnumber"]=len(lines)-2
                    for k,line in enumerate(lines[2:]): 
                        words=line.split()
                        atom=words[0]
                        x=float(words[1])
                        y=float(words[2])
                        z=float(words[3])
                        dset[k]=(atom,x,y,z)
            except:
                pass
        f.close()
        
            
        
    def run_simulation(self, fitness_parameters,hdf5_file_name="local",max_gen=-1):
        """
        Starts simulation of genetic algorithm. Performs a number of iterations using self.fitness_function, by using the fitness parameters. 
        Automatically updates self.info, and writes information and final xyz coordinates in hdf5 file.
        Parameters
        ----------
        fitness_parameters : LIST
            List of parameters that are handed to fitness function. Usually contains computational details, database for benchmarks etc.
        hdf5_file_name : STRING
            Full path to hdf5 file to be written into.
        max_gen : INT, optional
            Optional: Number of generations to be performed in this run. Does not overwrite maximum number of generations. 
            Can be set to -1 to continue until maximum number is reached.

        """
        
        if max_gen==-1:
            max_gen=self.max_generation
            
        
        if not os.path.isdir(self.work_path):
            os.mkdir(self.work_path)
        if not os.path.isdir(self.storage_path):
            os.mkdir(self.storage_path)
        
        if hdf5_file_name=="local":
            hdf5_file_name="{}/OutputFile.hdf5".format(os.getcwd())
        file=h5py.File(hdf5_file_name,"w")
        number_of_specimen=int(self.popsize*(1+self.max_generation/2))
        dset = file.create_dataset("Compounds", (number_of_specimen), dtype=get_Idt(self.gene_length))
        dset.attrs["current_population"]=list(range(self.popsize))
        dset.attrs["current_ID"]=0
        dset.attrs["popsize"]=self.popsize
        dset.attrs["gene_length"]=self.gene_length
        dset.attrs["mutation_rate"]=self.mutation_rate
        dset.attrs["max_generation"]=self.max_generation
        dset.attrs["number_ligands"]=len(self.ligand_list)
        dset.attrs["placeholder"]=self.placeholder_cost
        dset.attrs["generations"]=0
        dset2 = file.create_dataset("Growth", (self.max_generation+1,3),dtype="f4")
        file.close()
        
        self.init_first_gen()
        charges, multiplicity, D_values, compounds=self.evaluate_fitness(self.population,fitness_parameters)
        self.save_coordinates(self.population,hdf5_file_name, charges, multiplicity, D_values, compounds)
        self.get_information()
        f=h5py.File(hdf5_file_name,"r+")
        dset=f["Growth"]
        dset[0]=self.info[0]
        f.close()
        gen=-1
        for gen in range(max_gen):
            self.update_population()
            charges, multiplicity, D_values, compounds=self.evaluate_fitness(self.population[int(self.popsize/2):],fitness_parameters)
            f=h5py.File(hdf5_file_name,"r+")
            self.generations+=1
            self.save_coordinates(self.population[int(self.popsize/2):], hdf5_file_name, charges, multiplicity, D_values, compounds)
            self.get_information()
            dset=f["Growth"]
            dset[gen+1]=self.info[-1]
            
            f["Compounds"].attrs["generations"]=self.generations
            f.close()
        return(self.parent_select()[0])
    
             
        
class genetic_optimizer_semi_dynamic(genetic_optimizer):
    def __init__(self, ligand_list, popsize, max_generation, mutation_rate, fitness_function, gene_length, ligand_path,ligand_storage_path, work_path, storage_path,minl,maxl,static_ligand,static_ligand_smile,RFmodel=None):
        """
        Subclass of the genetic optimizer, used for semi dynamic encoding. It inherits the same behaviour, 
        but instead passes some additional information for the selfie generation to the specimen, and allows 
        for a pre-filtering using a trained random forest model. Will eventually be absorbed into the main class.    

        Parameters
        ----------
        ligand_list : LIST
            List of all available ligands, must be stored as xyz files in ligand_path.
        popsize : INT
            Number of total population before culling. Number of new compounds each generation is half that.
        max_generation : INT
            Maximum number of expected generations to be performed to prepare adequate space in hdf5 file. Generation loop automatically ends at this number/
        mutation_rate : FLOAT
            Rate of mutation for specimen.
        fitness_function : Python function
            Python fitness function that takes a list of specimen, a dictionary of paths, and a list of parameters as input, and returns charges, multiplicities and anisotropies. It should store a final xyz file called ID_compound.xyz in the storage location if storage in HDF5 is wanted.
        gene_length : INT
            Number of unique ligands in each compound.
        ligand_path : STRING
            Location of ligand xyz files.
        ligand_storage_path : STRING
            DESCRIPTION.
        work_path : STRING
            Location for work of fitness function to be performed in.
        storage_path : STRING
            Location for final storage of output files and compound xyz data.
        minl : INT
            Minimum number of tokens in Selfie string.
        maxl : OMT
            Maximum number of tokens in Selfie string.
        static_ligand : STRING
            Static ligand Smile to reference xyz file in ligand_path.
        static_ligand_smile : STRING
            Smile describing the static ligand. Will be stored in specimen and used for compound generation and pre-filtering
        RFmodel : model, optional
            Optional: random forest model to be queried for pre-filtering. See ECFP_RFClassifier.py for documentation.

        """
        super(genetic_optimizer_semi_dynamic, self).__init__(ligand_list,popsize,max_generation,mutation_rate,fitness_function,gene_length,ligand_path, work_path, storage_path)
        self.speciman_type=speciman_semi_dynamic
        self.minl=minl
        self.maxl=maxl
        self.gen_info=[minl, maxl, static_ligand]
        self.model=RFmodel
        self.static_ligand=static_ligand
        self.static_ligand_smile=static_ligand_smile
        self.ligand_storage_path=ligand_storage_path
        self.paths={
        "ligand_path":ligand_path,
        "work_path":work_path,
        "storage_path":storage_path,
        }
        if not os.path.isdir(self.ligand_storage_path):
            os.mkdir(self.ligand_storage_path)
            
            
    def run_simulation(self, fitness_parameters,hdf5_file_name,max_gen=-1):
        """
        Starts simulation of genetic algorithm. Performs a number of iterations using self.fitness_function, by using the fitness parameters. 
        Automatically updates self.info, and writes information and final xyz coordinates in hdf5 file.
        Modified from original main class due to different parameters needed (minl,maxl,fixed ligand).
        Parameters
        ----------
        fitness_parameters : LIST
            List of parameters that are handed to fitness function. Usually contains computational details, database for benchmarks etc.
        hdf5_file_name : STRING
            Full path to hdf5 file to be written into.
        max_gen : INT, optional
            Optional: Number of generations to be performed in this run. Does not overwrite maximum number of generations. 
            Can be set to -1 to continue until maximum number is reached.

        """
        
        if max_gen==-1:
            max_gen=self.max_generation
        if not os.path.isdir(self.work_path):
            os.mkdir(self.work_path)
        if not os.path.isdir(self.storage_path):
            os.mkdir(self.storage_path)
        
        file=h5py.File(hdf5_file_name,"w")
        number_of_specimen=int(self.popsize*(1+self.max_generation/2))
        dset = file.create_dataset("Compounds", (number_of_specimen), dtype=get_Idt(self.gene_length))
        dset.attrs["current_population"]=list(range(self.popsize))
        dset.attrs["current_ID"]=0
        dset.attrs["popsize"]=self.popsize
        dset.attrs["gene_length"]=self.gene_length
        dset.attrs["mutation_rate"]=self.mutation_rate
        dset.attrs["max_generation"]=self.max_generation
        dset.attrs["number_ligands"]=len(self.ligand_list)
        dset.attrs["placeholder"]=self.placeholder_cost
        dset.attrs["generations"]=0
        dset.attrs["minl"]=self.minl
        dset.attrs["maxl"]=self.maxl
        dset.attrs["static_ligand"]=self.static_ligand
        dset.attrs["static_ligand_smile"]=self.static_ligand_smile
        dset2 = file.create_dataset("Growth", (self.max_generation+1,3),dtype="f4")
        file.close()
        self.init_first_gen()
        charges, multiplicity, D_values, compounds=self.evaluate_fitness(self.population,fitness_parameters)
        self.save_coordinates(self.population,hdf5_file_name, charges, multiplicity, D_values, compounds)
        self.get_information()
        f=h5py.File(hdf5_file_name,"r+")
        dset=f["Growth"]
        dset[0]=self.info[0]
        f.close()
        gen=-1
        for gen in range(max_gen):
            self.update_population()
            charges, multiplicity, D_values, compounds=self.evaluate_fitness(self.population[int(self.popsize/2):],fitness_parameters)
            f=h5py.File(hdf5_file_name,"r+")
            self.generations+=1
            self.save_coordinates(self.population[int(self.popsize/2):], hdf5_file_name, charges, multiplicity, D_values, compounds)
            self.get_information()
            dset=f["Growth"]
            dset[gen+1]=self.info[-1]
            
            f["Compounds"].attrs["generations"]=self.generations
            f.close()
        return(self.parent_select()[0])
        
    def update_population(self):
        """
        Creates a new batch of specimen based on the current population to create a new generation after the elite selection.
        Modified from original main class to implement pre-filtering using random forest model.
        """
        parents=self.parent_select()
        new_pop=deepcopy(parents)
        breed_counter=0
        while len(new_pop)<self.popsize and breed_counter<self.max_breed:
            parent1=parents[np.random.randint(0,len(parents))]
            parent2=parents[np.random.randint(0,len(parents))]
            children=parent1.crossover_mutate(self.mutation_rate,parent2)
            for child in children:
                smile1=self.static_ligand_smile
                smile2=sf.decoder(child.genome[1])
                compound=child.get_compound()
                if (self.model==None or query_RF(self.model, [smile1], [smile2])==1) and (not compound in self.known_compounds):
                    self.known_compounds.append(compound)
                    child.ID=self.node_counter
                    self.node_counter+=1
                    new_pop.append(child)
                else:
                    pass
                if len(new_pop)==self.popsize:
                    break
            breed_counter+=1
        self.population=new_pop
        
    def init_first_gen(self):
        init_counter=0
        while len(self.population)<self.popsize:
            init_counter+=1
            pop=self.speciman_type("random",self.gen_info,self.gene_length)
            smile1=[self.static_ligand_smile]
            smile2=[sf.decoder(pop.genome[1])]
            if self.model==None or query_RF(self.model, smile1, smile2)==1:
                pop.ID=self.node_counter
                self.node_counter+=1
                self.population.append(pop)
        
