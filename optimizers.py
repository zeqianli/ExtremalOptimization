from heuristic_problems import *


class HeuristicOptimizer:
    """ Parent class for heuristic optimizers.
    
    Attributes:
        t: time
        problem: HeuristicProblem to be optimized
        max_iteration: maximum iteration 
        search_history: energy history along the searching path
    
    Methods: 
        step: make a single step 
        optimize: the main function to perform the optimization.
    """
    def __init__(self,problem:HeuristicProblem,max_iteration=100):
        self.t=0 
        self.problem=problem
        self.max_iteration=max_iteration
        self.search_history=[self.problem.energy] 
    
    def step(self):
        """Make one optimization step. 
        
        1. Select from a series of steps (e.g. by Boltzman distribution, by power law, etc. )
        2. Make the step
        """
        raise NotImplementedError

    def optimize(self,inspect=None):
        """ Main function to perform optimization. 
        
        inspect: if None:   don't show message
                 else:      display a message every inspect iterations
        """
        for i in range(self.max_iteration):
            self.step()
            self.search_history.append(self.problem.energy)
            if (not inspect) and i%inspect==0:
                print("Iteration=%d, energy=%f" % (self.t,self.problem.energy))
        

class SimulatedAnnealing(HeuristicOptimizer):
    """ Simulated annealing optimizer. 
    
    """
    def __init__(self,problem, max_iteration=100,
                    T0=1,cooling_schedule='exp',alpha=0.99,
                    nstep=1,max_fail=50,debug=False):
        HeuristicOptimizer.__init__(self,problem,max_iteration=max_iteration)

        self.T=T0
        self.cooling_schedule=cooling_schedule
        self.alpha=alpha
        #self.max_move=max_move
        self.max_fail=max_fail
        #TODO: nstep

        self.energy_list=[]
        
    def step(self):
        """
        TODO: two ways to do this:
            1. get a list of candidates
            2. apply one random step, and accept by probability. 
        """
        candidates,energies =self.get_local_step_candidates() #TODO: differentiate energy change/total energy
        for step in candidates:
            self.energy_list.append(self.calculate_energy_change(step))
                #TODO: make step passable
        self.energy_list=np.array(self.energy_list)
        
        prob=np.exp(-self.energy_list/self.T)
        prob/=np.sum(prob)
        
        chosen_step=np.random.choice(np.arange(len(candidates)),p=prob)

        self.problem.make_local_step(chosen_step)

    
    def update_T(self):
        if self.cooling_schedule is "exp":
            self.T*=alpha
        else:
            raise NotImplemented


class ExtremalOptimization(HeuristicOptimizer):
    """ Extremal optimization optimizer

    Recipe:
    1. In each iteration, obtain candidates;
    2. Express total energy of all candidates as a sum of individual steps;
        (Note that unlike simulated annealing, we energy instead of energy change.)
    3. Select a step by P~E^(-tau);
    4. Repeat after max_iteration iterations.

    Attributes: 
        t: time
        problem: HeuristicProblem to be optimized
        tau: exponent of power-law distribution when selecting candidates
    
    """
    def __init__(self,problem,max_iteration=100,tau=0.5):
        HeuristicOptimizer.__init__(self,problem,max_iteration=max_iteration)
        self.tau=tau

    def step(self):
        """ Make one step in optimization. 

        Note to not use total energy change, but total energy of all candidates of individual steps, and choose one.  

        1. Obtain step candidates 
        2. Select a step by P~E^(-\\tau)
        3.  Make the step
        """
        
        candidates,energies=self.problem.get_local_step_candidates()
        ps=energies**(-self.tau)
        ps/=np.sum(ps)     
        step=np.random.choice(candidates,p=ps)   
        self.problem.make_local_step(step)
        self.t+=1
