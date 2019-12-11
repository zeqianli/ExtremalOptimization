from graph import *
from convex_optimizers import *

class HeuristicProblem:
    """ Parent class for heuristic problems. """
    
    def __init__(self,data):
        self.energy=None
        
    def get_data(self):
        """ Return an attribute recording the data (e.g. graph). """
        raise NotImplementedError

    def get_state(self):
        """ Return an attribute recording the data (e.g. hierarchy in FAS)."""
        raise NotImplementedError

    def calculate_energy(self):
        """ Calculate energy from scratch. 
        
        Unless specified, this function is computationally costly and should not be called often. 
        """
        raise NotImplementedError
    
    class HeuristicStep:
        """ Heuristic step class. 
        
        This class only acts as a collection of attributes. """
        def __init__(self):
            raise NotImplementedError
            
    def get_local_step_candidates(self,floor0=False, return_energy=True):
        """ Not implemented for now. """
        raise NotImplementedError

    def propose_step(self, return_energy_change=True):
        """Propose a step.
        
        Used for algorithms such as simulated annealing.

        Attributes:
            return_energy_change: 
                if True, return the change after applying the step (e.g used for SA);
                if False, return the "energy" associated with the step (problem specific).

        Return: step and energy.  
        """
        raise NotImplementedError

    def make_local_step(self,step:HeuristicStep):
        """ Make a step in local search that update state and energy locally.  

        May or may not call calculate_energy_change(). If not, remember to update energy in another way.   
        """
        raise NotImplementedError

    def calculate_energy_change(self):
        """ Return updated energy after a local step. 
        
        This function DOES NOT MAKE MOVES.

        Unless specified, this function should be much cheaper than calculate_energy().
        """
        raise NotImplementedError

    def _check_updated_energy(self):
        """ Check if calculate_updated_energy() is working properly by comparing with calculate_energy. """

        raise NotImplementedError


class MinimumFeedbackArc(HeuristicProblem):
    """ Minimum feedback arc problem.
    
    """
    def __init__(self,graph:DirectedGraph):
        """ Assume following conditions:
            - double edge removed (by caling graph.reduced_graph())
            - self edge removed
            - no repeated edges
        
        """
        HeuristicProblem.__init__(self,graph)

        self.graph=graph

        # initial hierarchy: sort SCCs, and randomly permute within each SCC. 
        initial_h=np.argsort(self.graph.scc_ind) 
        scc_ind_sorted=self.graph.scc_ind[initial_h]
        for i in range(self.graph.n_scc):
            initial_h[scc_ind_sorted==i]=np.random.permutation(initial_h[scc_ind_sorted==i])
        self.h=initial_h # h[i]: node index of ith position in hierarchy
        self.h_of_node=np.argsort(self.h) # h_of_node[i]: hierarchy position of ith node
        
        self.fas_ind=self.get_FAS_ind()
        self.energy=self.calculate_energy(self.h_of_node)

    def get_data(self):
        return self.graph

    def get_state(self):
        return self.h

    def calculate_energy(self,h_of_node=None):
        """ Calculate energy (# feedback arcs) from scratch give hierarchy h_of_node. .
        
        This function costs O(E) and should not be called often. 
        """
        return len(self.get_FAS_ind(h_of_node=h_of_node))

    class FASStep(HeuristicProblem.HeuristicStep):  
    
        def __init__(self,p,c,floor0=False,move_parent=True,energy_change=None):
            """ FAS step. 

            p: parent node
            c: children node
            floor0: TODO
            move_parent:
                if True:    move p to c's top (--c--------p--   ->   --pc----------)
                if False:   move c to p's top (--c--------p--   ->   ----------pc--)
            energy_change: energy change after applying the step
            """

            self.p=p
            self.c=c
            self.floor0=floor0
            self.move_parent=move_parent
            self.energy_change=energy_change
        
        def __str__(self):
            return "p=%d, c=%d, floor0=%s, move_parent=%s, energy_change=%s" % (self.p,self.c,self.floor0,self.move_parent,self.energy_change)

    def get_local_step_candidates(self,return_energy_change=False):
        # TODO
        pass

    def propose_step(self, edge_ind=None, floor0=False, move_parent=True,cal_energy_change=True):
        """Propose a step. Used for SA. 
        
        edge_ind: index of the edge to be moved. 
            if None: choose randomly from fas_ind
        floor0: unclear what this is for
        move_parent: see FASStep
        cal_energy_change: True if calculate energy change of this step and store in FASStep object.
        """
        if edge_ind is None:
            edge_ind=np.random.choice(self.fas_ind)
        p,c=self.graph.edges[edge_ind]
        energy_change=None
        if cal_energy_change:
            energy_change=self.calculate_energy_change(p,c,floor0=floor0,move_parent=move_parent)
        
        return MinimumFeedbackArc.FASStep(p,c,floor0=floor0,move_parent=move_parent,energy_change=energy_change)

    def make_local_step(self,step):
        """ Apply step. 

        energy_change:
            if None: energy change is not calculated beforehand, calculate again
            if given: apply this energy change, avoid repetitive energy change calculation. 
        """
        # TODO: change to FASStep class
        p,c,floor0,move_parent,energy_change=step.p,step.c,step.floor0,step.move_parent,step.energy_change
        hc, hp = self.h_of_node[c], self.h_of_node[p]
        if energy_change is None:
            energy_change=self.calculate_energy_change(p,c,floor0=floor0,move_parent=move_parent)
        self.energy+=energy_change

        if hp<hc:
            raise ValueError("Not a feedback arc.")
        if move_parent:
            self.h_of_node[self.h[hc:hp]] += 1
            self.h_of_node[p] = hc
            self.h[hc + 1:hp + 1] = self.h[hc:hp]
            self.h[hc] = p
        else:
            self.h_of_node[self.h[hc + 1:hp + 1]] -= 1
            self.h_of_node[c] = hp
            self.h[hc:hp] = self.h[hc + 1:hp + 1]
            self.h[hp] = c
        
        # TODO: update fas_ind efficiently
        self.fas_ind=self.get_FAS_ind()

    def calculate_energy_change(self,p,c,floor0=False,move_parent=True):
        """ Return updated energy after exchanging n1 and n2 in the hierarchy (self.h). Complexity is O(max_degree).   
        
        p, c: nodes to be moved. Assume p is below c: --c--------p--
        floor0: whether floor energy change to 0. [TODO: unclear what this is for]
        move_parent: Two ways to move nodes:
                    True if move p to just above c:     --pc----------
                    False if move c to just below p:    ----------pc--
        """
        hp,hc=self.h_of_node[p],self.h_of_node[c]
        if hp<hc:
            raise ValueError("p is above c!") # p is above c, so no need to exchange them. 
        
        pp, pc = self.graph.parents_of[p], self.graph.children_of[p]
        hpp, hpc = self.h_of_node[pp], self.h_of_node[pc]
        cp, cc = self.graph.parents_of[c], self.graph.children_of[c]
        hcp, hcc = self.h_of_node[cp], self.h_of_node[cc]

        if move_parent:
            e = np.count_nonzero(np.logical_and(hc < hpp, hpp < hp)) - np.count_nonzero(np.logical_and(hc <= hpc, hpc < hp))
        else:
            e = -np.count_nonzero(np.logical_and(hc < hcp, hcp <= hp)) + np.count_nonzero(np.logical_and(hc < hcc, hcc < hp))

        if floor0:
            # Not clear what this was for
            raise NotImplementedError
            e= max(0, e)
        
        return e
        
    def get_FAS_ind(self,h_of_node=None):
        """ Calculate feedback arc set from scratch give hierarchy h_of_node. .
        
        This function costs O(E) and should not be called often. 
        """
        if h_of_node is None:
            h_of_node=self.h_of_node
        return np.nonzero(h_of_node[self.graph.edges[:, 0]]> h_of_node[self.graph.edges[:, 1]])[0]

    def _check_updated_energy(self):
        # TODO
        pass

    def calculate_levels(self,fas_ind=None):
        """Calculate levels.
        
        Return: 
            levels: ndarray. Level # of each node
        """
        if fas_ind is None:
            fas_ind=self.fas_ind
        ff_edges=self.graph.edges[np.setdiff1d(np.arange(len(self.graph.edges)),fas_ind)]
        ff_graph=DirectedGraph(self.graph.n_node,ff_edges)
        levels=np.zeros(self.graph.n_node,dtype=int)

        for node in self.h:
            children=ff_graph.children_of[node]
            levels[children]=np.maximum(levels[children],levels[node]+1)
        
        return levels


class CPDecomposition(HeuristicProblem):
    """ CPDecomposition problem. 

    Algorithm 1:
    0. p: tensor order; k rank; In: dimension along nth order 
    1. Start with p (In, k) random unitary matrices, u_q (q=1,...,p)
    2. Each step, make p (In, In) random unitary matrices v_q. 
    3. In each step, u_q->u_q v_q^\\alpha. \\alpha is a tuned parameter (remains same or design a schedule)

    Algorithm 2 (EO): 
    0. Initialize k random rank 1 tensors: U_k=u_1k*u_2k...*u_pk. E=|T-\\sum_k u_1k*...*u_pk|^2
    2. Calculate gradient \\grad_(u_pk^i_n) E, and use convex optimizers to a local minima (e.g. gradient descent) (TODO: use stochastic gradient descent?)
    3. E_k=-2<T, U_k>+\\sum_l<U_k,U_l>. Select a U_k by probability P(k)\\prop E_k^\\tau. Random walk vectors[k]. 
    4. Repeat 2 until convergence

    Attributes:
        T: ndarray(I1,I2,...,Ip), tensor
        shape: T.shape
        p: int, tensor order
        K: int, total rank, K>=1
        vectors: ndarray(k,I1+...Ip). kth row is kth rank's vectors connected in order.
        # gradients: same shape as vectors; gradients
        Us: ndarray(k,I1,...,Ip), Us[k]=u_1k*...*U_pk
        energy: |T-\\sum_k U_k|^2 

        loc_initial: mean of vectors initialization
        scale_initial: standard deviation of vectors initialization
        scale_resample: standard deviation of randomization 
        optimizer: optimizer for converging to local minima.

    """
    
    def __init__(self, T,K,optimizer: ConvexOptimizer,
            loc_initial=0,scale_initial=1.0,scale_resample=1.0):
        
        self.T=T
        self.shape=self.T.shape
        self.p=len(T.shape) 
        self.K=K

        self.loc_initial=loc_initial
        self.scale_initial=scale_initial
        self.scale_resample=scale_resample
        self.optimizer=optimizer
        
        self.vectors=None
        self._randomize_vectors(k=None) 
        self.Us=self._calculate_Us()
        self.T_approximate=self._calculate_T_approximate()
        self.energy=self.calculate_energy()

        self._local_step_candidates=[self.CPDStep(k,self.optimizer) for k in range(self.K)]
          
    def get_data(self):
        return self.T
    
    def get_state(self):
        return self.vectors
    
    def calculate_energy(self):
        return self._tensor_norm_square(self.T-self.T_approximate)

    class CPDStep(HeuristicProblem.HeuristicStep):
        """Collecting parameters used in each optimization step. 
        
        k: kth rank to be randomized
        optimizer: ConvexOptimizer to be used to the local minima
        """

        def __init__(self, k,optimizer):
            self.k=k
            self.optimizer=optimizer

    def get_local_step_candidates(self,return_energy_change=False):
        """Can only accept return_energy_change=False because the other way makes no sense. 
        
        Energy calculation:
               E=||T-\sum_k U_k||^2=||T||^2+\sum_k (-2<U_k,T>)+\sum_k,l <U_k,U_l>
                =||T||^2+\sum_k (-2<U_k,T>+\sum_l <U_k,U_l>)
                =C+\sum_k <U_k, -2T+T_approximate>
        """
        if return_energy_change:
            raise NotImplementedError
        else:
            energies=np.array([self._tensor_inner_product(self.Us[k],-2*self.T+self.T_approximate) for k in range(self.K)])+self._tensor_norm_square(self.T)/self.K
        return self._local_step_candidates,energies
    
    def make_local_step(self,step:CPDStep):
        """Make a step in local search: 
            1. randomize step.kth rank vectors
            2. Converge to a local minima
            3. Update Us,T_approximate, energy
        """

        self._randomize_vectors(k=step.k)
        
        vectors_flatten_optimized=step.optimizer.optimize(self._flatten_vectors(self.vectors),lambda _: self._gradients(_,recalculate_T_approximate=True))
        self.vectors=self._deflatten_vectors(vectors_flatten_optimized)

        self.Us=self._calculate_Us()
        self.T_approximate=self._calculate_T_approximate()
        self.energy=self.calculate_energy()

    def calculate_energy_change(self):
        """This probelm has no efficient energy change calculation."""
        raise NotImplementedError

    def _check_updated_energy(self):
        """This problem has no efficient energy change calculation."""
        raise NotImplementedError

    def _gradients(self,vectors_flatten,recalculate_T_approximate=False):
        """ Calculate gradients. 
        
        Derivation:
        (Notation: \sum_\\in    = \sum_i1...in-1in+1...ip (summing excluding index in),
                   ...\\n...    : excluding n in this sequence)
                   
                    E   =T^2+\sum_k \\sum_i1...ip (-2)T^i1...ip u_1k^i1...u_pk^ip
                            +\sum_k \\sum_i1...ip (u_1k^i1...u_pk^ip)^2
                            +\sum_l\\neqk \sum_i1...ip u_1k^i1...u_pk^ip u_1l^i1...u_pl^in
        grad_(u_nk^in)  = -2\sum_\\in T^i1...ip u_1k^i1...\\n...u_pk^ip 
                          +2\sum_\\in (u_1k^i1...\\n...u_pk^ip)^2 u_nk^in
                          +2\sum_\\in \\sum_l\\neq k u_1k^i1...\\n...u_pk^ip u_1l^i1...u_pl^ip
                        = -2\sum_\\in u_1k^i1...\\n...u_pk^ip (T-T_approx)^i1...ip
                        = -2 / u_ik^in * \\sum_\\in U_k^i1...ip (T-T_approx)^i1...ip
                (The code below further uses vectorization on k and in)

        vectors_flatten: ndarray(K*(I1+...+Ip),) 
        Return: gradients as a 1d vector of length K*(I1+...+Ip)

        """
        
        vectors=self._deflatten_vectors(vectors_flatten)
        gradients=np.zeros(vectors.shape)
        _inds=np.add.accumulate((0,*self.shape))
        if not recalculate_T_approximate:
            Us=self.Us
            T_approximate=self.T_approximate
        else:
            Us=np.array([self._calculate_U(vectors[k,:]) for k in range(self.K)])
            T_approximate=np.sum(Us,axis=0)
        _error_tensor=np.tile(self.T-T_approximate,[self.K]+[1]*self.p)
        for n in range(1,self.p+1):
            _einsum_ind=list(range(self.p+1))
            gradients[:,_inds[n-1]:_inds[n]]=-2/vectors[:,_inds[n-1]:_inds[n]]*np.einsum(Us,_einsum_ind,_error_tensor,_einsum_ind,[0,n])
        
        return self._flatten_vectors(gradients)
            
    def _flatten_vectors(self,vectors):
        """ Return a 1d vector of self.vectors(K*(i1+i2+...),)"""
        return np.concatenate(vectors,axis=0)
    
    def _deflatten_vectors(self,vectors_flatten):
        return vectors_flatten.reshape((self.K,-1))

    def _randomize_vectors(self, k=None):
        """Return randomize vectors.
        
        Different ways to do this:
            1. From what sample? 
            2. Random walk from the past point, or resample from scratch?
        
        k: if None: resample all K ranks' vectors from scratch
            else:   random walk kth rank' vectors
        """

        if k is None:
            self.vectors=np.random.normal(loc=self.loc_initial,scale=self.scale_initial,size=(self.K,np.sum(self.T.shape)))
        else:
            self.vectors[k]+=np.random.normal(loc=0,scale=self.scale_resample,size=self.vectors.shape[1])

    def _calculate_T_approximate(self):
        return np.sum(self.Us,axis=0)
    
    def _calculate_Us(self):
        """Given vectors of shape (k,I1+...+Ip), return rank-1 tensors of shape (k,I1,...,Ip)."""
        
        return np.array([self._calculate_U(self.vectors[k,:]) for k in range(self.K)])
    
    def _calculate_U(self,vector):
        """Given a vector of shape (I1+...+Ip,), return a rank-1 tensors in (I1,...,Ip)"""

        return self._outer(self._split_vector(vector,self.shape))
    
    def _outer(self,vectors):
        """Outer product of many vectors (expand np.outer).

        See https://stackoverflow.com/questions/43148829/how-to-3-way-outer-product-in-numpy

        vectors: an array of vectors. 
            If vector lengths are I1,...,Ip, the product is (I1,...,Ip)
        """
        return np.prod(np.ix_(*vectors))
    
    def _split_vector(self,vector,shape):
        """Split a 1d vector of (I1+...+Ip,) into a list of vectors of lengths I1,...,Ip.
        
        vector: a 1d vector of length (I1+...+Ip,)
        shape: (I1,...,Ip)
        """
        
        return np.split(vector,np.add.accumulate(shape))[:-1]
    
    def _tensor_norm_square(self,A):
        """ Square of Frobenius tensor norm. 

        ||A||**2=<A,A>
        """
        return self._tensor_inner_product(A,A)

    def _tensor_inner_product(self,A,B):
        """Frobenius tensor inner product. 
        
        <A,B>=\\sum_{i1...ip} A^{i1...ip}B^{i1...ip}
        """
        return np.sum(A*B)


if __name__=="__main__":
    pass
    # Ideas
    # Other tensor decomposition problems?
    # 
    # class MinimumFeedbackVertex(HeuristicProblem):
    #     """ Minimum feedback vertex problem. 
    #     """
    #     pass
    #
    # class TravelingSalesman(HeuristicProblem):
    #     """ Traveling salesman problem.
    #     """
    #
    #     def __init__(self,data):
    #         pass
    #
    #     def update_energy(self):
    #         pass
    #
    # class ExamTimetabling(HeuristicProblem):
    #     """ Exam timetabling problem.
    #     """
    #     pass

