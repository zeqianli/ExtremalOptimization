import numpy as np 
import scipy.sparse as sp

class Graph:
    """ Parent class for graphs
    """

    def __init__(self,n_node,edges,names=None):
        """ Initialize a graph (undirected or directed). Forbid self edge for now. 

        Processing:
            - Check that 0<=edges<n_node
            - Remove repetitive edges
            - Remove self edges
        n_node: # nodes
        edges: ndarray(n, 2), edges. Each entry must be in [0,n_node).
        names: names of nodes

        """

        self.n_node=n_node
        self.names=names
        
        # check entries in edges are legal (0<=edges< n_node)
        if not (np.all(edges>=0) and np.all(edges<n_node)):
            raise ValueError("Illegal edge entry. Edge entries must be in [0,n_mode).")
        
        # remove repetitive edges
        _edge_hash=edge_hash = edges[:, 0] * self.n_node + edges[:, 1] 
        _,_unique_edge_ind=np.unique(_edge_hash, return_index=True)
        if len(_unique_edge_ind)<len(edges):
            print("WARNING: found repeated edges. They are removed.")
        edges=edges[_unique_edge_ind,:]

        # remove self edges
        _is_self_edge=(edges[:,0]==edges[:,1])
        if np.any(_is_self_edge):
            print("WARNING: found self edges. They are removed.")
        self.edges=np.array(edges[np.logical_not(_is_self_edge)].copy())
        self.n_edge=len(self.edges)

        # multiple ways to accessing nodes/edges
        self.adjacency_matrix=None # sp.coo_matrix(n_node,n_node); adjacency matrix
        self.neighbors_of=None # dict{i: ndarray[neighbors of i]}; neighbors of a node; directed graph has parents_of and children_of.
        self.edges_of=None # dict{i: ndarray[edges connected to i]}; edges connected to a node; directed graph has in_edge_of and out_edge_of. 
        self.degrees=None # ndarray(n_node,); degrees; directed graph needs in_degrees and out_degrees.
        self.max_degree=None # maximum degree; directed graph needs max_in_degree and max_out_degree.
        
        # strongly connected components
        self.n_scc=None # number of strongly connected components
        self.scc_ind=None # list[ndarray[indices of ith scc]]; indices of strongly connected components 

class DirectedGraph(Graph):
    """ Directed graph class
    """
    def __init__(self,n_node,edges,names=None):
        """ Initialize a directed graph
        
        n_node: # nodes
        edges: ndarray(n, 2), directional edges; Each entry must be in [0,n_node).
        """
        Graph.__init__(self,n_node,edges,names=names)
        
        self.adjacency_matrix=sp.coo_matrix((np.ones(self.n_edge),(self.edges[:,0],self.edges[:,1])),shape=(self.n_node,self.n_node))

        self.parents_of={n:[] for n in range(self.n_node)}
        self.children_of={n:[] for n in range(self.n_node)}
        self.neighbors_of={n:[] for n in range(self.n_node)}
        self.in_degrees=np.zeros(self.n_node,dtype=int)
        self.out_degrees=np.zeros(self.n_node,dtype=int)
        self.degrees=np.zeros(self.n_node,dtype=int)

        for p,c in self.edges:
            self.parents_of[c]+=[p]
            self.children_of[p]+=[c]
            self.in_degrees[c]+=1
            self.out_degrees[p]+=1
        self.neighbors_of={n:self.parents_of[n]+self.children_of[n] for n in range(self.n_node)}  
        self.degrees=self.in_degrees+self.out_degrees  

        self.max_in_degree=np.max(self.in_degrees)
        self.max_out_degree=np.max(self.out_degrees)
        self.max_degree=np.max(self.degrees)

        self.n_scc, self.scc_ind = sp.csgraph.connected_components(self.adjacency_matrix, directed=True, connection='strong',return_labels=True)  # scc is indexed automatically by its feed-forward order


    def reduced_graph(self, return_double_edges=False):
        """ Reduce the graph by removing double edges. 
        """
        _hash=self.edges[:,0]*self.n_node + self.edges[:,1]
        _hash_reverse=self.edges[:,1]*self.n_node + self.edges[:,0]
        is_double_edge=np.in1d(_hash,_hash_reverse)
        reduced_graph=DirectedGraph(self.n_node,self.edges[np.logical_not(is_double_edge),:])

        if return_double_edges:
            return reduced_graph,self.edges[is_double_edge]
        else:
            return reduced_graph


class UndirectedGraph(Graph):
    """ Undirected graph class
    """
    
    def __init__(self, n_node, edges, inspect=True):
        """ Initialize a undirected graph
        
        n_node: # nodes
        edges: ndarray(n, 2), directional edges; Each entry must be in [0,n_node).
        """

        raise NotImplementedError
        Graph.__init__(self,n_node,edges)
        
        raise NotImplementedError
        #TODO: make edges double 

        self.adjacency_matrix=None # TODO

        self.neighbors_of=None

        self.degrees=None
        self.max_degree=np.max(self.degrees)

        self.n_scc, self.scc_ind = sp.csgraph.connected_components(self.adjacency_matrix, directed=False, connection='strong',return_labels=True)  # scc is indexed automatically by its feed-foward order

        pass

def load_graph_from_file(filename):
    raise NotImplementedError
    
