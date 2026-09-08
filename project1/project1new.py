import sys
import itertools
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import loggamma
import networkx as nx


class Variable():
    def __init__(self, name: str, r: int):
        self.name = name
        self.r = r  # number of possible values

    def __str__(self):
        return "(" + self.name + ", " + str(self.r) + ")"


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def prior(variables: list[Variable], graph: nx.DiGraph):
    '''genertaing a prior alpha ijk where all entries are 1, takes in a list of variables vars and strcture g'''
    n = len(variables)
    r = [var.r for var in variables]
    q = np.array([int(np.prod([r[j] for j in graph.predecessors(i)])) for i in range(n)])
    return [np.ones((q[i], r[i])) for i in range(n)]

def statistics(variables: list[Variable], graph: nx.DiGraph, data: np.ndarray):
    '''this is a function for extracting counts from data, assuming a Bayesian network w/ variable and structure G
    it returns an array M of length n. the ith component consists of a qi *ri matrix of counts'''
    n = len(variables)
    r = [var.r for var in variables]
    q = np.array([int(np.prod([r[j] for j in graph.predecessors(i)])) for i in range(n)])
    M = [np.zeros((q[i], r[i])) for i in range(n)] #number of datapoints
    for o in data.T:
        for i in range(n):
            k = int(o[i])  # Ensure integer
            parents = list(graph.predecessors(i))
            j = 0
            if len(parents) != 0:
                j = np.ravel_multi_index(o[parents].astype(int), [r[p] for p in parents]) #[r[p] for p in parents]
            M[i][j, k] += 1.0
    return M


def bayesian_score_component(M: np.ndarray, alpha: np.ndarray) -> float:
    '''log baysian score'''
    alpha_0 = np.sum(alpha, axis=1)
    p = np.sum(loggamma(alpha + M))
    p -= np.sum(loggamma(alpha))
    p += np.sum(loggamma(alpha_0))
    p -= np.sum(loggamma(alpha_0 + np.sum(M, axis=1)))
    return p

def bayesian_score(variables: list[Variable], graph: nx.DiGraph, data: np.ndarray) -> float:
    '''sum the p from bayesian_score_component'''
    n = len(variables)
    M = statistics(variables, graph, data)
    alpha = prior(variables, graph)
    return np.sum([bayesian_score_component(M[i], alpha[i]) for i in range(n)])


class DirectedGraphSearchMethod(ABC):
    @abstractmethod
    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        pass


class K2Search(DirectedGraphSearchMethod):
    '''This variable ordering imposes a topological ordering in the resutlting graph'''
    def __init__(self, ordering: list[int]):
        self.ordering = ordering

    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(variables)))

        for k, i in enumerate(self.ordering[1:]):
            y = bayesian_score(variables, graph, data)
            
            while True:
                y_best, j_best = -np.inf, 0
                
                for j in self.ordering[:k]:
                    if not graph.has_edge(j, i):
                        graph.add_edge(j, i)
                        y_prime = bayesian_score(variables, graph, data)
                        if y_prime > y_best:
                            y_best, j_best = y_prime, j
                        graph.remove_edge(j, i)
                
                if y_best > y:
                    y = y_best
                    graph.add_edge(j_best, i)
                else:
                    break

        return graph



class K2SearchFast(DirectedGraphSearchMethod):
    def __init__(self, ordering: list[int]):
        self.ordering = ordering

    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(variables)))
        
        # Precompute statistics once for all nodes
        M = statistics(variables, graph, data)
        alpha = prior(variables, graph)
        
        # Cache individual node scores
        node_scores = {}
        for i in range(len(variables)):
            node_scores[i] = bayesian_score_component(M[i], alpha[i])

        for k, i in enumerate(self.ordering[1:], start=1):
            # Current score for node i
            y = node_scores[i]
            
            while True:
                y_best, j_best = -np.inf, 0
                
                for j in self.ordering[:k]:
                    if not graph.has_edge(j, i):
                        # Add edge and recompute only node i's score
                        graph.add_edge(j, i)
                        
                        # Recompute statistics and score only for node i
                        M_i = self._compute_node_statistics(variables, graph, data)
                        alpha_i = self._compute_node_prior(variables, graph)
                        y_prime = bayesian_score_component(M_i, alpha_i)
                        
                        if y_prime > y_best:
                            y_best, j_best = y_prime, j
                        
                        graph.remove_edge(j, i)
                
                if y_best > y:
                    y = y_best
                    graph.add_edge(j_best, i)
                    
                    # Update cached score for node i
                    M_i = self._compute_node_statistics(variables, graph, data)
                    alpha_i = self._compute_node_prior(variables, graph)
                    node_scores[i] = bayesian_score_component(M_i, alpha_i)
                else:
                    break

        return graph
    
    def _compute_node_statistics(self, variables: list[Variable], graph: nx.DiGraph, 
                                  data: np.ndarray, node_idx: int) -> np.ndarray:
        """Compute statistics for a single node."""
        r = [var.r for var in variables]
        parents = list(graph.predecessors(node_idx))
        
        if len(parents) == 0:
            q_i = 1
        else:
            q_i = int(np.prod([r[p] for p in parents]))
        
        M_i = np.zeros((q_i, r[node_idx]))
        
        for o in data.T:
            k = int(o[node_idx])
            if len(parents) != 0:
                j = np.ravel_multi_index(o[parents].astype(int), 
                                        np.array([r[p] for p in parents], dtype=int))
            else:
                j = 0
            M_i[j, k] += 1.0
        
        return M_i
    
    def _compute_node_prior(self, variables: list[Variable], graph: nx.DiGraph, 
                           node_idx: int) -> np.ndarray:
        """Compute prior for a single node."""
        r = [var.r for var in variables]
        parents = list(graph.predecessors(node_idx))
        
        if len(parents) == 0:
            q_i = 1
        else:
            q_i = int(np.prod([r[p] for p in parents]))
        
        return np.ones((q_i, r[node_idx]))
