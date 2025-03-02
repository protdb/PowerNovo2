import numpy as np
from cylp.cy import CyClpSimplex



class knapsackModel(object):
    def __init__(self,
                 residues_weights:np.array,
                 remaining_mass:int,
                 cost: np.array,
                 length: int,
                 max_pep_len:int,
                 low_bound:int,
                 max_nodes: int = 255,
                 max_seconds:int =0.05):
        self.mask = cost > 0
        self.weights = residues_weights[self.mask].astype(int)
        self.remaining_mass = remaining_mass
        self.items = len(self.weights)
        self.max_pep_len = max_pep_len
        self.cost = cost[self.mask].astype(float)
        self.length = length
        self.max_seconds = max_seconds
        self.low_bound = low_bound
        self.max_nodes = max_nodes



    def solve_cylp(self, n_iter=0):
        model = CyClpSimplex()
        n = len(self.weights)
        x = model.addVariable('x', n, isInt=True)
        model.optimizationDirection = 'max'
        model.objective = self.cost
        model.addConstraint(self.weights @ x <= self.remaining_mass)
        model.addConstraint(self.weights @ x >= self.low_bound)
        model.addConstraint(x <= 1)
        model.addConstraint(x >= 0)
        model.addConstraint(np.ones(n) @ x <= self.length + 1)
        cbcModel = model.getCbcModel()
        cbcModel.logLevel = False
        cbcModel.maximumSeconds = self.max_seconds * (n_iter + 1)
        cbcModel.maximumNodes = self.max_nodes

        cbcModel.solve()
        x_sol = np.array(cbcModel.primalVariableSolution['x'].round()).astype(int)
        return x_sol

    def solve_simplex(self, n_iter=0):
        model = CyClpSimplex()
        n = len(self.weights)
        x = model.addVariable('x', n, isInt=True)
        model.optimizationDirection = 'max'
        model.objective = self.cost
        model.addConstraint(self.weights @ x <= self.remaining_mass)
        model.addConstraint(self.weights @ x >= self.low_bound)
        model.addConstraint(x <= 1)
        model.addConstraint(x >= 0)
        model.addConstraint(np.ones(n) @ x <= self.length + 1)
        model.addConstraint(np.ones(n) @ x >= self.length - 1)
        model.logLevel = False
        model.dualWithPresolve()
        x_sol = np.array(model.primalVariableSolution['x'].round()).astype(int)
        return x_sol












