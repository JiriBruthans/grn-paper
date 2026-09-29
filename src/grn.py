#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.special
import networkx as nx
from smallworld import grouped_scale_free_graph


_README="""
A script to generate synthetic data from gene regulatory networks with grouped small-world 
 network structure. 

Author: Matthew Aguirre (SUNET: magu)
"""


def main():
    # get graph generation parameters and whether we want to make plots
    import argparse
    parser = argparse.ArgumentParser(description=_README)
    parser.add_argument('--out',        type=str, required=True, metavar='my_grn', default='my_grn')
    parser.add_argument('--num-genes',  type=int, required=False, metavar=2000, default=2000)
    parser.add_argument('--num-groups', type=int, required=False, metavar=1, default=1)
    parser.add_argument('--r'    ,      type=float, required=False, metavar=4, default=4)
    parser.add_argument('--delta-in',   type=float, required=False, metavar=100, default=100)
    parser.add_argument('--delta-out',  type=float, required=False, metavar=1, default=1)
    parser.add_argument('--w',          type=float, required=False, metavar=10, default=10)
    parser.add_argument('--kos',        action='store_true')
    parser.add_argument('--cores',      type=int, required=False, metavar=1, default=1)
    args = parser.parse_args()
    
    # dump log
    pd.DataFrame(vars(args), index=['value']).T.to_csv(args.out+'.log', sep='\t')
     
    # generate gene regulatory network
    G = grn().add_structure(n = args.num_genes, 
                            k = args.num_groups, 
                            alpha = 1e-99, 
                            beta  = 1 - 1./args.r, 
                            gamma = 1./args.r, 
                            kappa = args.w,
                            delta_in  = args.delta_in, 
                            delta_out = args.delta_out
                           )
    # save these useful matrixes: all pairs path distances and module affinity map
    G.dist = pd.DataFrame(dict(nx.all_pairs_shortest_path_length(G))).values
    G.module = np.array([[G.groups[i] == G.groups[j] for j in G.nodes()] for i in G.nodes()])
     
    # make perturbations
    if args.kos:
        G.ko = G.ko_all_nodes(n_jobs = args.cores)

    # save to file
    nx.write_gpickle(G, args.out + '.gpickle')
    


class grn(nx.DiGraph):
    def __init__(self, adjacency_matrix=None, groups=None):
        super(grn, self).__init__()
        # load provided weights if given, otherwise this is an empty graph
        if adjacency_matrix is not None:
            nx.to_networkx_graph(adjacency_matrix != 0, create_using=self)
            self.n = self.number_of_nodes()
            self.groups = groups
            self.beta = adjacency_matrix
            self.add_expression_parameters()

    
    def add_structure(self, n, k, alpha, beta, gamma, delta_in, delta_out, kappa, expression_params=True):
        # wrapper for grouped_scale_free_graph
        G = grouped_scale_free_graph(n=n, k=k, alpha=alpha, beta=beta, gamma=gamma, 
                                     delta_in=delta_in, delta_out=delta_out, kappa=kappa)
        nx.to_networkx_graph(G, create_using=self)
        
        # store relevant info in self
        self.n = self.number_of_nodes()
        self.groups = nx.get_node_attributes(self, 'group').copy()
        
        # add parameters for RNA model
        if expression_params:
            self.add_expression_parameters(G)
        
        return self

    
    def add_expression_parameters(self, G=None, inflate_edges=True):
        # these are for the gene expression function
        self.link = scipy.special.expit
        self.observe_rna = self.observation_model
        
        # set edge weights in self.beta if we weren't given them to start
        if G is not None: 
            S = np.random.normal(0, 1, size=(self.n, self.n))
            #E = nx.convert_matrix.to_numpy_matrix(self, multigraph_weight=sum)
            E = nx.to_numpy_array(self, multigraph_weight=sum)
            # remove self loops
            E -= np.diag(np.diag(E))
            if inflate_edges:
                S += np.sign(S) 
            self.beta = np.array(np.multiply(S, E))
        
        # generate gene attributes: production rate alpha, degradation rate l
        self.alpha = scipy.special.logit(np.random.beta(2, 8, self.n)).reshape(-1,1)
        self.l = np.maximum(self.link(-self.alpha), np.random.beta(8, 2, self.n).reshape(-1,1))
        
        return self
    

    def simulate_rna(self, x0=None, alpha=None, beta=None, l=None, link=None,
                     s=1e-4, dt=1e-2, tmax=20000, n=1, tol=1e-3, step=1000, burnin=5000, save=True,
                     production_scale=1., extra_production=0., clamp=None, return_info=False):
        # n is n_samples and self.n is n_genes
        # simulate gene expression according to an SDE regulatory model:
        #     X(t+dt) = X(t) + dt * [link(alpha + beta.T X) - l*X + N(0, s^2 X / dt)]
        
        # setup
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if l is None:
            l = self.l
        if link is None:
            link = self.link
        
        # traceline goes here (this could be made more efficient)
        X = np.zeros((n, tmax, self.n)) 
         
        # set initial condition
        if x0 is None:
            X[:,0,:] = np.zeros((n,self.n))  #np.random.random((n,self.n))
        elif x0.shape == (n, self.n):
            X[:,0,:] = x0
        elif x0.shape == (self.n,n):
            X[:,0,:] = x0.T
        elif x0.shape == (self.n,):
            X[:,0,:] = np.vstack([x0 for _ in range(n)])
        if clamp is not None:
            clamp = np.broadcast_to(np.asarray(clamp, dtype=float), (n, self.n))
            fixed = np.isfinite(clamp)
            X[:, 0, :][fixed] = clamp[fixed]
       
        # run simulations: x.shape=(tmax,self.n)
        converged = False
        for i in range(tmax-1):
            dpos = production_scale * link(alpha.T + X[:,i,:] @ beta) + extra_production
            dneg = l.T * X[:,i,:]
            X[:,i+1,:] = X[:,i,:] + dt*(dpos - dneg)
            X[:,i+1,:] += s * np.sqrt(dt * X[:,i,:]) * np.random.normal(0, 1, size=(n, self.n))
            X[:,i+1,:] = np.maximum(0, X[:,i+1,:]) # clip negative values
            if clamp is not None:
                X[:, i+1, :][fixed] = clamp[fixed]

            # check convergence
            if i % step == 0 and i - step > burnin:
                now  = X[0, burnin:i, :].mean(axis=0)
                then = X[0, burnin:(i-step), :].mean(axis=0)
                if np.max(np.abs(np.log2(now/then))[now > s]) < tol:
                    X = X[:, :i, :]
                    converged = True
                    break        
        
        # pass into observation model
        X1 = self.observe_rna(X[:, burnin:i, :])
        # done!
        if save:
            self.converged = converged
            self.s = s
            self.dt = dt
            self.tol = tol
            self.rna = X1[0]
        if return_info:
            return X1, {'converged': converged, 'steps': i + 1}
        return X1

    
    def observation_model(self, X, t=100000):
        return np.mean(X[:,-t:,:], axis=1)

    
    def set_rna_observation_model(self, f):
        self.observe_rna=lambda X: f(X)
        return self
    
    
    def perturb(self, new_alpha=None, new_beta=None, new_l=None, **kwargs):
        # self.simulate_rna will automatically populate old parameters if new ones are empty 
        #  and additional keyword arguments are passed on
        return self.simulate_rna(save=False, alpha=new_alpha, beta=new_beta, l=new_l, **kwargs)

    
    def ko_all_nodes(self, n_jobs=1, **kwargs):
        # use parallel processing and keep a fun little progress bar
        from joblib import Parallel, delayed
        from tqdm import tqdm
        
        # compute baseline rna if we need it
        if not hasattr(self, 'rna'):
            self.simulate_rna(save=True, **kwargs)
        
        # do each ko in turn, using the helper function below
        return np.array(Parallel(n_jobs = n_jobs, pre_dispatch = 'n_jobs', prefer='processes'
                                )(delayed(self.ko_one_node)(i) for i in tqdm(range(self.n))))
    

    def ko_one_node(self, gene, stat='logfc', **kwargs):
        # gene is an index from 0, ..., self.n - 1
        new_beta = self.beta.copy()
        new_beta[gene,:] = 0
        
        # kwargs get passed directly into self.simulate_rna by way of self.perturb
        new_rna = self.perturb(new_beta = new_beta, x0 = self.rna, **kwargs)
       
        # meh 
        if stat == 'logfc':
            return np.log2(new_rna.flatten()) - np.log2(self.rna.flatten())

    def ko_nodes(self, genes, stats=('new_rna',), **kwargs):
        genes = np.atleast_1d(np.asarray(genes))

        # make sure the baseline steady state exists
        if not hasattr(self, 'rna'):
            self.simulate_rna(save=True, **kwargs)

        new_beta = self.beta.copy()
        new_beta[genes, :] = 0                       # nullify outgoing edges [1]
        new_rna, info = self.perturb(new_beta=new_beta, x0=self.rna,
                                   return_info=True, **kwargs)

        outputs = {}
        if 'logfc' in stats:
            outputs['logfc'] = np.log2(new_rna.flatten()) - np.log2(self.rna.flatten())
        if 'new_rna' in stats:
            outputs['new_rna'] = new_rna
        if 'convergence' in stats:
            outputs['convergence'] = info['converged']
        return outputs

    def simulate_steady_state(self, x0=None, production_scale=1.,
                              extra_production=0., alpha_shift=0., clamp=None,
                              dt=0.2, max_steps=5000, atol=1e-10, rtol=1e-8,
                              check_every=25, save=False):
        """Deterministic, memory-bounded integration of the original RNA ODE.

        Conditions can be batched as (conditions, genes) arrays. The solver uses
        explicit midpoint integration with local step subdivision to preserve
        nonnegative expression, not a root finder: it follows dynamics
        from x0 and does not accept an unreachable/unstable mathematical root.
        A row converges only if every free gene has |dx/dt| <= atol + rtol*|x|.
        Unconverged rows are returned with a False flag; callers must exclude
        them from equilibrium analyses. No stochastic or observation noise is
        included. Use simulate_rna for the original SDE/time-average estimator.
        """
        from scipy.sparse import csr_matrix

        if not np.isfinite(dt) or dt <= 0 or dt * np.max(self.l) >= 1:
            raise ValueError('Require 0 < dt * max(l) < 1 for positive integration.')
        if max_steps < 1 or check_every < 1 or atol <= 0 or rtol < 0:
            raise ValueError('Invalid step count or tolerance.')
        x = np.asarray(np.zeros(self.n) if x0 is None else x0, dtype=float)
        x = np.atleast_2d(x).copy()
        if x.shape[1] != self.n or not np.all(np.isfinite(x)) or np.any(x < 0):
            raise ValueError('x0 must be finite, nonnegative, and have n genes.')
        arrays = []
        for value in (production_scale, extra_production, alpha_shift):
            arr = np.broadcast_to(np.asarray(value, dtype=float), x.shape).copy()
            if not np.all(np.isfinite(arr)):
                raise ValueError('Perturbation parameters must be finite.')
            arrays.append(arr)
        scale, extra, shift = arrays
        if np.any(scale < 0) or np.any(extra < 0):
            raise ValueError('Production scale and extra production must be nonnegative.')
        fixed_values = np.full_like(x, np.nan) if clamp is None else np.broadcast_to(clamp, x.shape).copy()
        if np.any(np.isinf(fixed_values)) or np.any(fixed_values[np.isfinite(fixed_values)] < 0):
            raise ValueError('Clamp values must be nonnegative or NaN (free).')
        fixed = np.isfinite(fixed_values)
        x[fixed] = fixed_values[fixed]
        alpha = self.alpha.ravel()
        decay = self.l.ravel()
        # beta[source, target]; multiplying beta.T by x.T preserves that convention.
        matrix = csr_matrix(self.beta.T)
        converged = np.zeros(len(x), dtype=bool)
        steps = np.full(len(x), max_steps, dtype=int)
        residual = np.full(len(x), np.inf)
        step_halvings = np.zeros(len(x), dtype=int)
        active = np.arange(len(x))

        def rhs(y, rows):
            value = scale[rows] * self.link(alpha + shift[rows] + (matrix @ y.T).T)
            value += extra[rows] - decay * y
            value[fixed[rows]] = 0.
            return value

        def advance(y, rows, h, drift=None, depth=0):
            if drift is None:
                drift = rhs(y, rows)
            midpoint = y + 0.5 * h * drift
            updated = y + h * rhs(midpoint, rows)
            if np.any(~np.isfinite(updated)):
                raise FloatingPointError('Non-finite integration state; reduce dt.')
            negative = np.any(updated < -atol, axis=1)
            if np.any(negative):
                if depth >= 12:
                    raise FloatingPointError('Integration failed; reduce dt.')
                affected = rows[negative]
                step_halvings[affected] += 1
                half = advance(y[negative], affected, h / 2, depth=depth + 1)
                updated[negative] = advance(half, affected, h / 2, depth=depth + 1)
            updated[fixed[rows]] = fixed_values[rows][fixed[rows]]
            return np.maximum(updated, 0.)

        for iteration in range(max_steps + 1):
            y = x[active]
            drift = rhs(y, active)
            if iteration % check_every == 0 or iteration == max_steps:
                err = np.max(np.abs(drift) / (atol + rtol * np.abs(y)), axis=1)
                residual[active] = np.max(np.abs(drift), axis=1)
                done = err <= 1.
                converged[active[done]] = True
                steps[active[done]] = iteration
                active = active[~done]
                if len(active) == 0 or iteration == max_steps:
                    break
                y, drift = y[~done], drift[~done]
            x[active] = advance(y, active, dt, drift=drift)
        if save:
            if len(x) != 1:
                raise ValueError('save=True requires one wild-type condition.')
            self.rna = x[0].copy()
            self.converged = bool(converged[0])
        return {'new_rna': x, 'convergence': converged,
                'steps': steps, 'max_abs_drift': residual, 'step_halvings': step_halvings}

    def expression_parameters(self, genes, fold_change, mechanism='clamp',
                              saturation='raise'):
        """Build parameters without changing this graph or its WT expression.

        production: multiply endogenous synthesis capacity (KD or activation).
        promoter: shift alpha so synthesis changes by fold_change at the WT
          regulatory input; constrained by the original sigmoid ceiling.
        transgene: add (fold_change-1)*l*x_WT to synthesis, outside the sigmoid.
        clamp: hold target RNA at fold_change*x_WT, a dose-matched idealization.

        For production/promoter/transgene, feedback can change the achieved
        fold. These are phenomenological interventions, not guide-level models.
        """
        if not hasattr(self, 'rna'):
            raise ValueError('Compute WT RNA first (simulate_rna or simulate_steady_state).')
        raw = np.atleast_1d(genes)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('genes must contain integer matrix indices.')
        genes = raw.astype(int)
        if np.any((genes < 0) | (genes >= self.n)) or len(np.unique(genes)) != len(genes):
            raise ValueError('Gene indices must be unique and within the network.')
        folds = np.broadcast_to(np.asarray(fold_change, dtype=float), genes.shape)
        if np.any(~np.isfinite(folds)) or np.any(folds < 0):
            raise ValueError('fold_change must be finite and nonnegative.')
        wt = np.asarray(self.rna).ravel()
        if wt.shape != (self.n,) or np.any(~np.isfinite(wt)) or np.any(wt < 0):
            raise ValueError('Invalid WT expression.')
        pars = dict(production_scale=np.ones(self.n), extra_production=np.zeros(self.n),
                    alpha_shift=np.zeros(self.n), clamp=np.full(self.n, np.nan))
        capped = np.zeros(len(genes), dtype=bool)
        if mechanism == 'production':
            pars['production_scale'][genes] = folds
        elif mechanism == 'transgene':
            if np.any(folds < 1):
                raise ValueError('Transgene overexpression requires fold_change >= 1.')
            pars['extra_production'][genes] = (folds - 1) * self.l.ravel()[genes] * wt[genes]
        elif mechanism == 'clamp':
            pars['clamp'][genes] = folds * wt[genes]
        elif mechanism == 'promoter':
            if saturation not in ('raise', 'clip'):
                raise ValueError("saturation must be 'raise' or 'clip'.")
            if self.link is not scipy.special.expit:
                raise ValueError('Promoter calibration requires the logistic link.')
            z = self.alpha.ravel() + wt @ self.beta
            desired = folds * scipy.special.expit(z[genes])
            changed = folds != 1
            capped = (desired >= 1 - 1e-9) & changed
            if np.any(capped) and saturation == 'raise':
                raise ValueError("Requested activation exceeds the sigmoid ceiling. "
                                 "Use saturation='clip', production, or clamp explicitly.")
            positive = (folds > 0) & changed
            pars['alpha_shift'][genes[positive]] = scipy.special.logit(
                np.clip(desired[positive], np.finfo(float).tiny, 1 - 1e-9)) - z[genes[positive]]
            pars['production_scale'][genes[folds == 0]] = 0.
        else:
            raise ValueError('Unknown mechanism: ' + str(mechanism))
        return pars, {'genes': genes, 'requested_fold': folds.copy(), 'capped': capped}

    def expression_nodes(self, genes, fold_change, mechanism='clamp',
                         stats=('new_rna', 'logfc', 'convergence'),
                         solver='steady_state', saturation='raise', **kwargs):
        """Perturb one or several genes; returns the same dict style as ko_nodes.

        Default solver is deterministic. solver='trajectory' uses simulate_rna
        (including its s, burnin, and observation model). Zero expression has
        logFC=-inf; mask targeted genes before constructing interaction residuals.
        """
        pars, info = self.expression_parameters(genes, fold_change, mechanism, saturation)
        if solver == 'steady_state':
            result = self.simulate_steady_state(x0=self.rna, **pars, **kwargs)
            rna, converged = result['new_rna'], bool(result['convergence'][0])
            info.update(steps=int(result['steps'][0]), max_abs_drift=float(result['max_abs_drift'][0]))
        elif solver == 'trajectory':
            shift = pars.pop('alpha_shift')
            rna, diagnostics = self.simulate_rna(x0=self.rna, save=False,
                alpha=self.alpha + shift[:, None], return_info=True, **pars, **kwargs)
            converged = diagnostics['converged']
            info.update(diagnostics)
        else:
            raise ValueError("solver must be 'steady_state' or 'trajectory'.")
        with np.errstate(divide='ignore', invalid='ignore'):
            info['achieved_fold'] = rna[0, info['genes']] / np.asarray(self.rna).ravel()[info['genes']]
            logfc = np.log2(rna) - np.log2(np.asarray(self.rna).ravel())
        outputs = {'new_rna': rna, 'logfc': logfc.ravel(),
                   'convergence': converged, 'diagnostics': info}
        unknown = set(stats) - set(outputs)
        if unknown:
            raise ValueError('Unknown stats: ' + str(unknown))
        return {key: outputs[key] for key in stats}

    def knockdown_nodes(self, genes, remaining=0.1, mechanism='production', **kwargs):
        """CRISPRi/KD; remaining=0.1 means 90% lower synthesis capacity."""
        if np.any(np.asarray(remaining) < 0) or np.any(np.asarray(remaining) > 1):
            raise ValueError('remaining must be between 0 and 1.')
        return self.expression_nodes(genes, remaining, mechanism=mechanism, **kwargs)

    def crispri_nodes(self, genes, remaining=0.1, **kwargs):
        return self.knockdown_nodes(genes, remaining=remaining, **kwargs)

    def crispra_nodes(self, genes, fold_change=2., mechanism='production', **kwargs):
        """CRISPRa idealization: multiply endogenous synthesis capacity.

        mechanism='promoter' instead shifts alpha with the original synthesis
        ceiling; mechanism='clamp' fixes the achieved target dose exactly.
        """
        if np.any(np.asarray(fold_change) < 1):
            raise ValueError('CRISPRa requires fold_change >= 1.')
        return self.expression_nodes(genes, fold_change, mechanism=mechanism, **kwargs)

    def overexpress_nodes(self, genes, fold_change=10., mechanism='transgene', **kwargs):
        """Add a constitutive source, calibrated at the WT background."""
        if np.any(np.asarray(fold_change) < 1):
            raise ValueError('Overexpression requires fold_change >= 1.')
        return self.expression_nodes(genes, fold_change, mechanism=mechanism, **kwargs)



if __name__=="__main__":
    main()
