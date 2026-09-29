import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grn import grn
from modality_analysis import choose_pairs, screen, residuals, save_screen, load_screen


def network(beta=None):
    G = grn(np.zeros((4, 4)) if beta is None else np.asarray(beta, dtype=float))
    G.alpha = np.full((G.n, 1), -2.)
    G.l = np.full((G.n, 1), .9)
    G.simulate_steady_state(save=True)
    return G


class ModalitiesTests(unittest.TestCase):
    def test_independent_genes_analytic_and_no_interaction(self):
        G = network()
        wt = expit(-2.) / .9
        np.testing.assert_allclose(G.rna, wt, rtol=1e-7)
        for mechanism, folds in [('production', [.01, .5, 2, 10]),
                                 ('transgene', [2, 10, 30]), ('clamp', [0, .5, 2, 30])]:
            for fold in folds:
                result = G.expression_nodes([0, 1], fold, mechanism)
                self.assertTrue(result['convergence'])
                np.testing.assert_allclose(result['new_rna'][0, :2], fold * wt, atol=1e-8)
                np.testing.assert_allclose(result['new_rna'][0, 2:], wt, atol=1e-8)
                data = screen(G, choose_pairs(G.n), mechanism, fold)
                r = residuals(data)
                np.testing.assert_allclose(r['log'][r['mask']], 0, atol=1e-7)

    def test_null_identity_nonmutation_and_vector_doses(self):
        G = network([[0, 1, 0, 0], [0, 0, -2, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        before = [a.copy() for a in (G.beta, G.alpha, G.l, G.rna)]
        for mechanism in ('production', 'transgene', 'promoter', 'clamp'):
            out = G.expression_nodes([0, 2], 1, mechanism)
            np.testing.assert_allclose(out['new_rna'][0], G.rna, atol=1e-9)
        out = G.expression_nodes([0, 2], [.25, 3], 'clamp')
        np.testing.assert_allclose(out['new_rna'][0, [0, 2]], G.rna[[0, 2]] * [.25, 3])
        for a, b in zip(before, (G.beta, G.alpha, G.l, G.rna)):
            np.testing.assert_array_equal(a, b)

    def test_ko_matches_edge_deletion_away_from_targets(self):
        G = network([[0, 2, 1, 0], [0, 0, -2, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        clamped = G.expression_nodes([0, 1], 0)['new_rna'][0]
        B = G.beta.copy(); B[[0, 1]] = 0
        out = solve_ivp(lambda t, x: expit(G.alpha.ravel() + x @ B) - G.l.ravel()*x,
                        (0, 100), G.rna, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(clamped[2:], out.y[2:, -1], atol=1e-8)

    def test_batch_and_independent_adaptive_ode_agree(self):
        G = network([[0, 1, -1, 0], [.3, 0, 1, 0], [0, -.5, 0, 1], [0, 0, 0, 0]])
        for mechanism, dose in [('production', .1), ('production', 5), ('transgene', 10), ('promoter', 2)]:
            data = screen(G, np.array([[0, 1], [1, 2]]), mechanism, dose, batch_size=1)
            for k, genes in enumerate(data['pairs']):
                pars, _ = G.expression_parameters(genes, dose, mechanism, saturation='clip')
                rhs = lambda t, x: pars['production_scale']*expit(G.alpha.ravel()+pars['alpha_shift']+x@G.beta)+pars['extra_production']-G.l.ravel()*x
                sol = solve_ivp(rhs, (0, 100), G.rna, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(data['double'][k], sol.y[:, -1], atol=1e-7)
                direct = G.expression_nodes(genes, dose, mechanism, saturation='clip')['new_rna'][0]
                np.testing.assert_allclose(data['double'][k], direct, atol=1e-9)

    def test_promoter_ceiling_and_achieved_dose(self):
        G = network()
        out = G.crispra_nodes([0], 2, mechanism='promoter', stats=('new_rna', 'diagnostics'))
        np.testing.assert_allclose(out['diagnostics']['achieved_fold'], 2, rtol=1e-7)
        with self.assertRaises(ValueError):
            G.crispra_nodes([0], 100, mechanism='promoter')
        out = G.crispra_nodes([0], 100, mechanism='promoter', saturation='clip', stats=('new_rna', 'diagnostics'))
        self.assertTrue(out['diagnostics']['capped'][0])
        self.assertLess(out['new_rna'][0, 0], 1 / .9 + 1e-8)

    def test_large_regulatory_slope_preserves_nonnegative_expression(self):
        G = grn(np.array([[-1000.]]))
        G.alpha = np.array([[10.]])
        G.l = np.ones((1, 1))
        out = G.simulate_steady_state(x0=np.array([.01]), max_steps=20)
        self.assertTrue(np.all(out['new_rna'] >= 0))
        self.assertGreater(out['step_halvings'][0], 0)

    def test_failures_are_not_scored_as_equilibria(self):
        G = network()
        data = screen(G, choose_pairs(G.n), 'production', .1, solver={'max_steps': 1})
        self.assertFalse(data['double_converged'].any())
        self.assertFalse(residuals(data)['mask'].any())
        # Original trajectory KO must report this run, not the saved WT flag.
        G.converged = True
        with np.errstate(all='ignore'):
            out = G.ko_nodes([0], stats=('convergence',), s=0, tmax=10, burnin=2)
        self.assertFalse(out['convergence'])
        self.assertTrue(G.converged)

    def test_masks_remove_both_targets_and_floor(self):
        G = network()
        data = screen(G, np.array([[0, 1]]), 'clamp', 2)
        mask = residuals(data)['mask'][0]
        np.testing.assert_array_equal(mask, [False, False, True, True])
        data['double'][0, 2] = 1e-8
        self.assertFalse(residuals(data)['mask'][0, 2])
        self.assertTrue(residuals(data, floor_policy='wt')['mask'][0, 2])

    def test_invalid_requests_and_pair_sampling(self):
        G = network()
        for genes in ([0, 0], [-1], [G.n], [.5]):
            with self.assertRaises(ValueError): G.expression_nodes(genes, 2)
        for fold in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError): G.expression_nodes([0], fold)
        with self.assertRaises(ValueError): G.knockdown_nodes([0], 1.1)
        with self.assertRaises(ValueError): G.overexpress_nodes([0], .5)
        all_pairs = choose_pairs(256)
        self.assertEqual(len(all_pairs), 32640)
        sample = choose_pairs(256, 4096)
        self.assertEqual(len(np.unique(sample, axis=0)), 4096)
        self.assertTrue(np.all(sample[:, 0] < sample[:, 1]))
        self.assertTrue(np.all((sample >= 0) & (sample < 256)))


if __name__ == '__main__':
    unittest.main()
