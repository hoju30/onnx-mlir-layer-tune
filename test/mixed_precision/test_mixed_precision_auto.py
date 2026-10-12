"""Check automatic decisions against exhaustive tiny assignment problems."""
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/mixed_precision'))
from mp_auto import auto_select, auto_select_3d


class AutomaticSelection(unittest.TestCase):
    def setUp(self):
        self.nodes = [{'name': 'A', 'weight_const': True},
                      {'name': 'B', 'weight_const': True}, {'name': 'relu'}]
        self.elems, self.flops = {'A': 1, 'B': 2}, {'A': 3, 'B': 1}
        self.formats = ['int4', 'int8', 'fp32']
        self.bits = {'int4': 4, 'int8': 8, 'fp32': 32}

    def cost(self, n, w, a):
        return (1 if n == 'A' else 2) * (
            {'int4': 5, 'int8': .02, 'fp32': 0}[w] +
            {'int4': 10, 'int8': .01, 'fp32': 0}[a])

    def select(self, value=None, **kw):
        return auto_select(self.nodes, self.formats, self.formats,
                           value or self.cost, self.elems, self.flops,
                           samples=7, refinements=1, **kw)

    def test_every_sample_matches_exhaustive_optimum(self):
        answer = self.select()
        choices = list(itertools.product(self.formats, repeat=2))
        exhaustive = []
        for ca, cb in itertools.product(choices, repeat=2):
            cs = [ca, cb]
            size = sum(self.elems[n] * self.bits[w] / 8
                       for n, (w, a) in zip(['A', 'B'], cs)) / 3
            bops = sum(self.flops[n] * self.bits[w] * self.bits[a]
                       for n, (w, a) in zip(['A', 'B'], cs)) / 256
            c = sum(self.cost(n, w, a) for n, (w, a) in zip(['A', 'B'], cs))
            exhaustive.append((max(size, bops), c))
        for row in answer['candidates']:
            if row['origin'] != 'milp':
                continue
            optimum = min(c for r, c in exhaustive if r <= row['search_budget'] + 1e-12)
            self.assertAlmostEqual(row['objective'], optimum)
            self.assertEqual(row['solver_status'], 0)
        self.assertFalse(answer['accuracy_guarantee'])
        self.assertEqual(answer['selected']['formats']['relu'], {'w': None, 'a': 'fp32'})
        for a, b in zip(answer['frontier'], answer['frontier'][1:]):
            self.assertLess(a['resource'], b['resource'])
            self.assertGreater(a['objective'], b['objective'])

    def test_changing_loss_units_does_not_change_choice(self):
        a = self.select()
        b = self.select(lambda n, w, f: 1e6 * self.cost(n, w, f))
        self.assertEqual(a['selected']['formats'], b['selected']['formats'])

    def test_all_zero_prefers_cheapest(self):
        a = self.select(lambda *args: 0)
        for n in ['A', 'B']:
            self.assertEqual(a['selected']['formats'][n], {'w': 'int4', 'a': 'int4'})

    def test_nonfinite_options_never_selected(self):
        def cost(n, w, a):
            if w == 'int4':
                return float('nan')
            if a == 'int4':
                return float('inf')
            return self.cost(n, w, a)
        answer = self.select(cost)
        self.assertGreater(answer['excluded_nonfinite_options'], 0)
        for row in answer['candidates']:
            for n in ['A', 'B']:
                self.assertNotIn('int4', row['formats'][n].values())

    def select_3d(self, value=None):
        return auto_select_3d(self.nodes, self.formats, self.formats, value or self.cost,
                              self.elems, self.flops, samples=7, refinements=1)

    def test_3d_pick_is_not_dominated_by_any_assignment(self):
        # Every MILP candidate is optimal within its budgets, so no assignment of the
        # whole space may be at least as good on size, BOPs and BC and better on one.
        answer = self.select_3d()
        pick = answer['selected']
        choices = list(itertools.product(self.formats, repeat=2))
        for ca, cb in itertools.product(choices, repeat=2):
            cs = list(zip(['A', 'B'], [ca, cb]))
            size = sum(self.elems[n] * self.bits[w] / 8 for n, (w, a) in cs) / 3
            bops = sum(self.flops[n] * self.bits[w] * self.bits[a] for n, (w, a) in cs) / 256
            c = sum(self.cost(n, w, a) for n, (w, a) in cs)
            p = (pick['size'], pick['bops'], pick['objective'])
            q = (size, bops, c)
            self.assertFalse(all(x <= y + 1e-12 for x, y in zip(q, p)) and
                             any(x < y - 1e-12 for x, y in zip(q, p)), (cs, q, p))
        front = [answer['candidates'][i] for i in answer['frontier_indices']]
        self.assertIn(pick, front)
        self.assertEqual(pick['selection_score'], min(r['selection_score'] for r in front))
        self.assertEqual(pick['formats']['relu'], {'w': None, 'a': 'fp32'})
        self.assertFalse(answer['accuracy_guarantee'])

    def test_3d_changing_loss_units_does_not_change_choice(self):
        a = self.select_3d()
        b = self.select_3d(lambda n, w, f: 1e6 * self.cost(n, w, f))
        self.assertEqual(a['selected']['formats'], b['selected']['formats'])

    def test_no_allowed_option_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'every option is forbidden'):
            self.select(lambda *args: None)

    def test_negative_cost_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            self.select(lambda *args: -1)

    def test_pipeline_compiler_runner_keeps_optional_env_and_cwd(self):
        from run_native_bc_pipeline import erm
        result, _ = erm.run([sys.executable, '-c', 'print("runner-ok")'])
        self.assertEqual(result.stdout.strip(), 'runner-ok')
        with tempfile.TemporaryDirectory() as folder:
            result, _ = erm.run([sys.executable, '-c',
                                 'import os; print(os.getcwd()); print(os.environ["AUTO_TEST_VALUE"])'],
                                cwd=folder, env={'AUTO_TEST_VALUE': 'present'}, timeout=10)
            self.assertEqual(result.stdout.splitlines(), [folder, 'present'])


if __name__ == '__main__':
    unittest.main()
