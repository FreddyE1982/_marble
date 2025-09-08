import os
import subprocess
import sys
import unittest


class TestDecisionControllerDeterministic(unittest.TestCase):
    def test_constraint_determinism(self):
        code = (
            "import json, os\n"
            "import marble.decision_controller as dc\n"
            "dc.LAST_STATE_CHANGE.clear()\n"
            "dc.TAU_THRESHOLD = 0.0\n"
            "dc.BUDGET_LIMIT = 5.0\n"
            "dc.LINEAR_CONSTRAINTS_A = [[1, 1]]\n"
            "dc.LINEAR_CONSTRAINTS_B = [1]\n"
            "h_t = {'B': {'cost': 1}, 'A': {'cost': 1}}\n"
            "x_t = {'B': 'on', 'A': 'on'}\n"
            "res = dc.decide_actions(h_t, x_t, [], all_plugins=set(h_t.keys()))\n"
            "print(json.dumps(res, sort_keys=True))\n"
        )
        outputs = []
        for seed in ['1', '2', '3']:
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = seed
            result = subprocess.run(
                [sys.executable, '-c', code], capture_output=True, text=True, env=env
            )
            outputs.append(result.stdout.strip())
        print('subprocess outputs:', outputs)
        self.assertTrue(all(o == outputs[0] for o in outputs))


if __name__ == '__main__':  # pragma: no cover
    unittest.main(verbosity=2)
