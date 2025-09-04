import unittest
from marble import Trajectory, importance_weights, doubly_robust


class TestOffPolicy(unittest.TestCase):
    def test_doubly_robust(self):
        traj = Trajectory()
        traj.log(0, 1.0, 0.5, 0.75)
        traj.log(1, 2.0, 0.5, 0.25)
        weights = importance_weights(traj.logged_probs, traj.new_probs)
        print("weights:", weights)
        v = doubly_robust(traj, [1.5, 0.5, 0.0])
        print("V_hat:", v)
        self.assertAlmostEqual(v, 2.375, places=6)


if __name__ == "__main__":
    unittest.main()
