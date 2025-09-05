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
        self.assertAlmostEqual(v, 4.5, places=6)

    def test_baseline_stable_with_identical_policies(self):
        traj = Trajectory()
        # two steps, identical policy probabilities and zero rewards
        traj.log(0, 0.0, 0.5, 0.5)
        traj.log(1, 0.0, 0.5, 0.5)
        baseline = [5.0, 3.0, 1.0]
        v = doubly_robust(traj, baseline)
        print("baseline:", baseline[0], "V_hat:", v)
        self.assertAlmostEqual(v, baseline[0], places=6)


if __name__ == "__main__":
    unittest.main()
