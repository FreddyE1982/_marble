import unittest
import types
from marble.marblemain import Brain, run_wanderer_training
from marble.reporter import REPORTER, clear_report_group

class TestWandererLockTimeout(unittest.TestCase):
    def setUp(self):
        self.reporter = REPORTER
        clear_report_group("lock_timeout")

    def test_lock_neuron_nonblocking(self):
        b = Brain(1, mode="grid")
        b.add_neuron((0,), tensor=[0.0])
        timeouts = []

        def lock_neuron(self, neuron, timeout=None):
            timeouts.append(timeout)
            class Dummy:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    return False
            return Dummy()
        b.lock_neuron = types.MethodType(lock_neuron, b)
        run_wanderer_training(b, num_walks=1, max_steps=1, lr=0.0)
        self.reporter.item["timeouts", "lock_timeout", "metrics"] = timeouts
        print("lock timeouts:", timeouts)
        self.assertTrue(all(t == 0.0 for t in timeouts))

if __name__ == "__main__":
    unittest.main(verbosity=2)
