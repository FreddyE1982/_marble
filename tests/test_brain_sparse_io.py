import os
import unittest
from pathlib import Path


class TestBrainSparseIO(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain
        self.Brain = Brain

    def test_bulk_add_and_export_import(self):
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        coords = [
            (0.0, 0.0),
            (1.5, 2.5),
            (1000.0, 2000.0),
        ]
        created = b.bulk_add_neurons(coords, tensor=[1.0, 2.0], weight=2.0, bias=0.5, age=1)
        self.assertEqual(len(created), len(coords))

        # Add a connection
        b.connect(coords[0], coords[1], direction="bi", age=3, type_name=None)

        # Export
        out = Path("sparse_brain_export.json")
        try:
            b.export_sparse(str(out))
            self.assertTrue(out.exists())
            print("sparse export path:", str(out))

            # Import back
            b2 = self.Brain.import_sparse(str(out))
            # Ensure neurons exist
            for p in coords:
                self.assertIsNotNone(b2.get_neuron(p))
            # Ensure a synapse is present
            print("sparse import neurons:", len(b2.neurons), "synapses:", len(b2.synapses))
            self.assertGreaterEqual(len(b2.synapses), 1)
        finally:
            if out.exists():
                os.remove(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
