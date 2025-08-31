import tempfile
import kuzu
import unittest
from marble.marblemain import Brain


class BrainKuzuTest(unittest.TestCase):
    def test_updates(self):
        db_path = tempfile.mktemp()
        brain = Brain(2, size=3, formula="1", kuzu_path=db_path)
        n1 = brain.add_neuron((0, 0), tensor=0.0)
        brain.add_neuron((0, 1), tensor=0.0)
        syn = brain.connect((0, 0), (0, 1))
        conn = kuzu.Connection(kuzu.Database(db_path))
        nodes = list(conn.execute("MATCH (n:Neuron) RETURN COUNT(*)"))[0][0]
        edges = list(conn.execute("MATCH ()-[:Synapse]->() RETURN COUNT(*)"))[0][0]
        print("initial", nodes, edges)
        self.assertEqual(nodes, 2)
        self.assertEqual(edges, 1)
        brain.remove_synapse(syn)
        conn = kuzu.Connection(kuzu.Database(db_path))
        edges_after = list(conn.execute("MATCH ()-[:Synapse]->() RETURN COUNT(*)"))[0][0]
        print("after_remove_syn", edges_after)
        self.assertEqual(edges_after, 0)
        del conn
        brain.remove_neuron(n1)
        conn2 = kuzu.Connection(kuzu.Database(db_path))
        nodes_after = list(conn2.execute("MATCH (n:Neuron) RETURN COUNT(*)"))[0][0]
        print("after_remove_neuron", nodes_after)
        self.assertEqual(nodes_after, 1)


if __name__ == "__main__":
    unittest.main()
