import marble.plugin_graph as pg


def test_recommendation_order():
    pg.PLUGIN_GRAPH.reset()
    g = pg.PLUGIN_GRAPH
    g.add_plugin("A")
    g.add_plugin("B")
    g.add_plugin("C")
    g.add_dependency("A", "B")
    g.add_dependency("B", "C")
    assert set(pg.recommend_next_plugin()) == {"A"}
    assert set(pg.recommend_next_plugin("A")) == {"B"}
    assert set(pg.recommend_next_plugin("B")) == {"C"}
    assert pg.recommend_next_plugin("C") == []
