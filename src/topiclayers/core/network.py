from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


class NetworkCreator:
    def __init__(self, df: pd.DataFrame, name: str, output_dir: Path):
        self.df = df
        self.name = name
        self.output_dir = Path(output_dir)
        self.graph_dir = self.output_dir / "networks"
        self.proj_graphs: dict = {}
        self.ml_network = None

    def create_retweet_network(self, reporter=None) -> nx.DiGraph:
        edges, missing_refs = self._aggregate_edges(self.df, reporter=reporter)

        G = nx.DiGraph()
        G.add_nodes_from(self.df["author"].unique())
        G.add_weighted_edges_from(
            (src, tgt, int(w)) for src, tgt, w in edges.itertuples(index=False, name=None)
        )

        if missing_refs:
            print(f"  {missing_refs} retweets reference tweets not in dataset — skipped.")

        self._save_graph(G, "retweet")
        return G

    @staticmethod
    def _aggregate_edges(df: pd.DataFrame, reporter=None) -> tuple[pd.DataFrame, int]:
        """Aggregate retweet rows into unique (author, target_author, weight) edges.

        Returns the aggregated edge frame and the number of retweet rows whose
        referenced tweet is not present in ``df`` (unresolvable target).
        """
        if reporter is not None:
            reporter.update(0, total=len(df))

        rt = df.loc[df["referenced_id"].notna(), ["author", "referenced_id"]]
        targets = rt["referenced_id"].map(df["author"])
        missing_refs = int(targets.isna().sum())

        edges = (
            pd.DataFrame({"author": rt["author"], "target": targets})
            .dropna(subset=["target"])
            .groupby(["author", "target"])
            .size()
            .reset_index(name="weight")
        )

        if reporter is not None:
            reporter.update(len(df), total=len(df))
        return edges, missing_refs

    def create_retweet_ml(self, reporter=None):
        import uunet.multinet as ml

        topics = self.df["topic"].unique()
        topics = topics[topics != -1]

        if len(topics) == 0:
            raise ValueError(
                "Cannot create multilayer network: all topics are -1 (outliers). "
                "This means BERTopic found no topic clusters. "
                "Try: <1> reduce min_topic_size, <2> use a larger dataset, "
                "<3> check the 'text' column is not empty."
            )

        ml_network = ml.empty()
        dropped_edges_total = 0

        rt = self.df.loc[
            self.df["referenced_id"].notna(), ["author", "referenced_id", "topic"]
        ]
        total_work = int((self.df["topic"] != -1).sum())
        done = 0

        for topic, df_tmp in self.df.groupby("topic"):
            if topic == -1:
                continue

            G = nx.DiGraph()
            G.add_nodes_from(df_tmp["author"].unique())

            rt_topic = rt[rt["topic"] == topic]
            edges, dropped = self._aggregate_slice(df_tmp, rt_topic)
            dropped_edges_total += dropped

            G.add_weighted_edges_from(
                (src, tgt, int(w)) for src, tgt, w in edges.itertuples(index=False, name=None)
            )

            done += len(df_tmp)
            if reporter is not None:
                reporter.update(done, total=total_work)

            ml.add_nx_layer(ml_network, G, str(topic))

        if dropped_edges_total:
            print(f"  {dropped_edges_total} edges dropped (referenced tweet missing in topic slice).")

        self._save_multilayer(ml_network)
        self.ml_network = ml_network
        return ml_network

    @staticmethod
    def _aggregate_slice(
        df_slice: pd.DataFrame, rt_slice: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        """Aggregate retweet rows of one topic slice into weighted edges.

        Returns (edges, dropped) where ``dropped`` counts rows whose referenced
        tweet is not present in ``df_slice``.
        """
        targets = rt_slice["referenced_id"].map(df_slice["author"])
        dropped = int(targets.isna().sum())
        edges = (
            pd.DataFrame({"author": rt_slice["author"], "target": targets})
            .dropna(subset=["target"])
            .groupby(["author", "target"])
            .size()
            .reset_index(name="weight")
        )
        return edges, dropped

    def create_ttnetwork(self, project: bool = True):
        A = self.df["author"].unique()
        M = self.df.index.tolist()
        x = self.df["text"].to_dict()
        topics = self.df["topic"].to_dict()
        author = self.df["author"].to_dict()
        is_retweet = self.df["referenced_type"].to_dict()
        is_retweet = {k: "original" if v is None else v for k, v in is_retweet.items()}

        g = nx.DiGraph()

        g.add_nodes_from(A, bipartite=0)
        g.add_nodes_from(M, bipartite=1)

        edges = list(zip(self.df["author"], self.df.index))
        ref_edges = [
            (idx, str(ref_id))
            for idx, ref_id in zip(self.df.index, self.df["referenced_id"])
            if ref_id is not None and not (isinstance(ref_id, float) and pd.isna(ref_id))
        ]

        men_edges = []
        if "mentions_name" in self.df.columns:
            exploded = self.df["mentions_name"].explode().dropna()
            exploded = exploded[exploded.map(lambda m: isinstance(m, str))]
            men_edges = list(zip(exploded.index, exploded))

        g.add_edges_from(edges, weight=10)
        g.add_edges_from(ref_edges, weight=1)

        nodes_to_remove = [node for node in g.nodes if "bipartite" not in g.nodes[node]]
        g.remove_nodes_from(nodes_to_remove)

        date_lookup = self.df["date"].to_dict()

        t = {}
        for e in g.edges():
            if e[1] in date_lookup:
                t[e] = date_lookup[e[1]]

        g.add_edges_from(men_edges)

        nodes_to_set = [node for node in g.nodes if "bipartite" not in g.nodes[node]]
        for node in nodes_to_set:
            g.nodes[node].setdefault("bipartite", 0)

        nx.set_edge_attributes(g, t, "date")
        nx.set_node_attributes(g, x, "text")
        nx.set_node_attributes(g, topics, "topics")
        nx.set_node_attributes(g, author, "author")
        nx.set_node_attributes(g, is_retweet, "is_retweet")

        for node in g.nodes():
            if "label" not in g.nodes[node]:
                g.nodes[node]["label"] = str(node)
            if "author" not in g.nodes[node] and g.nodes[node].get("bipartite") == 0:
                g.nodes[node]["author"] = str(node)

        self._save_graph(g, "ttt")

        if project:
            self._project_network(g, title=self.name)

        return g

    def _project_network(self, nx_graph=None, path=None, title="") -> dict:
        from igraph import Graph

        def recursive_explore(graph, node, start_node, previous_node=None, edges=None, topic=None, depth=0):
            neighbors = graph.neighborhood(node.index, mode="out")

            if edges is None:
                edges = {}

            if node["bipartite"] == 0.0:
                if depth == 2:
                    edges.setdefault(topic, []).append((start_node["label"], node["label"]))
                    return edges
                elif depth > 2:
                    edges.setdefault(topic, []).append((start_node["label"], previous_node["author"]))
                    return edges
            else:
                if topic is None:
                    topic = node["topics"]
                if len(neighbors) == 1:
                    edges.setdefault(topic, []).append((start_node["label"], node["author"]))
                    return edges

            for neighbor_idx in neighbors[1:]:
                new_node = graph.vs[neighbor_idx]
                recursive_explore(
                    graph, node=new_node, previous_node=node,
                    start_node=start_node, depth=depth + 1, edges=edges, topic=topic,
                )

            return edges

        if path is not None:
            g = Graph.Read_GML(path)
        elif nx_graph is not None:
            g = Graph.from_networkx(nx_graph)
            g.vs["label"] = [str(n) for n in nx_graph.nodes()]
            g.vs["_nx_name"] = [str(n) for n in nx_graph.nodes()]
        else:
            raise ValueError(
                "Provide either a path to a GML file or a networkx graph. "
                "Call create_ttnetwork(project=True) to project the bipartite TTT network."
            )

        all_edges: dict = {}

        for n in g.vs.select(bipartite_eq=0):
            result = recursive_explore(g, n, start_node=n)
            if result is None:
                continue
            for key in set(list(all_edges.keys()) + list(result.keys())):
                existing = all_edges.get(key, [])
                incoming = result.get(key, [])
                if incoming is not None:
                    all_edges[key] = list(set(existing + incoming))

        deduped = {e: list(set(all_edges[e])) for e in all_edges}
        deduped.pop(None, None)

        self._save_projected(deduped)
        return self.proj_graphs

    def _save_projected(self, edges: dict) -> None:
        prj_dir = self.graph_dir / "projected"
        prj_dir.mkdir(parents=True, exist_ok=True)

        for t, e in edges.items():
            t_str = str(t)
            if t_str.lower() == "nan":
                continue
            g = nx.from_edgelist(e, create_using=nx.DiGraph())
            self.proj_graphs[t_str] = g
            nx.write_gml(g, prj_dir / f"{self.name}__prj_{t_str}.gml")

    def _save_graph(self, G, title: str) -> Path:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        filename = self.graph_dir / f"{self.name}_{title}.gml"
        nx.write_gml(G, filename)
        return filename

    def _save_multilayer(self, ml_network) -> Path:
        import uunet.multinet as ml

        self.graph_dir.mkdir(parents=True, exist_ok=True)
        filename = self.graph_dir / f"{self.name}_retweet_network_ml.gml"
        ml.write(ml_network, file=str(filename))
        return filename
