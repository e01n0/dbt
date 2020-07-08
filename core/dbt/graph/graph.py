from typing import (
    Set, Iterable, Iterator, Optional
)
import networkx as nx  # type: ignore

from dbt.contracts.graph.manifest import Manifest
from dbt.exceptions import RuntimeException, InternalException
from dbt.node_types import NodeType

# it would be nice to use a NewType for this, but that will cause problems with
# str interop, which dbt relies on implicilty all over.
UniqueId = str


class Graph:
    """A wrapper around the networkx graph that understands SelectionCriteria
    and how they interact with the graph.
    """
    def __init__(self, graph):
        self.graph = graph

    def nodes(self) -> Set[UniqueId]:
        return set(self.graph.nodes())

    def edges(self):
        return self.graph.edges()

    def __iter__(self) -> Iterator[UniqueId]:
        return iter(self.graph.nodes())

    def ancestors(self, node, max_depth: Optional[int] = None) -> Set[str]:
        """Returns all nodes having a path to `node` in `graph`"""
        if not self.graph.has_node(node):
            raise InternalException(f'Node {node} not found in the graph!')
        with nx.utils.reversed(self.graph):
            anc = nx.single_source_shortest_path_length(G=self.graph,
                                                        source=node,
                                                        cutoff=max_depth)\
                .keys()
        return anc - {node}

    def descendants(self, node, max_depth: Optional[int] = None) -> Set[str]:
        """Returns all nodes reachable from `node` in `graph`"""
        if not self.graph.has_node(node):
            raise InternalException(f'Node {node} not found in the graph!')
        des = nx.single_source_shortest_path_length(G=self.graph,
                                                    source=node,
                                                    cutoff=max_depth)\
            .keys()
        return des - {node}

    def select_childrens_parents(
        self, selected: Set[UniqueId]
    ) -> Set[UniqueId]:
        ancestors_for = self.select_children(selected) | selected
        return self.select_parents(ancestors_for) | ancestors_for

    def select_children(
        self, selected: Set[UniqueId], max_depth: Optional[int] = None
    ) -> Set[UniqueId]:
        descendants: Set[UniqueId] = set()
        for node in selected:
            descendants.update(self.descendants(node, max_depth))
        return descendants

    def select_parents(
        self, selected: Set[UniqueId], max_depth: Optional[int] = None
    ) -> Set[UniqueId]:
        ancestors: Set[UniqueId] = set()
        for node in selected:
            ancestors.update(self.ancestors(node, max_depth))
        return ancestors

    def select_successors(self, selected: Set[UniqueId]) -> Set[UniqueId]:
        successors: Set[UniqueId] = set()
        for node in selected:
            successors.update(self.graph.successors(node))
        return successors

    def get_subset_graph(self, selected: Iterable[UniqueId]) -> 'Graph':
        """Create and return a new graph that is a shallow copy of the graph,
        but with only the nodes in include_nodes. Transitive edges across
        removed nodes are preserved as explicit new edges.
        """
        new_graph = nx.algorithms.transitive_closure(self.graph)

        include_nodes = set(selected)

        for node in self:
            if node not in include_nodes:
                new_graph.remove_node(node)

        for node in include_nodes:
            if node not in new_graph:
                raise RuntimeException(
                    "Couldn't find model '{}' -- does it exist or is "
                    "it disabled?".format(node)
                )
        return Graph(new_graph)

    def subgraph(self, nodes: Iterable[UniqueId]) -> 'Graph':
        return Graph(self.graph.subgraph(nodes))

    def sorted_ephemeral_ancestors(
        self, manifest: Manifest, unique_id: str
    ) -> Iterable[str]:
        """Get the ephemeral ancestors of unique_id, stopping at the first
        non-ephemeral node in each chain, in graph-topological order.
        """
        to_check: Set[str] = {unique_id}
        ephemerals: Set[str] = set()
        visited: Set[str] = set()

        while to_check:
            # note that this avoids collecting unique_id itself
            nextval = to_check.pop()
            for pred in self.graph.predecessors(nextval):
                if pred in visited:
                    continue
                visited.add(pred)
                node = manifest.expect(pred)

                if node.resource_type != NodeType.Model:
                    continue
                if node.get_materialization() != 'ephemeral':  # type: ignore
                    continue
                # this is an ephemeral model! We have to find everything it
                # refs and do it all over again until we exhaust them all
                ephemerals.add(pred)
                to_check.add(pred)

        ephemeral_graph = self.get_subset_graph(ephemerals)
        # we can just topo sort this because we know there are no cycles.
        return nx.topological_sort(ephemeral_graph.graph)

    def get_dependent_nodes(self, node: UniqueId):
        return nx.descendants(self.graph, node)
