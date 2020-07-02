import itertools
import os
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from typing import (
    Set, Iterator, List, Optional, Dict, Union, Any, Type, Sequence
)
from typing_extensions import Protocol

from .graph import Graph, UniqueId
from .queue import GraphQueue
from .selector_methods import (
    MethodName,
    SelectorMethod,
    QualifiedNameSelectorMethod,
    TagSelectorMethod,
    SourceSelectorMethod,
    PathSelectorMethod,
)
from dbt.logger import GLOBAL_LOGGER as logger
from dbt.node_types import NodeType
from dbt.exceptions import RuntimeException, InternalException, warn_or_error
from dbt.contracts.graph.compiled import NonSourceNode, CompileResultNode
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.parsed import ParsedSourceDefinition


RAW_SELECTOR_PATTERN = re.compile(
    r'\A'
    r'(?P<childs_parents>(\@))?'
    r'(?P<parents>((?P<parents_depth>(\d*))\+))?'
    r'((?P<method>(\w+)):)?(?P<value>(.*?))'
    r'(?P<children>(\+(?P<children_depth>(\d*))))?'
    r'\Z'
)

INTERSECTION_DELIMITER = ','


def _probably_path(value: str):
    """Decide if value is probably a path. Windows has two path separators, so
    we should check both sep ('\\') and altsep ('/') there.
    """
    if os.path.sep in value:
        return True
    elif os.path.altsep is not None and os.path.altsep in value:
        return True
    else:
        return False


def _match_to_int(match: Dict[str, str], key: str) -> Optional[int]:
    raw = match.get(key)
    # turn the empty string into None, too.
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeException(
            f'Invalid node spec - could not handle parent depth {raw}'
        ) from exc


SelectionSpecComponent = Union[
    'SelectionCriteria',
    'SelectionIntersection',
    'SelectionDifference',
    'SelectionUnion',
]


@dataclass
class SelectionCriteria:
    raw: str
    method: MethodName
    value: str
    select_childrens_parents: bool
    select_parents: bool
    select_parents_max_depth: Optional[int]
    select_children: bool
    select_children_max_depth: Optional[int]

    def __post_init__(self):
        if self.select_children and self.select_childrens_parents:
            raise RuntimeException(
                f'Invalid node spec {self.raw} - "@" prefix and "+" suffix '
                'are incompatible'
            )

    @classmethod
    def default_method(cls, value: str) -> MethodName:
        if _probably_path(value):
            return MethodName.Path
        else:
            return MethodName.FQN

    @classmethod
    def parse_method(cls, raw: str, groupdict: Dict[str, Any]) -> MethodName:
        raw_method = groupdict.get('method')
        if raw_method is None:
            return cls.default_method(groupdict['value'])

        try:
            return MethodName(raw_method)
        except ValueError:
            raise RuntimeException(
                f'unknown selector filter "{raw_method}" in "{raw}"'
            ) from None

    @classmethod
    def from_single_spec(cls, raw: str) -> 'SelectionCriteria':
        result = RAW_SELECTOR_PATTERN.match(raw)
        if result is None:
            # bad spec!
            raise RuntimeException(f'Invalid selector spec "{raw}"')
        result_dict = result.groupdict()

        if 'value' not in result_dict:
            raise RuntimeException(
                f'Invalid node spec "{raw}" - no search value!'
            )

        method = cls.parse_method(raw, result_dict)

        parents_max_depth = _match_to_int(result_dict, 'parents_depth')
        children_max_depth = _match_to_int(result_dict, 'children_depth')

        return cls(
            raw=raw,
            method=method,
            value=result_dict['value'],
            select_childrens_parents=bool(result_dict.get('childs_parents')),
            select_parents=bool(result_dict.get('parents')),
            select_parents_max_depth=parents_max_depth,
            select_children=bool(result_dict.get('children')),
            select_children_max_depth=children_max_depth,
        )

    def collect_models(
        self, graph: Graph, selected: Set[UniqueId]
    ) -> Set[UniqueId]:
        additional: Set[UniqueId] = set()
        if self.select_childrens_parents:
            additional.update(graph.select_childrens_parents(selected))
        if self.select_parents:
            additional.update(
                graph.select_parents(selected, self.select_parents_max_depth)
            )
        if self.select_children:
            additional.update(
                graph.select_children(selected, self.select_children_max_depth)
            )
        return additional


class SelectorProtocol(Protocol):
    def get_nodes_from_spec(
        self,
        graph: Graph,
        spec: SelectionCriteria,
    ) -> Set[str]:
        ...

    def get_selected(self) -> Set[str]:
        ...


class BaseSelectionGroup(metaclass=ABCMeta):
    def __init__(
        self,
        components: Sequence[SelectionSpecComponent],
        expect_exists: bool = False,
        raw: Any = None,
    ):
        self.components: List[SelectionSpecComponent] = list(components)
        self.expect_exists = expect_exists
        self.raw = raw

    def __iter__(self) -> Iterator[SelectionSpecComponent]:
        for component in self.components:
            yield component

    @abstractmethod
    def combine_selections(self, selections: List[Set[str]]) -> Set[str]:
        raise NotImplementedError(
            '_combine_selections not implemented!'
        )


class SelectionIntersection(BaseSelectionGroup):
    def combine_selections(self, selections: List[Set[str]]) -> Set[str]:
        return set.intersection(*selections)


class SelectionDifference(BaseSelectionGroup):
    def combine_selections(self, selections: List[Set[str]]) -> Set[str]:
        return set.difference(*selections)


class SelectionUnion(BaseSelectionGroup):
    def combine_selections(self, selections: List[Set[str]]) -> Set[str]:
        return set.union(*selections)


def _parse_union_from_default(
    raw: Optional[List[str]], default: List[str]
) -> SelectionUnion:
    raw_components: List[str]
    if raw is None:
        expect_exists = False
        raw_components = default
    else:
        expect_exists = True
        raw_components = raw

    # turn ['a b', 'c'] -> ['a', 'b', 'c']
    raw_specs = itertools.chain.from_iterable(
        r.split(' ') for r in raw_components
    )
    union_components: List[SelectionSpecComponent] = []

    # ['a', 'b', 'c,d'] -> union('a', 'b', intersection('c', 'd'))
    for raw_spec in raw_specs:
        intersection_components: List[SelectionSpecComponent] = [
            SelectionCriteria.from_single_spec(part)
            for part in raw_spec.split(INTERSECTION_DELIMITER)
        ]
        union_components.append(SelectionIntersection(
            components=intersection_components,
            expect_exists=expect_exists,
            raw=raw_spec,
        ))

    return SelectionUnion(
        components=union_components,
        expect_exists=expect_exists,
        raw=raw_components,
    )


def get_package_names(nodes):
    return set([node.split(".")[1] for node in nodes])


def alert_non_existence(raw_spec, nodes):
    if len(nodes) == 0:
        warn_or_error(
            f"The selector '{str(raw_spec)}' does not match any nodes and will"
            f" be ignored"
        )


class InvalidSelectorError(Exception):
    pass


DEFAULT_INCLUDES: List[str] = ['fqn:*', 'source:*']
DEFAULT_EXCLUDES: List[str] = []


class NodeSelector:
    """The node selector is aware of the graph and manifest,
    """
    SELECTOR_METHODS: Dict[str, Type[SelectorMethod]] = {}

    def __init__(
        self,
        graph: Graph,
        manifest: Manifest,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.full_graph = graph
        self.manifest = manifest
        self.include = include
        self.exclude = exclude

    @classmethod
    def register_method(cls, name, selector: Type[SelectorMethod]):
        cls.SELECTOR_METHODS[name] = selector

    def get_selector(self, method: str) -> SelectorMethod:
        if method in self.SELECTOR_METHODS:
            cls: Type[SelectorMethod] = self.SELECTOR_METHODS[method]
            return cls(self.manifest)
        else:
            raise InvalidSelectorError(method)

    def select_included(
        self, included_nodes: Set[str], spec: SelectionCriteria,
    ) -> Set[str]:
        selector = self.get_selector(spec.method)
        return set(selector.search(included_nodes, spec.value))

    def get_nodes_from_criteria(
        self, graph: Graph, spec: SelectionCriteria
    ) -> Set[str]:
        nodes = graph.nodes()
        try:
            collected = self.select_included(nodes, spec)
        except InvalidSelectorError:
            valid_selectors = ", ".join(self.SELECTOR_METHODS)
            logger.info(
                f"The '{spec.method}' selector specified in {spec.raw} is "
                f"invalid. Must be one of [{valid_selectors}]"
            )
            return set()

        specified = spec.collect_models(graph, collected)
        collected.update(specified)
        result = self.expand_selection(graph, collected)
        return result

    def get_nodes_from_selection_spec(
        self, graph: Graph, spec: SelectionSpecComponent
    ) -> Set[str]:
        if isinstance(spec, SelectionCriteria):
            result = self.get_nodes_from_criteria(graph, spec)
        else:
            node_selections = [
                self.get_nodes_from_selection_spec(graph, component)
                for component in spec
            ]
            if node_selections:
                result = spec.combine_selections(node_selections)
            else:
                result = set()
            if spec.expect_exists:
                alert_non_existence(spec.raw, result)
        return result

    def select_nodes(self, graph: Graph) -> Set[str]:
        included = _parse_union_from_default(self.include, DEFAULT_INCLUDES)
        excluded = _parse_union_from_default(self.exclude, DEFAULT_EXCLUDES)

        full_selection = SelectionDifference(components=[included, excluded])
        return self.get_nodes_from_selection_spec(graph, full_selection)

    def _is_graph_member(self, unique_id: str) -> bool:
        if unique_id in self.manifest.sources:
            source = self.manifest.sources[unique_id]
            return source.config.enabled
        node = self.manifest.nodes[unique_id]
        return not node.empty and node.config.enabled

    def node_is_match(
        self,
        node: Union[ParsedSourceDefinition, NonSourceNode],
    ) -> bool:
        return True

    def _is_match(self, unique_id: str) -> bool:
        node: CompileResultNode
        if unique_id in self.manifest.nodes:
            node = self.manifest.nodes[unique_id]
        elif unique_id in self.manifest.sources:
            node = self.manifest.sources[unique_id]
        else:
            raise InternalException(
                f'Node {unique_id} not found in the manifest!'
            )
        return self.node_is_match(node)

    def build_graph_member_subgraph(self) -> Graph:
        graph_members = {
            unique_id for unique_id in self.full_graph.nodes()
            if self._is_graph_member(unique_id)
        }
        return self.full_graph.subgraph(graph_members)

    def filter_selection(self, selected: Set[str]) -> Set[str]:
        return {
            unique_id for unique_id in selected if self._is_match(unique_id)
        }

    def expand_selection(
        self, filtered_graph: Graph, selected: Set[str]
    ) -> Set[str]:
        return selected

    def get_selected(self) -> Set[str]:
        """get_selected runs trhough the node selection process:

            - build a subgraph containing only non-empty, enabled nodes and
                enabled sources
            - node selection occurs. Based on the include/exclude sets, the set
                of matched unique IDs is returned
            - the set of selected nodes is expanded (implemented per-task)
                TODO: this should return a new graph, not a new set of selected
                nodes!
                this is where tests are added
            - filtering (implemented per-task)
                - on tests: only return data tests/schema tests as selected
                - node type filtering
        """
        filtered_graph = self.build_graph_member_subgraph()
        selected_nodes = self.select_nodes(filtered_graph)
        # expanded_nodes = self.expand_selection(filtered_graph, selected_nodes)
        filtered_nodes = self.filter_selection(selected_nodes)
        return filtered_nodes

    def get_graph_queue(self) -> GraphQueue:
        """Returns a queue over nodes in the graph that tracks progress of
        dependecies.
        """
        selected_nodes = self.get_selected()
        new_graph = self.full_graph.get_subset_graph(selected_nodes)
        # should we give a way here for consumers to mutate the graph?
        return GraphQueue(new_graph.graph, self.manifest, selected_nodes)


class ResourceTypeSelector(NodeSelector):
    def __init__(
        self,
        graph: Graph,
        manifest: Manifest,
        resource_types: List[NodeType],
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        super().__init__(
            graph=graph,
            manifest=manifest,
            include=include,
            exclude=exclude,
        )
        self.resource_types: Set[NodeType] = set(resource_types)

    def node_is_match(self, node):
        return node.resource_type in self.resource_types


NodeSelector.register_method(MethodName.FQN, QualifiedNameSelectorMethod)
NodeSelector.register_method(MethodName.Tag, TagSelectorMethod)
NodeSelector.register_method(MethodName.Source, SourceSelectorMethod)
NodeSelector.register_method(MethodName.Path, PathSelectorMethod)
