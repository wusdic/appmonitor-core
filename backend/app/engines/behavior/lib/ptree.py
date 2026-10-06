"""Pattern tree of the progressive core (docs/lib3/progressive.md §5.5.1, §6.6, §6.7, §6.8).

STATUS: implemented (W-P0). Structure only: routing, split / collapse /
exception / retire operations with lineage and versions, budget counting.
The learning rules (when to split, prune, revise) are P04's.

model.ptree@(tree key, '__system__') = PTreeModel {'fmt': 1, 'version', 'kinds': {kind: Tree}}

Tree
    root, nodes {nid: Node}, next_id (ids never reused), lineage Ring(4096) of
    (ts, op, nid, parents, children, detail), retired {nid: summary} (FIFO,
    <= N_max / 2), dormant {nid: record} (<= 10 % of N_max), budget {...}
Split(attr, level, groups, children, other)
    child i holds the values of groups[i] at (attr, level); `other` receives
    every value not in a group, so unseen values always land in a general
    pattern. ABSENT (⊥) is grouped like any value.
route(tree, get, hier) -> [nid_0 ... nid_leaf]
    get(attr) returns the event's value (ABSENT when absent); hier is a
    phier.Hierarchies. A split on an attribute declared `gone` routes by its
    heaviest child until P04 collapses it (pass gone={attr, ...}).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .pevent import ABSENT
from .pnode import Node

D_MAX = 8
LINEAGE_MAX = 4096
TIERS = {"XS": 32, "S": 256, "M": 1024, "L": 4096}
OPS = ("create", "split", "merge", "prune", "replace", "exc_add", "exc_del", "content", "retire",
       "dormant", "revive", "collapse")


@dataclass
class Split:
    attr: str
    level: int
    groups: List[FrozenSet[Any]]
    children: List[int]
    other: int
    index: Dict[Any, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.reindex()

    def reindex(self) -> None:
        self.index = {}
        for g, c in zip(self.groups, self.children):
            for v in g:
                self.index[v] = c

    def child_for(self, g: Any) -> int:
        try:
            return self.index.get(g, self.other)
        except TypeError:                       # unhashable value
            return self.other

    def all_children(self) -> List[int]:
        return list(self.children) + [self.other]


class Tree:
    """One pattern tree (per tree key and event kind)."""

    def __init__(self, kind: int, t: float, n_max: int = TIERS["M"]) -> None:
        self.kind = int(kind)
        self.nodes: Dict[int, Node] = {}
        self.next_id = 0
        self.lineage: Deque[Tuple[float, str, int, Tuple[int, ...], Tuple[int, ...], Any]] = \
            deque(maxlen=LINEAGE_MAX)
        self.retired: "Dict[int, Dict[str, Any]]" = {}
        self.dormant: "Dict[int, Dict[str, Any]]" = {}
        self.budget: Dict[str, Any] = {"n_max": int(n_max), "tier": "M"}
        self.version = 1
        self.root = self._new(None, 0, (), t)
        self._log(t, "create", self.root, (), (), "root")

    # ------------------------------------------------------------ basics
    def _new(self, parent: Optional[int], depth: int, ctx: Tuple, t: float,
             is_exc: bool = False) -> int:
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = Node(nid, parent, depth, self.kind, ctx, t, is_exc)
        return nid

    def _log(self, t: float, op: str, nid: int, parents: Sequence[int], children: Sequence[int],
             detail: Any = None) -> None:
        self.lineage.append((float(t), op, int(nid), tuple(parents), tuple(children), detail))

    def node(self, nid: int) -> Node:
        return self.nodes[nid]

    def __len__(self) -> int:
        return len(self.nodes)

    def live_count(self) -> int:
        return len(self.nodes)

    def leaves(self) -> List[int]:
        return [n for n, nd in self.nodes.items() if nd.split is None and not nd.is_exc]

    def children(self, nid: int) -> List[int]:
        nd = self.nodes[nid]
        out = nd.split.all_children() if nd.split is not None else []
        return out + list(nd.exc.values())

    def path_to(self, nid: int) -> List[int]:
        out = [nid]
        while self.nodes[out[-1]].parent is not None:
            out.append(self.nodes[out[-1]].parent)
        return out[::-1]

    def exceptions_count(self) -> int:
        return sum(1 for nd in self.nodes.values() if nd.is_exc)

    # ----------------------------------------------------------- routing
    def route(self, get: Callable[[str], Any], hier: Any,
              gone: Optional[Iterable[str]] = None, mass_t: Optional[float] = None) -> List[int]:
        """Path of node ids from the root to the event's leaf (§5.5.1).
        Exceptions are not entered here (P03/P04 check node.exc[ip])."""
        gone_set = set(gone or ())
        nid = self.root
        path = [nid]
        nd = self.nodes[nid]
        while nd.split is not None and len(path) <= D_MAX:
            sp = nd.split
            if sp.attr in gone_set:
                kids = sp.all_children()
                t = mass_t if mass_t is not None else (nd.last_seen or 0.0)
                nid = max(kids, key=lambda c: self.nodes[c].mass_at(t))
            else:
                g = hier.gen(sp.attr, sp.level, get(sp.attr))
                nid = sp.child_for(g)
            path.append(nid)
            nd = self.nodes[nid]
        return path

    # -------------------------------------------------------- operations
    def split(self, nid: int, attr: str, level: int, groups: Sequence[Iterable[Any]], t: float,
              detail: Any = None) -> Split:
        """Specialise leaf nid on (attr, level) into one child per value group
        plus the `other` child. Children start as candidates (§6.5.5)."""
        nd = self.nodes[nid]
        if nd.split is not None:
            raise ValueError(f"node {nid} is already split")
        if nd.depth + 1 > D_MAX:
            raise ValueError("depth limit D_max reached")
        gs = [frozenset(g) for g in groups if len(frozenset(g))]
        union = frozenset().union(*gs) if gs else frozenset()
        kids = []
        for g in gs:
            kids.append(self._new(nid, nd.depth + 1, nd.ctx + ((attr, int(level), g, False),), t))
        other = self._new(nid, nd.depth + 1, nd.ctx + ((attr, int(level), union, True),), t)
        nd.split = Split(attr, int(level), gs, kids, other)
        nd.version += 1
        self.version += 1
        self._log(t, "split", nid, (nid,), tuple(kids) + (other,),
                  detail if detail is not None else {"attr": attr, "level": int(level)})
        return nd.split

    def collapse(self, nid: int, t: float, op: str = "prune", reason: Any = None) -> List[int]:
        """Collapse nid's split: every descendant's summaries are merged into
        nid (all summaries are mergeable), descendants retire with lineage
        `merged_into`; nid becomes a leaf. Returns the retired ids."""
        nd = self.nodes[nid]
        if nd.split is None:
            return []
        gone: List[int] = []
        stack = list(nd.split.all_children())
        while stack:
            c = stack.pop()
            cn = self.nodes.get(c)
            if cn is None:
                continue
            stack.extend(self.children(c))
            gone.append(c)
        # Every event updates the core summaries (mass, evidence, who, when) of
        # every node on its path and the targets of its ancestors (thinned,
        # §6.5.1), so nid already holds the subtree's core statistics; only
        # targets / pairs nid does not track are rebuilt from its direct
        # children, which partition nid's events since the split.
        direct = [self.nodes[c] for c in nd.split.all_children() if c in self.nodes]
        # in the children's order (a set of names iterated in the salted hash
        # order made the node's target dict order, and so which targets a later
        # trim drops, depend on PYTHONHASHSEED)
        missing = [a for a in dict.fromkeys(a for cn in direct for a in cn.targets) if a not in nd.targets]
        for a in missing:
            parts = [cn.targets[a] for cn in direct if a in cn.targets]
            base = parts[0]
            for other in parts[1:]:
                if type(other) is type(base):
                    base.merge(other)
            nd.targets[a] = base
        for key in [k for k in dict.fromkeys(k for cn in direct for k in cn.pairs) if k not in nd.pairs]:
            parts = [cn.pairs[key] for cn in direct if key in cn.pairs]
            base = parts[0]
            for other in parts[1:]:
                base.merge(other)
            nd.pairs[key] = base
        for c in gone:                   # nid is a leaf again: its sources' extremes come back
            nd.absorb_extremes(self.nodes[c])
        nd.split = None
        nd.version += 1
        for c in gone:
            self._retire_node(c, t, {"merged_into": nid, "reason": reason})
        self.version += 1
        self._log(t, op, nid, (nid,), tuple(gone), reason)
        return gone

    def merge_siblings(self, parent: int, a: int, b: int, t: float) -> int:
        """Unite two children of parent's split (their value groups and
        summaries); b retires into a. `other` may be either side (the merged
        group then becomes `other`)."""
        pn = self.nodes[parent]
        sp = pn.split
        if sp is None:
            raise ValueError("parent has no split")
        if b == sp.other:
            a, b = b, a
        if self.nodes[b].split is not None or self.nodes[a].split is not None:
            raise ValueError("only leaf siblings merge")
        ia = sp.children.index(a) if a != sp.other else None
        ib = sp.children.index(b)
        gb = sp.groups[ib]
        if ia is not None:
            sp.groups[ia] = sp.groups[ia] | gb
        del sp.groups[ib]
        del sp.children[ib]
        sp.reindex()
        an = self.nodes[a]
        an.absorb(self.nodes[b])
        attr, level, vals, neg = an.ctx[-1]
        if ia is not None:
            an.ctx = an.ctx[:-1] + ((attr, level, vals | gb, False),)
        else:
            union = frozenset().union(*sp.groups) if sp.groups else frozenset()
            an.ctx = an.ctx[:-1] + ((attr, level, union, True),)
        an.version += 1
        self._retire_node(b, t, {"merged_into": a})
        self.version += 1
        self._log(t, "merge", a, (a, b), (a,), None)
        return a

    def add_exception(self, nid: int, ip: str, t: float) -> int:
        nd = self.nodes[nid]
        if ip in nd.exc:
            return nd.exc[ip]
        c = self._new(nid, nd.depth + 1,
                      nd.ctx + (("net.src", 0, frozenset({ip}), False),), t, is_exc=True)
        nd.exc[ip] = c
        nd.version += 1
        self.version += 1
        self._log(t, "exc_add", c, (nid,), (c,), ip)
        return c

    def remove_exception(self, nid: int, ip: str, t: float) -> Optional[int]:
        nd = self.nodes[nid]
        c = nd.exc.pop(ip, None)
        if c is None:
            return None
        self._retire_node(c, t, {"exc_removed": ip})
        nd.version += 1
        self.version += 1
        self._log(t, "exc_del", c, (nid,), (), ip)
        return c

    def replace_subtree(self, old: int, new_root_split: Split, t: float) -> None:
        """Record an EFDT revision / HAT alternate replacement: lineage only
        (P04 builds the new children with split() on a fresh node first)."""
        self._log(t, "replace", old, (old,), tuple(new_root_split.all_children()), None)

    def _retire_node(self, nid: int, t: float, detail: Dict[str, Any]) -> None:
        nd = self.nodes.pop(nid, None)
        if nd is None:
            return
        nd.state = "retired"
        for ip, c in list(nd.exc.items()):
            self._retire_node(c, t, {"parent_retired": nid})
        self.retired[nid] = {"ctx": nd.ctx, "retired": float(t), "first_seen": nd.first_seen,
                             "last_seen": nd.last_seen, "days_bits": nd.days_bits,
                             "version": nd.version, "cver": nd.cver, "detail": detail,
                             "ref": nd.ref}
        cap = max(1, self.budget.get("n_max", TIERS["M"]) // 2)
        while len(self.retired) > cap:
            self.retired.pop(next(iter(self.retired)))
        self._log(t, "retire", nid, (), (), detail)

    def retire_leaf(self, nid: int, t: float, reason: Any = None) -> None:
        """Retire a leaf (budget prune / stale): it is removed from its
        parent's split; its values then route to `other`. The `other` child
        itself cannot be retired this way (collapse the parent instead)."""
        nd = self.nodes[nid]
        if nd.split is not None:
            raise ValueError("retire_leaf on an internal node")
        par = self.nodes.get(nd.parent) if nd.parent is not None else None
        if par is not None and par.split is not None:
            sp = par.split
            if nid == sp.other:
                raise ValueError("cannot retire the other child; collapse the parent")
            i = sp.children.index(nid)
            del sp.children[i]
            del sp.groups[i]
            sp.reindex()
            par.version += 1
        elif par is not None and nd.is_exc:
            for ip, c in list(par.exc.items()):
                if c == nid:
                    del par.exc[ip]
        self._retire_node(nid, t, {"reason": reason})
        self.version += 1

    def make_dormant(self, nid: int, t: float) -> None:
        """Keep only context, reference snapshot and date bitmap (§6.8.1)."""
        rec = self.retired.get(nid)
        if rec is None:
            return
        self.dormant[nid] = {"ctx": rec["ctx"], "ref": rec.get("ref"), "days_bits": rec["days_bits"],
                             "since": float(t)}
        cap = max(1, self.budget.get("n_max", TIERS["M"]) // 10)
        while len(self.dormant) > cap:
            self.dormant.pop(next(iter(self.dormant)))
        self._log(t, "dormant", nid, (), (), None)

    # ------------------------------------------------------------ budget
    def over_budget(self) -> int:
        return max(0, len(self.nodes) - int(self.budget.get("n_max", TIERS["M"])))

    def set_tier(self, tier: str) -> None:
        self.budget["tier"] = tier
        self.budget["n_max"] = TIERS[tier]

    def nbytes(self) -> int:
        return int(sum(n.nbytes() for n in self.nodes.values()) + 200 * len(self.lineage)
                   + 300 * len(self.retired) + 1000 * len(self.dormant) + 1000)


class PTreeModel:
    """model.ptree: {'fmt': 1, 'version', 'kinds': {kind: Tree}}."""

    def __init__(self, tree_key: str) -> None:
        self.fmt = 1
        self.tree_key = tree_key
        self.version = 1
        self.kinds: Dict[int, Tree] = {}

    def tree(self, kind: int, t: float, create: bool = True, n_max: int = TIERS["M"]) -> Optional[Tree]:
        tr = self.kinds.get(int(kind))
        if tr is None and create:
            tr = self.kinds[int(kind)] = Tree(int(kind), t, n_max)
        return tr

    def nbytes(self) -> int:
        return int(sum(t.nbytes() for t in self.kinds.values()) + 200)
