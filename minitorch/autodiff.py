from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    args_list1 = list(vals)
    args_list1[arg] += epsilon
    args_list2 = list(vals)
    args_list2[arg] -= epsilon
    return (f(*args_list1)-f(*args_list2))/(2*epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # reverse post-order of dfs
    def dfs(cur: Variable):
        visited.add(cur.unique_id)
        if cur.is_leaf():
            res.append(cur)
            return
        
        for next in cur.parents:
            if next.is_constant() or next.unique_id in visited:
                continue
            dfs(next)
        res.append(cur)
    
    res = []
    visited = set()
    dfs(variable)
    return res[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    from collections import defaultdict, deque
    deriv_dict = defaultdict(int)
    deriv_dict[variable.unique_id] = deriv
    t_sorted = topological_sort(variable)
    queue = deque(t_sorted)
    while queue:
        poped = queue.popleft()
        if poped.is_leaf():
            continue
        d_out = deriv_dict[poped.unique_id]
        backs = poped.chain_rule(d_out)
        for (va,deriv) in backs:
            if va.is_leaf():
                va.accumulate_derivative(deriv)
            else:
                deriv_dict[va.unique_id] += deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
