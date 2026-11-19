# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module provides the deterministic state machine used to constrain
sequence-level (trajectory) predictions.

The state machine operates on **business labels**, i.e. the labels exposed to
users, and not on the integer class columns produced by a predictive model.
The mapping between class columns and business labels is handled by
:mod:`deel.puncc.api.sequence`.
"""

from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import Hashable
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Tuple


_MISSING = object()


def _require_hashable(value: Any, description: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"{description} must be hashable, received {value!r}."
        ) from exc


class StateMachine:
    """Deterministic finite state machine over business labels.

    A state machine defines which label sequences are legal.  It is entirely
    deterministic: for a given state and label, at most one next state exists.
    Transition tables that map the same ``(state, label)`` pair to different
    targets are rejected.

    The machine does not represent complete sequences explicitly.  It is used
    as the constraint component when building the shared candidate graph of
    :class:`deel.puncc.api.sequence.SequenceCandidateGraph`.

    :param transitions: transition description. Accepted forms are:

        * a mapping ``{(state, label): next_state}``;
        * a mapping of mappings ``{state: {label: next_state}}``;
        * an iterable of triples ``(state, label, next_state)``.
    :param initial_state: initial state. The aliases ``initial`` is also
        accepted.
    :param accepting_states: states that make a sequence complete. The alias
        ``accepting`` is also accepted.
    :param states: optional iterable of states. States appearing in
        ``transitions``, ``initial_state`` and ``accepting_states`` are
        automatically registered; this argument is useful to declare states
        without outgoing transitions.

    :raises TypeError: a state or a label is not hashable, or the transitions
        argument has an unsupported type.
    :raises ValueError: the machine is not deterministic, the initial state is
        missing/unknown, or an accepting/transition state is unknown.
    """

    def __init__(
        self,
        transitions: Any,
        initial_state: Optional[Hashable] = None,
        accepting_states: Optional[Iterable[Hashable]] = None,
        states: Optional[Iterable[Hashable]] = None,
        *,
        initial: Optional[Hashable] = None,
        accepting: Optional[Iterable[Hashable]] = None,
    ) -> None:
        if initial_state is None:
            initial_state = initial
        elif initial is not None and initial != initial_state:
            raise ValueError(
                "Arguments `initial_state` and `initial` are inconsistent: "
                f"{initial_state!r} != {initial!r}."
            )

        if accepting_states is None:
            accepting_states = accepting
        elif accepting is not None and set(accepting_states) != set(accepting):
            raise ValueError(
                "Arguments `accepting_states` and `accepting` are "
                "inconsistent."
            )

        transition_map = self._normalize_transitions(transitions)

        registered = set()
        if states is not None:
            for state in states:
                _require_hashable(state, "A state")
                registered.add(state)

        for (state, label), next_state in transition_map.items():
            _require_hashable(state, "A state")
            _require_hashable(label, "A business label")
            _require_hashable(next_state, "A next state")
            registered.add(state)
            registered.add(next_state)

        if initial_state is None:
            raise ValueError(
                "A state machine requires an explicit initial state."
            )
        _require_hashable(initial_state, "The initial state")
        registered.add(initial_state)

        accepting_set = set(accepting_states or ())
        for state in accepting_set:
            _require_hashable(state, "An accepting state")
            registered.add(state)

        self._transitions: Dict[Tuple[Hashable, Hashable], Hashable] = dict(
            transition_map
        )
        self._states: FrozenSet[Hashable] = frozenset(registered)
        self._initial_state: Hashable = initial_state
        self._accepting_states: FrozenSet[Hashable] = frozenset(accepting_set)

    @staticmethod
    def _normalize_transitions(
        transitions: Any,
    ) -> Dict[Tuple[Hashable, Hashable], Hashable]:
        """Normalize the supported transition-table representations."""
        normalized: Dict[Tuple[Hashable, Hashable], Hashable] = {}

        def add(state, label, next_state):
            key = (state, label)
            if key in normalized and normalized[key] != next_state:
                raise ValueError(
                    "State machine is not deterministic: label "
                    f"{label!r} from state {state!r} maps to both "
                    f"{normalized[key]!r} and {next_state!r}."
                )
            normalized[key] = next_state

        if isinstance(transitions, Mapping):
            for key, value in transitions.items():
                if isinstance(key, tuple) and len(key) == 2:
                    if isinstance(value, Mapping):
                        raise TypeError(
                            "Ambiguous transition table: a (state, label) "
                            "key is associated with a mapping value."
                        )
                    add(key[0], key[1], value)
                elif isinstance(value, Mapping):
                    for label, next_state in value.items():
                        add(key, label, next_state)
                else:
                    raise TypeError(
                        "Transition mapping entries must be either "
                        "{(state, label): next_state} or "
                        "{state: {label: next_state}}. Received key "
                        f"{key!r} with value {value!r}."
                    )
        elif isinstance(transitions, Iterable) and not isinstance(
            transitions, (str, bytes)
        ):
            for entry in transitions:
                if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                    raise TypeError(
                        "Transitions provided as an iterable must contain "
                        "(state, label, next_state) triples. Received "
                        f"{entry!r}."
                    )
                add(entry[0], entry[1], entry[2])
        else:
            raise TypeError(
                "Unsupported type for `transitions`: "
                f"{type(transitions)}."
            )

        return normalized

    @property
    def initial_state(self) -> Hashable:
        """Initial state of the machine."""
        return self._initial_state

    @property
    def accepting_states(self) -> FrozenSet[Hashable]:
        """Set of accepting (final) states."""
        return self._accepting_states

    @property
    def states(self) -> FrozenSet[Hashable]:
        """Set of all registered states."""
        return self._states

    @property
    def alphabet(self) -> FrozenSet[Hashable]:
        """Set of business labels appearing in transitions."""
        return frozenset(label for (_, label) in self._transitions)

    @property
    def transitions(self) -> Dict[Tuple[Hashable, Hashable], Hashable]:
        """Copy of the deterministic transition table."""
        return dict(self._transitions)

    def is_accepting(self, state: Hashable) -> bool:
        """Return whether ``state`` is an accepting state.

        :param state: state to test.

        :raises ValueError: the state is unknown.
        """
        if state not in self._states:
            raise ValueError(f"Unknown state {state!r}.")
        return state in self._accepting_states

    def next_state(self, state: Hashable, label: Hashable) -> Optional[Any]:
        """Return the next state reached from ``state`` with ``label``.

        ``None`` is returned when the transition does not exist.  Note that
        ``None`` is never used internally to represent a state; a transition
        whose target is literally ``None`` is returned as ``None`` as well.
        Use :meth:`accepts` for membership tests instead of relying on this
        return value.

        :raises ValueError: the state is unknown.
        """
        if state not in self._states:
            raise ValueError(f"Unknown state {state!r}.")
        return self._transitions.get((state, label))

    def _step(self, state: Hashable, label: Hashable) -> Any:
        return self._transitions.get((state, label), _MISSING)

    def accepts(self, sequence: Iterable[Hashable]) -> bool:
        """Return whether a complete label sequence is accepted.

        Unknown labels and invalid transitions simply yield ``False``.

        :param sequence: business-label sequence.
        """
        state = self._initial_state
        for label in sequence:
            next_state = self._step(state, label)
            if next_state is _MISSING:
                return False
            state = next_state
        return state in self._accepting_states

    def validate_labels(self, labels: Iterable[Hashable]) -> None:
        """Check that every label of the machine belongs to ``labels``.

        The class-column to business-label mapping must be consistent with the
        state machine alphabet, otherwise transitions could refer to labels the
        model can never emit.

        :param labels: business labels associated with model class columns.

        :raises ValueError: the machine alphabet is not a subset of ``labels``.
        """
        available = set(labels)
        missing = self.alphabet - available
        if missing:
            raise ValueError(
                "State machine labels are not consistent with the class "
                "mapping. Missing business labels: "
                f"{sorted(map(repr, missing))}."
            )

    def __contains__(self, sequence: Iterable[Hashable]) -> bool:
        return self.accepts(sequence)

    def __repr__(self) -> str:
        return (
            "StateMachine("
            f"initial_state={self._initial_state!r}, "
            f"n_states={len(self._states)}, "
            f"n_accepting={len(self._accepting_states)}, "
            f"n_transitions={len(self._transitions)})"
        )
