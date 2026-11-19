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
This module implements sequence-level (trajectory) conformal prediction.

Motivation
----------
For tasks whose targets are *label sequences* constrained by a deterministic
state machine (for example a business process where only some event
trajectories are legal), calibrating each position independently does not
represent the uncertainty of the whole trajectory.  Positions belong to the
same sample, so their LAC decisions cannot be pooled, and intersecting
position-wise prediction sets would ignore the state-machine constraints while
still describing independent labels rather than whole trajectories.

Procedure
---------
The implemented procedure is a *sequence-level* extension of LAC:

1. A trained model provides, for every position of every sequence, its class
   probabilities.  These probabilities are consumed **as is**; the model is
   never retrained and no rescaling/normalization is applied.  They are still
   checked with puncc's native
   :func:`deel.puncc.api.utils.logit_normalization_check` (the same check used
   by :func:`deel.puncc.api.nonconformity_scores.lac_score`), so a model that
   does not emit probabilities fails explicitly.
2. For a calibration sequence, the native LAC score :math:`1 - p(y_t)` is
   computed at every position with
   :func:`deel.puncc.api.nonconformity_scores.lac_score`, and the **largest**
   score over the sequence is retained as the unique sequence-level
   nonconformity score.  One sequence is one sample, whatever its length.
3. The native corrected quantile (correction function, calibration-size check,
   ``+inf`` padding and empirical quantile, exactly as
   :class:`deel.puncc.api.calibration.BaseCalibrator` does) of those
   sequence-level scores yields a single threshold :math:`\\tau` shared by all
   positions of all new sequences.
4. For each new sequence, a **shared candidate graph** (a layered automaton)
   is built.  It represents exactly the set of complete label sequences that
   both follow the deterministic state machine (and end in an accepting
   state) and satisfy the native LAC rule :math:`1 - p_t(l_t) \\le \\tau`
   (equivalently :math:`p_t(l_t) \\ge 1 - \\tau`) at every position.  The same
   graph answers whole-sequence membership queries in linear time, returns the
   exact number of candidates with a dynamic program (all strings are never
   enumerated), and can lazily list the actual candidates when explicitly
   requested.  Position-wise label sets are *not* a representation of the
   sequence set and are only exposed as an auxiliary view.

When no legal sequence exists, the graph truthfully reports an empty candidate
set (count ``0``, empty iteration) instead of fabricating a candidate.
Insufficient calibration samples, invalid ``alpha``, un-normalized model
probabilities, incompatible class mappings and malformed graphs all raise
explicit errors.

Examples
--------
.. code-block:: python

    import numpy as np
    from deel.puncc.api.state_machine import StateMachine
    from deel.puncc.api.sequence import SequenceLACCalibrator

    # Business trajectory: OPEN -> PAID -> CLOSED (only legal path)
    machine = StateMachine(
        transitions=[
            ("start", "OPEN", "opened"),
            ("opened", "PAID", "paid"),
            ("paid", "CLOSED", "closed"),
        ],
        initial_state="start",
        accepting_states=["closed"],
    )
    labels = ["OPEN", "PAID", "CLOSED"]

    # Per-position class probabilities provided by an already trained model.
    y_pred_calib = [np.array([[0.8, 0.1, 0.1],
                              [0.1, 0.8, 0.1],
                              [0.1, 0.1, 0.8]]) for _ in range(30)]
    y_true_calib = [["OPEN", "PAID", "CLOSED"] for _ in range(30)]

    calibrator = SequenceLACCalibrator(machine, labels=labels)
    calibrator.fit(y_true=y_true_calib, y_pred=y_pred_calib)

    graphs, = calibrator.calibrate(alpha=0.1, y_pred=y_pred_calib[:2])
    graph = graphs[0]
    graph.count                     # exact number of legal candidates
    ("OPEN", "PAID", "CLOSED") in graph  # whole-sequence membership
"""

import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from deel.puncc.api.backend import get_backend
from deel.puncc.api.corrections import bonferroni
from deel.puncc.api.nonconformity_scores import lac_score
from deel.puncc.api.state_machine import StateMachine
from deel.puncc.api.state_machine import _MISSING
from deel.puncc.api.utils import alpha_calib_check
from deel.puncc.api.utils import logit_normalization_check
from deel.puncc.api.utils import quantile


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input normalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a backend array/tensor or array-like to a NumPy array.

    :raises TypeError: the input cannot be converted.
    """
    try:
        b = get_backend(x)
        return np.asarray(b.to_numpy(b.asarray(x)))
    except Exception as exc:
        raise TypeError(
            "Unsupported type for sequence predictions: "
            f"{type(x)}. Expected array-likes of class probabilities."
        ) from exc


def _infer_ndim(x: Any) -> int:
    """Infer the number of dimensions of a nested array-like object."""
    if x is None:
        return 0
    shape = getattr(x, "shape", None)
    if shape is not None:
        try:
            return len(shape)  # works for tuples and tf.TensorShape
        except TypeError:  # pragma: no cover - exotic shape objects
            pass
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return 1
        return 1 + _infer_ndim(x[0])
    return 0


def _first_element(x: Any) -> Any:
    """Return the first element of a non-empty array-like object."""
    return x[0]


def _as_probability_matrix(x: Any, *, index: Optional[int] = None) -> np.ndarray:
    """Validate and convert one sequence of per-position probabilities.

    :param x: array of shape ``(T, n_classes)``.
    :param index: optional sequence index used in error messages.

    :raises ValueError: wrong number of dimensions, empty class axis, or
        non-finite values.
    :raises TypeError: input is not an array-like of numbers.
    """
    where = "" if index is None else f" (sequence {index})"
    if x is None:
        raise TypeError(f"Sequence predictions{where} cannot be None.")
    try:
        converted = _to_numpy(x)
    except TypeError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise TypeError(
            f"Sequence predictions{where} must be numeric class probabilities."
        ) from exc

    try:
        arr = np.asarray(converted)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sequence predictions{where} must be a rectangular array of "
            "class probabilities."
        ) from exc

    if arr.dtype == object:
        raise TypeError(
            f"Sequence predictions{where} must be numeric class probabilities."
        )

    try:
        arr = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Sequence predictions{where} must be numeric class probabilities."
        ) from exc

    if arr.ndim != 2:
        raise ValueError(
            f"Each sequence prediction{where} must have shape "
            f"(n_positions, n_classes); received {arr.shape}."
        )
    if arr.shape[1] == 0:
        raise ValueError(f"Sequence predictions{where} have no classes.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Sequence predictions{where} contain non-finite probabilities."
        )
    return arr


def _as_sequence_batch(y_pred: Any) -> List[np.ndarray]:
    """Normalize predictions into a list of ``(T_i, n_classes)`` arrays.

    Accepted shapes are:

    * a single ``(T, n_classes)`` array (one sequence);
    * a ``(N, T, n_classes)`` array (``N`` equal-length sequences);
    * a list/tuple of ``(T_i, n_classes)`` arrays (``N`` variable-length
      sequences);
    * a nested list of per-position probability rows (one sequence).

    An empty iterable denotes an empty batch of sequences.

    :raises TypeError: unsupported input, or per-position class labels instead
        of probabilities.
    :raises ValueError: inconsistent sequence shapes.
    """
    if y_pred is None:
        raise TypeError("`y_pred` cannot be None.")

    if isinstance(y_pred, (list, tuple)):
        if len(y_pred) == 0:
            return []
        element_ndim = _infer_ndim(_first_element(y_pred))
        if element_ndim >= 2:
            # A batch of sequences. Handled item by item so that
            # variable-length (ragged) inputs are supported.
            return [
                _as_probability_matrix(seq, index=i)
                for i, seq in enumerate(y_pred)
            ]
        if element_ndim == 1:
            # A single sequence given as a list of per-position rows.
            return [_as_probability_matrix(y_pred)]
        raise TypeError(
            "`y_pred` must provide per-position class probabilities with "
            "shape (n_positions, n_classes) per sequence, not a sequence of "
            "scalars. Received an object with "
            f"{_infer_ndim(y_pred)} dimension(s)."
        )

    try:
        arr = np.asarray(_to_numpy(y_pred))
    except TypeError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise TypeError(
            "Sequence predictions must be array-likes of class "
            f"probabilities; received {type(y_pred)}."
        ) from exc

    if arr.dtype == object:
        raise TypeError(
            "Sequence predictions must be array-likes of numeric class "
            "probabilities. Received a ragged object array; provide a list "
            "of per-sequence 2D arrays instead."
        )

    if arr.ndim == 2:
        return [_as_probability_matrix(arr)]
    if arr.ndim == 3:
        return [
            _as_probability_matrix(arr[i], index=i)
            for i in range(arr.shape[0])
        ]
    if arr.ndim < 2:
        raise TypeError(
            "`y_pred` must provide per-position class probabilities with "
            "shape (n_positions, n_classes) per sequence, not a sequence of "
            "scalars or a single row. Received an object with "
            f"{arr.ndim} dimension(s)."
        )
    raise ValueError(
        "Sequence predictions must have at most 3 dimensions "
        f"((n_sequences, n_positions, n_classes)); received shape "
        f"{arr.shape}."
    )


def _as_label_sequence(x: Any, *, index: Optional[int] = None) -> np.ndarray:
    """Convert one sequence of true labels to a 1D array.

    A 2D sequence is accepted only when its rows are one-hot encodings, in
    which case the encoded class columns are returned.

    :raises ValueError: wrong shape or invalid one-hot encoding.
    :raises TypeError: unsupported input.
    """
    where = "" if index is None else f" (sequence {index})"
    if x is None:
        raise TypeError(f"True labels{where} cannot be None.")
    try:
        arr = np.asarray(_to_numpy(x))
    except TypeError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise TypeError(
            f"True labels{where} must be a sequence of labels."
        ) from exc

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        try:
            numeric = arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"True labels{where} must be a 1D sequence of labels; a 2D "
                "input is only accepted as one-hot encoded labels."
            ) from exc
        if numeric.shape[1] == 0:  # pragma: no cover - defensive guard
            raise ValueError(f"True labels{where} have no classes.")
        is_binary = np.all((numeric == 0) | (numeric == 1))
        rows_sum_one = np.allclose(numeric.sum(axis=1), 1.0)
        if not (is_binary and rows_sum_one):
            raise ValueError(
                f"True labels{where} must be a 1D sequence of labels. A 2D "
                "input is only accepted as one-hot encoded labels (binary "
                "rows summing to one)."
            )
        return np.argmax(numeric, axis=1)

    raise ValueError(
        f"Each true label sequence{where} must be 1D; received shape "
        f"{arr.shape}."
    )


def _as_label_batch(y_true: Any) -> List[np.ndarray]:
    """Normalize true label sequences into a list of 1D arrays."""
    if y_true is None:
        raise TypeError("`y_true` cannot be None.")

    if isinstance(y_true, (list, tuple)):
        if len(y_true) == 0:
            return []
        first = _first_element(y_true)
        if isinstance(first, (list, tuple, np.ndarray)) or _infer_ndim(
            first
        ) >= 1:
            # A batch of per-sequence label sequences. Detected from the
            # elements themselves so that a batch of length-1 label
            # sequences is not mistaken for a single sequence.
            return [
                _as_label_sequence(seq, index=i)
                for i, seq in enumerate(y_true)
            ]
        return [_as_label_sequence(y_true)]

    try:
        arr = np.asarray(_to_numpy(y_true))
    except TypeError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise TypeError("Unsupported type for `y_true`.") from exc

    if arr.dtype == object:
        raise TypeError(
            "`y_true` must be an array of labels or a list of per-sequence "
            "label arrays."
        )

    if arr.ndim == 1:
        return [_as_label_sequence(arr)]
    if arr.ndim == 2:
        return [
            _as_label_sequence(arr[i], index=i) for i in range(arr.shape[0])
        ]
    if arr.ndim == 3:
        # Batch of one-hot encoded label sequences.
        return [
            _as_label_sequence(arr[i], index=i) for i in range(arr.shape[0])
        ]
    raise TypeError(
        "`y_true` must be a sequence of labels with shape (n_positions,) per "
        f"sequence. Received an object with {arr.ndim} dimension(s)."
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Label mapping ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _resolve_labels(
    labels: Optional[Iterable[Hashable]],
    n_classes: int,
    state_machine: Optional[StateMachine] = None,
) -> Tuple[Hashable, ...]:
    """Resolve the class-column to business-label mapping.

    ``labels[j]`` is the business label associated with class column ``j``.
    When omitted, the identity mapping ``0, 1, ..., n_classes - 1`` is used.

    :raises ValueError: wrong mapping length, duplicated business labels, or
        a mapping incompatible with the state machine alphabet.
    :raises TypeError: the mapping is not an iterable of hashable labels.
    """
    if labels is None:
        resolved: Tuple[Hashable, ...] = tuple(range(n_classes))
    else:
        if isinstance(labels, (str, bytes)):
            raise TypeError(
                "`labels` must be an iterable of business labels, not a "
                "single string."
            )
        try:
            resolved = tuple(labels)
        except TypeError as exc:
            raise TypeError(
                "`labels` must be an iterable of business labels."
            ) from exc
        if len(resolved) != n_classes:
            raise ValueError(
                "The class mapping must contain one business label per class "
                f"column: expected {n_classes}, received {len(resolved)}."
            )

    seen = set()
    for label in resolved:
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError(
                f"Business label {label!r} is not hashable."
            ) from exc
        if label in seen:
            raise ValueError(
                "The class mapping must be injective: business label "
                f"{label!r} is associated with more than one class column."
            )
        seen.add(label)

    if state_machine is not None:
        state_machine.validate_labels(resolved)

    return resolved


def _encode_true_labels(
    y_true: np.ndarray,
    labels: Tuple[Hashable, ...],
    *,
    index: Optional[int] = None,
) -> np.ndarray:
    """Encode a true label sequence into class-column indices.

    Elements may be given either as class columns or as business labels. The
    encoding is unambiguous because the mapping is injective, and a business
    label that also happens to be a valid column index always denotes the
    column whose associated label it is.
    """
    where = "" if index is None else f" (sequence {index})"
    n_classes = len(labels)
    label_to_index = {label: j for j, label in enumerate(labels)}
    encoded = np.empty(len(y_true), dtype=np.int64)

    for t, value in enumerate(y_true):
        item = value.item() if hasattr(value, "item") else value
        try:
            in_mapping = item in label_to_index
        except TypeError:
            in_mapping = False

        if in_mapping:
            encoded[t] = label_to_index[item]
            continue

        try:
            as_int = int(item)
            is_integral = float(item) == as_int
        except (TypeError, ValueError):
            is_integral = False

        if is_integral and 0 <= as_int < n_classes:
            encoded[t] = as_int
            continue

        raise ValueError(
            f"Unknown true label{where} at position {t}: {item!r}. Expected "
            "either a class column in [0, n_classes) or a business label "
            "declared in the class mapping."
        )

    return encoded


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sequence LAC scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def sequence_lac_score(
    y_pred: Any,
    y_true: Any,
    *,
    labels: Optional[Iterable[Hashable]] = None,
) -> np.ndarray:
    """Sequence-level LAC nonconformity score.

    For each sequence, the native LAC score
    :math:`s_t = 1 - p(y_t)` is computed at every position with
    :func:`deel.puncc.api.nonconformity_scores.lac_score`, and the sequence
    score is the largest of them:

    .. math::

        S = \\max_{t=1,\\dots,T} \\big(1 - \\hat{p}_t(y_t)\\big).

    Each sequence contributes exactly one score, whatever its length.

    :param y_pred: per-sequence class probabilities. Accepted forms are the
        same as for :func:`sequence_lac_set`.
    :param y_true: per-sequence true labels (class columns or business
        labels).
    :param labels: optional class-column to business-label mapping.

    :returns: array of shape ``(n_sequences,)``.
    :rtype: numpy.ndarray

    :raises ValueError: inconsistent batch sizes, sequence lengths, unknown
        labels, or probabilities that are not normalized.
    :raises TypeError: unsupported inputs.
    :raises RuntimeError: no sequence provided.
    """
    preds = _as_sequence_batch(y_pred)
    truths = _as_label_batch(y_true)

    if len(preds) != len(truths):
        raise ValueError(
            "The number of predicted sequences and the number of true label "
            f"sequences differ: {len(preds)} != {len(truths)}."
        )
    if not preds:
        raise RuntimeError(
            "At least one sequence is required to compute sequence scores."
        )

    n_classes = preds[0].shape[1]
    resolved = _resolve_labels(labels, n_classes)

    scores = np.empty(len(preds), dtype=float)
    for i, (probs, truth) in enumerate(zip(preds, truths)):
        if probs.shape[1] != n_classes:
            raise ValueError(
                "All sequences must share the same number of class columns: "
                f"sequence {i} has {probs.shape[1]} instead of {n_classes}."
            )
        if probs.shape[0] != len(truth):
            raise ValueError(
                "Each sequence prediction must have one probability vector "
                f"per true label: sequence {i} has {probs.shape[0]} "
                f"positions but {len(truth)} labels."
            )
        if len(truth) == 0:
            raise ValueError(
                "A sequence of length 0 has no LAC score. Sequence-level "
                "calibration requires at least one position per sequence."
            )
        encoded = _encode_true_labels(truth, resolved, index=i)
        position_scores = np.asarray(lac_score(probs, encoded), dtype=float)
        scores[i] = float(np.max(position_scores))

    return scores


def sequence_score(
    y_pred: Any,
    y_true: Any,
    *,
    labels: Optional[Iterable[Hashable]] = None,
) -> np.ndarray:
    """Alias of :func:`sequence_lac_score`."""
    return sequence_lac_score(y_pred, y_true, labels=labels)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Candidate graph ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class SequenceCandidateGraph:
    """Shared candidate graph of complete label sequences.

    Given the per-position class probabilities of one sequence and a global
    threshold :math:`\\tau`, the graph represents exactly the set of complete
    label sequences :math:`(l_1,\\dots,l_T)` such that:

    * every prefix follows the deterministic state machine, and the final
      state is accepting;
    * at every position :math:`t`, the LAC score of the chosen label is
      ``<= tau``, that is :math:`1 - p_t(l_t) \\le \\tau`, equivalently
      :math:`p_t(l_t) \\ge 1 - \\tau` (the native LAC prediction-set rule).

    The graph is *shared*: membership, counting and enumeration all read the
    same layered structure.  Counting is exact and performed by a dynamic
    program; the set of all possible strings is never enumerated.  Enumeration
    of the actual candidates is lazy and only materialized when the caller
    iterates.

    :param y_pred: class probabilities of one sequence, shape
        ``(T, n_classes)``.
    :param scores_quantile: calibration threshold :math:`\\tau` (a scalar,
        possibly ``+inf``).
    :param state_machine: deterministic state machine over business labels.
    :param labels: optional class-column to business-label mapping.

    :raises TypeError: invalid state machine, labels, or predictions.
    :raises ValueError: non-finite threshold, inconsistent mapping, or
        invalid probabilities.
    """

    def __init__(
        self,
        y_pred: Any,
        scores_quantile: float,
        state_machine: StateMachine,
        labels: Optional[Iterable[Hashable]] = None,
    ) -> None:
        if not isinstance(state_machine, StateMachine):
            raise TypeError(
                "`state_machine` must be a StateMachine instance."
            )

        probs = _as_probability_matrix(y_pred)
        logit_normalization_check(probs)

        tau = self._validate_threshold(scores_quantile)
        resolved = _resolve_labels(labels, probs.shape[1], state_machine)

        self._state_machine = state_machine
        self._labels = resolved
        self._label_to_index = {
            label: j for j, label in enumerate(resolved)
        }
        self._probabilities = probs
        self._threshold = tau
        self._length = probs.shape[0]

        # Native LAC rule: include a label iff p >= 1 - tau.
        # A +inf threshold allows every label at every position.
        allowed = probs >= (1.0 - tau)
        self._allowed = np.asarray(allowed, dtype=bool)

        self._layers: List[Dict[Hashable, Dict[Hashable, Hashable]]] = []
        self._states_at: List[set] = []
        self._can_finish: List[set] = []
        self._count = 0
        self._build()

    @staticmethod
    def _validate_threshold(scores_quantile: Any) -> float:
        if scores_quantile is None:
            raise TypeError("`scores_quantile` cannot be None.")
        if isinstance(scores_quantile, (list, tuple, np.ndarray)) and np.size(
            scores_quantile
        ) != 1:
            raise TypeError(
                "`scores_quantile` must be a scalar threshold for "
                "sequence-level prediction, received an array."
            )
        try:
            tau = float(np.asarray(scores_quantile).reshape(-1)[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise TypeError(
                "`scores_quantile` must be a scalar threshold, received "
                f"{scores_quantile!r}."
            ) from exc
        if math.isnan(tau):
            raise ValueError("`scores_quantile` cannot be NaN.")
        if tau == -math.inf:
            raise ValueError("`scores_quantile` cannot be -inf.")
        return tau

    def _build(self) -> None:
        """Build the layered graph and compute the exact candidate count."""
        machine = self._state_machine
        initial = machine.initial_state
        accepting = machine.accepting_states
        length = self._length

        # Forward reachable states and transitions, layer by layer.
        states_at: List[set] = [{initial}]
        layers: List[Dict[Hashable, Dict[Hashable, Hashable]]] = []

        frontier = {initial}
        for t in range(length):
            layer: Dict[Hashable, Dict[Hashable, Hashable]] = {}
            next_frontier = set()
            allowed_columns = np.nonzero(self._allowed[t])[0]
            for state in frontier:
                successors: Dict[Hashable, Hashable] = {}
                for j in allowed_columns:
                    label = self._labels[int(j)]
                    next_state = machine._step(state, label)
                    if next_state is _MISSING:
                        continue
                    successors[label] = next_state
                    next_frontier.add(next_state)
                if successors:
                    layer[state] = successors
            layers.append(layer)
            frontier = next_frontier
            states_at.append(set(frontier))

        # Backward pruning: keep only states that can still reach an
        # accepting state after the remaining number of positions.
        can_finish: List[set] = [set() for _ in range(length + 1)]
        can_finish[length] = states_at[length] & set(accepting)
        for t in range(length - 1, -1, -1):
            layer = layers[t]
            good_next = can_finish[t + 1]
            good_current = set()
            for state in states_at[t]:
                successors = layer.get(state)
                if not successors:
                    continue
                if any(
                    next_state in good_next
                    for next_state in successors.values()
                ):
                    good_current.add(state)
            can_finish[t] = good_current

        # Exact count by dynamic programming over the pruned graph.
        count = 0
        if length == 0:
            count = 1 if initial in accepting else 0
        elif initial in can_finish[0]:
            counts: Dict[Hashable, int] = {initial: 1}
            for t in range(length):
                next_counts: Dict[Hashable, int] = {}
                good_next = can_finish[t + 1]
                for state, multiplicity in counts.items():
                    for next_state in layers[t][state].values():
                        if next_state in good_next:
                            next_counts[next_state] = (
                                next_counts.get(next_state, 0) + multiplicity
                            )
                counts = next_counts
            count = sum(counts.values())

        self._states_at = states_at
        self._layers = layers
        self._can_finish = can_finish
        self._count = int(count)

    # ------------------------- properties -------------------------

    @property
    def state_machine(self) -> StateMachine:
        """State machine used to constrain candidate sequences."""
        return self._state_machine

    @property
    def labels(self) -> Tuple[Hashable, ...]:
        """Class-column to business-label mapping."""
        return self._labels

    @property
    def threshold(self) -> float:
        """Global LAC threshold used to build the graph."""
        return self._threshold

    @property
    def length(self) -> int:
        """Number of positions of the sequence."""
        return self._length

    @property
    def n_classes(self) -> int:
        """Number of class columns."""
        return len(self._labels)

    @property
    def n_candidates(self) -> int:
        """Exact number of complete candidate sequences."""
        return self._count

    @property
    def count(self) -> int:
        """Exact number of complete candidate sequences.

        The count is computed by dynamic programming and is exact; it is
        available even when the candidate set is far too large to enumerate.
        """
        return self._count

    @property
    def total_count(self) -> int:
        """Alias of :attr:`count`."""
        return self._count

    @property
    def is_empty(self) -> bool:
        """Whether no legal candidate sequence exists."""
        return self._count == 0

    @property
    def candidates(self) -> "SequenceCandidateSet":
        """Set-like view over the candidate sequences."""
        return SequenceCandidateSet(self)

    @property
    def probabilities(self) -> np.ndarray:
        """Copy of the per-position class probabilities."""
        return self._probabilities.copy()

    def count_candidates(self) -> int:
        """Return the exact number of complete candidate sequences."""
        return self._count

    def allowed_labels(self, position: int, state: Hashable = None) -> Tuple:
        """Labels allowed at ``position`` with respect to the threshold.

        This auxiliary view reports the native LAC per-position decision. The
        state machine is applied only when ``state`` is provided. The result
        is **not** a representation of whole candidate sequences; use
        :meth:`contains` and :meth:`count` for sequence-level queries.

        :param position: position index, ``0 <= position < length``.
        :param state: optional machine state whose outgoing transitions are
            required.

        :raises IndexError: position is out of range.
        :raises ValueError: the provided state is unknown.
        """
        if not 0 <= position < self._length:
            raise IndexError(
                f"Position {position} is out of range for a sequence of "
                f"length {self._length}."
            )
        if state is None:
            columns = np.nonzero(self._allowed[position])[0]
            return tuple(self._labels[int(j)] for j in columns)

        if state not in self._state_machine.states:
            raise ValueError(f"Unknown state {state!r}.")
        result = []
        for j in np.nonzero(self._allowed[position])[0]:
            label = self._labels[int(j)]
            if self._state_machine._step(state, label) is not _MISSING:
                result.append(label)
        return tuple(result)

    # ------------------------- sequence queries -------------------------

    @staticmethod
    def _coerce_sequence(sequence: Any) -> tuple:
        if isinstance(sequence, (str, bytes)):
            return (sequence,)
        try:
            return tuple(sequence)
        except TypeError as exc:
            raise TypeError(
                "A candidate sequence must be an iterable of business "
                f"labels; received {sequence!r}."
            ) from exc

    def contains(self, sequence: Iterable[Hashable]) -> bool:
        """Return whether a whole label sequence is a legal candidate.

        The query walks the state machine once, so it is linear in the
        sequence length and does not depend on the number of candidates.

        :param sequence: complete business-label sequence.
        """
        items = self._coerce_sequence(sequence)
        if len(items) != self._length:
            return False

        state = self._state_machine.initial_state
        for t, label in enumerate(items):
            try:
                column = self._label_to_index.get(label)
            except TypeError:
                return False
            if column is None or not self._allowed[t, column]:
                return False
            next_state = self._state_machine._step(state, label)
            if next_state is _MISSING:
                return False
            state = next_state
        return state in self._state_machine.accepting_states

    def is_candidate(self, sequence: Iterable[Hashable]) -> bool:
        """Alias of :meth:`contains`."""
        return self.contains(sequence)

    def accepts(self, sequence: Iterable[Hashable]) -> bool:
        """Alias of :meth:`contains`."""
        return self.contains(sequence)

    def sequences(self) -> Iterator[Tuple]:
        """Lazily enumerate the complete candidate sequences.

        Iteration is performed on demand over the pruned graph. For large
        candidate sets, prefer :attr:`count` and :meth:`contains`, which do
        not enumerate strings.
        """
        if self._count == 0:
            return

        machine = self._state_machine
        accepting = machine.accepting_states
        length = self._length

        if length == 0:
            if machine.initial_state in accepting:
                yield ()
            return

        if machine.initial_state not in self._can_finish[0]:
            return

        def walk(t, state, prefix):
            if t == length:
                if state in accepting:
                    yield tuple(prefix)
                return
            successors = self._layers[t].get(state)
            if not successors:
                return
            good_next = self._can_finish[t + 1]
            for label, next_state in successors.items():
                if next_state in good_next:
                    prefix.append(label)
                    yield from walk(t + 1, next_state, prefix)
                    prefix.pop()

        yield from walk(0, machine.initial_state, [])

    def candidate_sequences(self) -> Iterator[Tuple]:
        """Alias of :meth:`sequences`."""
        return self.sequences()

    def __contains__(self, sequence: Any) -> bool:
        return self.contains(sequence)

    def __iter__(self) -> Iterator[Tuple]:
        return self.sequences()

    def __len__(self) -> int:
        if self._count > np.iinfo(np.intp).max:  # pragma: no cover
            raise OverflowError(
                "The number of candidates does not fit into `len`. Use "
                "`count` or `count_candidates()` for an exact big integer."
            )
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def __repr__(self) -> str:
        state = "empty" if self.is_empty else "non-empty"
        return (
            "SequenceCandidateGraph("
            f"length={self._length}, n_classes={self.n_classes}, "
            f"threshold={self._threshold}, count={self._count}, {state})"
        )


class SequenceCandidateSet:
    """Set-like view over the candidates of a :class:`SequenceCandidateGraph`.

    It exposes whole-sequence membership and exact cardinality, and iterates
    lazily over the actual candidates.

    :param graph: sequence candidate graph.
    """

    def __init__(self, graph: SequenceCandidateGraph) -> None:
        if not isinstance(graph, SequenceCandidateGraph):
            raise TypeError("`graph` must be a SequenceCandidateGraph.")
        self.graph = graph

    @property
    def threshold(self) -> float:
        """Global LAC threshold."""
        return self.graph.threshold

    @property
    def length(self) -> int:
        """Sequence length."""
        return self.graph.length

    @property
    def count(self) -> int:
        """Exact number of candidate sequences."""
        return self.graph.count

    @property
    def n_candidates(self) -> int:
        """Alias of :attr:`count`."""
        return self.graph.n_candidates

    @property
    def is_empty(self) -> bool:
        """Whether the candidate set is empty."""
        return self.graph.is_empty

    def count_candidates(self) -> int:
        """Return the exact number of candidate sequences."""
        return self.graph.count_candidates()

    def contains(self, sequence: Iterable[Hashable]) -> bool:
        """Return whether a whole label sequence belongs to the set."""
        return self.graph.contains(sequence)

    def sequences(self) -> Iterator[Tuple]:
        """Lazily enumerate candidate sequences."""
        return self.graph.sequences()

    def candidate_sequences(self) -> Iterator[Tuple]:
        """Alias of :meth:`sequences`."""
        return self.graph.sequences()

    def __contains__(self, sequence: Any) -> bool:
        return self.graph.contains(sequence)

    def __iter__(self) -> Iterator[Tuple]:
        return self.graph.sequences()

    def __len__(self) -> int:
        return len(self.graph)

    def __bool__(self) -> bool:
        return bool(self.graph)

    def __repr__(self) -> str:
        return (
            "SequenceCandidateSet("
            f"length={self.length}, count={self.count}, "
            f"threshold={self.threshold})"
        )


# Friendly aliases used in the documentation and by downstream code.
CandidateSequenceGraph = SequenceCandidateGraph
SharedStateGraph = SequenceCandidateGraph
SharedCandidateGraph = SequenceCandidateGraph


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prediction sets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def sequence_lac_set(
    Y_pred: Any,
    scores_quantile: float,
    state_machine: StateMachine = None,
    labels: Optional[Iterable[Hashable]] = None,
) -> Tuple[List[SequenceCandidateGraph]]:
    """Sequence-level LAC prediction sets.

    For every new sequence, build the shared candidate graph representing all
    complete legal label sequences w.r.t. the state machine and the calibrated
    threshold.

    :param Y_pred: per-sequence class probabilities. A single ``(T, C)``
        array, a ``(N, T, C)`` array, or a list of ``(T_i, C)`` arrays.
    :param float scores_quantile: calibrated sequence-level threshold.
    :param StateMachine state_machine: deterministic state machine over
        business labels.
    :param labels: optional class-column to business-label mapping.

    :returns: a 1-tuple containing the list of candidate graphs (one per
        sequence), mirroring the ``(prediction_sets,)`` convention of
        :func:`deel.puncc.api.prediction_sets.lac_set`.
    :rtype: Tuple[List[SequenceCandidateGraph]]

    :raises TypeError: missing or invalid state machine / predictions.
    :raises ValueError: invalid threshold or inconsistent mapping.
    """
    if state_machine is None:
        raise TypeError(
            "`state_machine` is required to build sequence prediction sets."
        )
    if not isinstance(state_machine, StateMachine):
        raise TypeError("`state_machine` must be a StateMachine instance.")

    batch = _as_sequence_batch(Y_pred)
    graphs = [
        SequenceCandidateGraph(
            probs, scores_quantile, state_machine, labels=labels
        )
        for probs in batch
    ]
    return (graphs,)


# Alias with the wording used in the procedure description.
sequence_prediction_set = sequence_lac_set


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calibrator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class SequenceLACCalibrator:
    """Calibrator for sequence-level LAC conformal prediction.

    The calibrator stores one score per calibration sequence: the maximum
    native LAC score over its positions (see :func:`sequence_lac_score`).  It
    then computes the native corrected quantile (correction function,
    calibration-size check, ``+inf`` padding and empirical quantile, exactly
    as :class:`deel.puncc.api.calibration.BaseCalibrator`) of these scores,
    yielding a single threshold shared by all positions of all new sequences.

    The underlying model is not part of the calibrator: probabilities are
    consumed as provided by the (already trained) model.

    The ``nonconf_score_func``, ``pred_set_func`` and ``weight_func`` keyword
    arguments exist for compatibility with
    :class:`deel.puncc.api.conformalization.SplitConformalPredictor`, which
    constructs its calibrator with them.  Sequence scoring and sequence
    prediction-set construction are implemented by this class.

    :param StateMachine state_machine: deterministic state machine over
        business labels. Can also be provided to :meth:`fit` or
        :meth:`calibrate`.
    :param labels: optional class-column to business-label mapping.
    :param Callable correction: correction function applied to ``alpha``
        before computing the quantile. Defaults to
        :func:`deel.puncc.api.corrections.bonferroni`.
    """

    def __init__(
        self,
        state_machine: Optional[StateMachine] = None,
        labels: Optional[Iterable[Hashable]] = None,
        correction: Callable = bonferroni,
        *,
        nonconf_score_func: Optional[Callable] = None,
        pred_set_func: Optional[Callable] = None,
        weight_func: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        del kwargs
        if state_machine is not None and not isinstance(
            state_machine, StateMachine
        ):
            raise TypeError(
                "`state_machine` must be a StateMachine instance."
            )
        if correction is None or not callable(correction):
            raise TypeError("`correction` must be a callable.")

        self.state_machine = state_machine
        self.labels = None if labels is None else tuple(labels)
        self.correction = correction

        # Kept for interface compatibility with BaseCalibrator.
        self.nonconf_score_func = (
            nonconf_score_func
            if nonconf_score_func is not None
            else sequence_lac_score
        )
        self.pred_set_func = (
            pred_set_func if pred_set_func is not None else sequence_lac_set
        )
        self.weight_func = weight_func

        self._residuals: Optional[np.ndarray] = None
        self._len_calib = 0
        self._feature_axis = None
        self._norm_weights = None

    # ------------------------- fitting -------------------------

    def fit(
        self,
        *,
        y_true: Any,
        y_pred: Any,
        X: Any = None,
        state_machine: Optional[StateMachine] = None,
        labels: Optional[Iterable[Hashable]] = None,
        **kwargs,
    ) -> None:
        """Estimate sequence-level nonconformity scores.

        :param y_true: true label sequences (class columns, business labels
            or one-hot rows), one entry per sequence.
        :param y_pred: per-sequence class probabilities.
        :param X: ignored; accepted for compatibility with
            :class:`deel.puncc.api.calibration.BaseCalibrator`.
        :param state_machine: optional state machine overriding the one given
            at construction (required if none was provided).
        :param labels: optional mapping overriding the one given at
            construction.

        :raises TypeError: unsupported inputs or missing state machine.
        :raises ValueError: inconsistent batch sizes or labels, empty
            sequences, or unnormalized probabilities.
        :raises RuntimeError: no calibration sequence provided.
        """
        del X, kwargs
        machine, resolved = self._resolve_configuration(
            state_machine, labels, y_pred
        )
        self._validate_true_sequences(y_true, machine, resolved)
        scores = sequence_lac_score(y_pred, y_true, labels=resolved)
        if scores.size == 0:
            raise RuntimeError(
                "At least one calibration sequence is required."
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "Sequence-level LAC scores must be finite; check the "
                "provided probabilities."
            )

        self.state_machine = machine
        self.labels = tuple(resolved)
        self._residuals = scores
        self._len_calib = int(scores.size)
        self._feature_axis = None
        self._norm_weights = None

    def _validate_true_sequences(
        self,
        y_true: Any,
        machine: StateMachine,
        labels: Tuple[Hashable, ...],
    ) -> None:
        """Check that calibration label paths obey the state machine.

        Business trajectories are defined as legal label sequences. A
        calibration sequence that the machine rejects (or that uses labels
        outside the class mapping) makes the calibration data inconsistent
        with the stated constraint, so it fails explicitly instead of
        silently producing a score for an impossible trajectory.
        """
        for i, truth in enumerate(_as_label_batch(y_true)):
            encoded = _encode_true_labels(truth, labels, index=i)
            sequence = tuple(labels[int(column)] for column in encoded)
            if not machine.accepts(sequence):
                raise ValueError(
                    f"Calibration sequence {i} is not accepted by the state "
                    f"machine: {sequence!r}."
                )

    def _resolve_configuration(
        self,
        state_machine: Optional[StateMachine],
        labels: Optional[Iterable[Hashable]],
        y_pred: Any,
    ) -> Tuple[StateMachine, Tuple[Hashable, ...]]:
        machine = (
            state_machine if state_machine is not None else self.state_machine
        )
        if machine is None:
            raise TypeError(
                "A state machine is required for sequence-level calibration."
            )
        if not isinstance(machine, StateMachine):
            raise TypeError("`state_machine` must be a StateMachine instance.")

        mapping = labels if labels is not None else self.labels
        if (
            self._residuals is not None
            and labels is not None
            and self.labels is not None
            and tuple(labels) != self.labels
        ):
            raise ValueError(
                "The class-column to business-label mapping must be "
                "consistent with calibration. Fitted mapping: "
                f"{self.labels!r}; provided mapping: {tuple(labels)!r}."
            )

        preds = _as_sequence_batch(y_pred)
        if not preds:
            raise ValueError("At least one sequence is required.")

        n_classes = preds[0].shape[1]
        resolved = _resolve_labels(mapping, n_classes, machine)
        for i, probs in enumerate(preds):
            if probs.shape[1] != n_classes:
                raise ValueError(
                    "All sequences must share the same number of class "
                    f"columns: sequence {i} has {probs.shape[1]} instead of "
                    f"{n_classes}."
                )
        return machine, resolved

    # ------------------------- calibration -------------------------

    def compute_quantile(
        self,
        *,
        alpha: float,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = None,
    ) -> float:
        """Compute the native corrected quantile of sequence scores.

        This mirrors
        :meth:`deel.puncc.api.calibration.BaseCalibrator.compute_quantile`:
        the correction function is applied to ``alpha``, the calibration size
        is checked (insufficient calibration sequences raise), an infinite
        score is appended (Tibshirani's lemma) and the ``1 - alpha``
        empirical quantile is returned.

        :param float alpha: significance level.
        :param weights: optional weights associated with calibration
            sequences. Defaults to None (equiprobable sequences).
        :param Callable correction: optional correction function overriding
            the one given at construction.

        :returns: threshold :math:`\\tau`.
        :rtype: float

        :raises RuntimeError: :meth:`fit` was not called.
        :raises ValueError: ``alpha`` is inconsistent with the calibration
            sample size (not enough calibration sequences), or invalid
            weights.
        """
        if self._residuals is None:
            raise RuntimeError(
                "Run `fit` method before calling `calibrate`."
            )

        if correction is not None:
            correction_func = correction
        else:
            correction_func = self.correction
        corrected_alpha = correction_func(alpha)

        weights_np = None
        if weights is not None:
            bw = get_backend(weights)
            weights_np = np.asarray(
                bw.to_numpy(bw.asarray(weights)), dtype=float
            )
        else:
            alpha_calib_check(alpha=corrected_alpha, n=self._len_calib)

        residuals = np.asarray(self._residuals, dtype=float)
        lemma_residuals = np.concatenate((residuals, np.array([np.inf])))

        residuals_q = quantile(
            lemma_residuals,
            1 - corrected_alpha,
            w=weights_np,
        )
        return float(residuals_q)

    def calibrate(
        self,
        *,
        alpha: float,
        y_pred: Any,
        X: Any = None,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = None,
        state_machine: Optional[StateMachine] = None,
        labels: Optional[Iterable[Hashable]] = None,
        **kwargs,
    ) -> Tuple[List[SequenceCandidateGraph]]:
        """Build sequence-level candidate graphs for new sequences.

        :param float alpha: significance level.
        :param y_pred: per-sequence class probabilities of the new
            sequences.
        :param X: ignored; accepted for compatibility with
            :class:`deel.puncc.api.calibration.BaseCalibrator`.
        :param weights: optional calibration-sequence weights.
        :param Callable correction: optional correction function override.
        :param state_machine: optional state machine override.
        :param labels: optional mapping override (must be consistent with the
            fitted mapping).

        :returns: a 1-tuple containing one candidate graph per sequence.
        :rtype: Tuple[List[SequenceCandidateGraph]]

        :raises RuntimeError: insufficient data.
        :raises ValueError: insufficient calibration samples, invalid
            threshold, inconsistent mapping, or model output errors.
        """
        del X, kwargs
        machine, resolved = self._resolve_configuration(
            state_machine, labels, y_pred
        )
        threshold = self.compute_quantile(
            alpha=alpha, weights=weights, correction=correction
        )
        return sequence_lac_set(
            y_pred,
            threshold,
            state_machine=machine,
            labels=resolved,
        )

    # ------------------------- accessors -------------------------

    def get_nonconformity_scores(self) -> Optional[np.ndarray]:
        """Return the sequence-level nonconformity scores, if fitted."""
        return self._residuals

    def get_threshold(
        self,
        *,
        alpha: float,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = None,
    ) -> float:
        """Alias of :meth:`compute_quantile`."""
        return self.compute_quantile(
            alpha=alpha, weights=weights, correction=correction
        )

    def get_norm_weights(self):
        """Return normalized weights, if any (interface compatibility)."""
        return self._norm_weights

    def set_norm_weights(self, norm_weights) -> None:
        """Set normalized weights (interface compatibility)."""
        self._norm_weights = norm_weights

    def get_weights(self):
        """Return normalized weights, if any (interface compatibility)."""
        return self._norm_weights

    @staticmethod
    def barber_weights(weights) -> np.ndarray:
        """Normalize inference weights (Barber et al.).

        Mirrors
        :meth:`deel.puncc.api.calibration.BaseCalibrator.barber_weights` so
        that sequence-level calibration stays compatible with
        :class:`deel.puncc.api.conformalization.SplitConformalPredictor`.
        """
        bw = get_backend(weights)
        values = np.asarray(bw.to_numpy(bw.asarray(weights)), dtype=float)
        total = np.sum(values)
        normalized = np.zeros(len(values) + 1)
        normalized[: len(values)] = values / (total + 1)
        normalized[len(values)] = 1 / (total + 1)
        return normalized


# Naming aliases kept for convenience and discoverability.
SequenceCalibrator = SequenceLACCalibrator
SequenceLevelCalibrator = SequenceLACCalibrator
