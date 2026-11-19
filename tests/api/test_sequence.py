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
"""Tests for sequence-level (trajectory) conformal prediction."""

import itertools

import numpy as np
import pytest

from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.nonconformity_scores import lac_score
from deel.puncc.api.prediction_sets import lac_set
from deel.puncc.api.sequence import SequenceCandidateGraph
from deel.puncc.api.sequence import SequenceCandidateSet
from deel.puncc.api.sequence import SequenceLACCalibrator
from deel.puncc.api.sequence import sequence_lac_score
from deel.puncc.api.sequence import sequence_lac_set
from deel.puncc.api.state_machine import StateMachine


LABELS = ["OPEN", "PAID", "CLOSED"]


def make_machine():
    """OPEN -> PAID -> CLOSED: a single legal trajectory."""
    return StateMachine(
        transitions=[
            ("start", "OPEN", "opened"),
            ("opened", "PAID", "paid"),
            ("paid", "CLOSED", "closed"),
        ],
        initial_state="start",
        accepting_states=["closed"],
    )


def make_branching_machine():
    """Two legal trajectories: A A A and A B B."""
    return StateMachine(
        transitions=[
            ("s0", "A", "s1"),
            ("s0", "B", "s1"),
            ("s1", "A", "s2"),
            ("s1", "B", "s2"),
            ("s2", "A", "s3"),
            ("s2", "B", "s3"),
        ],
        initial_state="s0",
        accepting_states=["s3"],
    )


def simplex_probs(rng, n_positions, n_classes):
    matrix = rng.random((n_positions, n_classes))
    return matrix / matrix.sum(axis=1, keepdims=True)


def brute_force_candidates(probs, tau, machine, labels):
    """Enumerate all candidate sequences by brute force."""
    allowed = [
        [labels[j] for j in range(probs.shape[1]) if probs[t, j] >= 1 - tau]
        for t in range(probs.shape[0])
    ]
    candidates = []
    for sequence in itertools.product(*allowed):
        if machine.accepts(sequence):
            candidates.append(tuple(sequence))
    return candidates


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ State machine ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_state_machine_supported_transition_forms():
    triples = [("s0", "A", "s1"), ("s1", "B", "s2")]
    nested = {"s0": {"A": "s1"}, "s1": {"B": "s2"}}
    pairs = {("s0", "A"): "s1", ("s1", "B"): "s2"}

    for transitions in (triples, nested, pairs):
        machine = StateMachine(
            transitions,
            initial_state="s0",
            accepting_states=["s2"],
        )
        assert machine.accepts(["A", "B"])
        assert not machine.accepts(["B", "A"])
        assert machine.accepts([]) is False


def test_state_machine_rejects_non_determinism():
    with pytest.raises(ValueError):
        StateMachine(
            transitions=[
                ("s0", "A", "s1"),
                ("s0", "A", "s2"),
            ],
            initial_state="s0",
            accepting_states=["s1"],
        )


def test_state_machine_requires_initial_state():
    with pytest.raises(ValueError):
        StateMachine(transitions=[("s0", "A", "s1")], accepting_states=["s1"])


def test_state_machine_empty_sequence_acceptance():
    machine = StateMachine(
        transitions=[("s0", "A", "s0")],
        initial_state="s0",
        accepting_states=["s0"],
    )
    assert machine.accepts([])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sequence scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_sequence_score_is_max_of_native_lac_scores():
    probs = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    truth = ["OPEN", "PAID", "CLOSED"]
    score = sequence_lac_score(probs, truth, labels=LABELS)[0]

    expected = np.max(lac_score(probs, np.array([0, 1, 2])))
    assert score == pytest.approx(expected)
    assert score == pytest.approx(0.4)


def test_sequence_score_uses_business_labels_through_mapping():
    labels = ["B", "A"]  # class column 0 -> B, column 1 -> A
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    truth = ["B", "A"]

    score = sequence_lac_score(probs, truth, labels=labels)[0]
    assert score == pytest.approx(0.2)

    # Integer columns denote class columns and must give the same result.
    score_columns = sequence_lac_score(probs, [0, 1], labels=labels)[0]
    assert score_columns == pytest.approx(score)


def test_sequence_score_mapping_must_be_injective():
    probs = np.zeros((2, 3))
    probs[:] = 1 / 3
    with pytest.raises(ValueError):
        sequence_lac_score(probs, ["A", "A"], labels=["A", "A", "B"])


def test_sequence_score_mapping_must_match_class_count():
    probs = np.full((2, 3), 1 / 3)
    with pytest.raises(ValueError):
        sequence_lac_score(probs, ["A", "A"], labels=["A", "B"])


def test_sequence_score_rejects_labels_outside_mapping():
    probs = np.full((2, 3), 1 / 3)
    with pytest.raises(ValueError):
        sequence_lac_score(probs, ["OPEN", "UNKNOWN"], labels=LABELS)


def test_sequence_score_accepts_one_hot_labels():
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    one_hot = np.array([[1, 0, 0], [0, 0, 1]])
    expected = sequence_lac_score(probs, ["OPEN", "CLOSED"], labels=LABELS)[0]

    # A single one-hot sequence wrapped in a list.
    score = sequence_lac_score(probs, [one_hot], labels=LABELS)[0]
    assert score == pytest.approx(expected)

    # A batch of one-hot sequences as a 3D array.
    score = sequence_lac_score(probs, one_hot[np.newaxis], labels=LABELS)[0]
    assert score == pytest.approx(expected)


def test_variable_length_sequences_count_as_one_sample_each():
    rng = np.random.RandomState(0)
    probs = [
        simplex_probs(rng, 2, 3),
        simplex_probs(rng, 5, 3),
        simplex_probs(rng, 1, 3),
    ]
    truth = [
        np.array([0, 1]),
        np.array([0, 1, 2, 2, 1]),
        np.array([2]),
    ]
    scores = sequence_lac_score(probs, truth, labels=LABELS)
    assert scores.shape == (3,)

    for i, (p, y) in enumerate(zip(probs, truth)):
        assert scores[i] == pytest.approx(np.max(1 - p[np.arange(len(y)), y]))


def test_sequence_score_batch_size_mismatch():
    probs = np.full((2, 2, 3), 1 / 3)
    with pytest.raises(ValueError):
        sequence_lac_score(probs, [["OPEN", "PAID"]])


def test_sequence_score_rejects_empty_sequences():
    probs = [np.full((0, 3), 1 / 3)]
    truth = [np.array([], dtype=int)]
    with pytest.raises(ValueError):
        sequence_lac_score(probs, truth, labels=LABELS)


def test_sequence_score_rejects_non_normalized_probabilities():
    probs = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.7]])
    with pytest.raises(ValueError):
        sequence_lac_score(probs, ["OPEN", "PAID"], labels=LABELS)


def test_sequence_score_rejects_scalar_predictions():
    with pytest.raises(TypeError):
        sequence_lac_score([0, 1], ["OPEN", "PAID"], labels=LABELS)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Candidate graph ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_graph_count_matches_brute_force():
    rng = np.random.RandomState(42)
    machine = make_branching_machine()
    labels = ["A", "B"]

    for _ in range(40):
        length = int(rng.randint(1, 4))
        probs = simplex_probs(rng, length, 2)
        tau = float(rng.uniform(0.0, 1.0))
        graph = SequenceCandidateGraph(probs, tau, machine, labels=labels)

        expected = brute_force_candidates(probs, tau, machine, labels)
        assert graph.count == len(expected)
        assert graph.n_candidates == len(expected)
        assert len(graph) == len(expected)

        got = tuple(graph.sequences())
        assert set(got) == set(expected)
        assert len(got) == len(expected)

        for sequence in itertools.product(labels, repeat=length):
            assert graph.contains(sequence) == (sequence in expected)


def test_graph_membership_is_whole_sequence():
    machine = make_machine()
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    graph = SequenceCandidateGraph(probs, 0.5, machine, labels=LABELS)

    assert graph.contains(("OPEN", "PAID", "CLOSED"))
    assert ("OPEN", "PAID", "CLOSED") in graph
    assert graph.is_candidate(("OPEN", "PAID", "CLOSED"))

    # Individual labels are all allowed position-wise, but the incomplete
    # sequences are not legal candidates.
    assert not graph.contains(("OPEN",))
    assert not graph.contains(("PAID", "OPEN", "CLOSED"))
    assert not graph.contains(("OPEN", "PAID"))
    assert graph.count == 1


def test_graph_threshold_rule_is_native_lac_rule():
    machine = make_machine()
    probs = np.array(
        [
            [0.75, 0.15, 0.10],
            [0.05, 0.90, 0.05],
            [0.20, 0.30, 0.50],
        ]
    )
    # tau = 0.4 -> keep labels with p >= 0.6: OPEN, PAID, (CLOSED: 0.5 dropped)
    graph = SequenceCandidateGraph(probs, 0.4, machine, labels=LABELS)
    assert graph.count == 0
    assert not graph
    assert list(graph.sequences()) == []

    # tau = 0.5 -> keep labels with p >= 0.5: all labels allowed
    graph = SequenceCandidateGraph(probs, 0.5, machine, labels=LABELS)
    assert graph.count == 1

    # An infinite threshold keeps every label.
    graph = SequenceCandidateGraph(probs, np.inf, machine, labels=LABELS)
    assert graph.count == 1


def test_graph_empty_candidate_set():
    machine = make_machine()
    # Never enough mass on OPEN at the first position.
    probs = np.array(
        [
            [0.1, 0.45, 0.45],
            [0.1, 0.45, 0.45],
            [0.1, 0.45, 0.45],
        ]
    )
    graph = SequenceCandidateGraph(probs, 0.5, machine, labels=LABELS)
    assert graph.is_empty
    assert graph.count == 0
    assert len(graph) == 0
    assert not graph
    assert graph.candidates.count == 0
    assert list(graph.candidates) == []


def test_graph_accepting_states_filter_complete_sequences():
    # Same transitions, but no accepting state: no candidate at all.
    machine = StateMachine(
        transitions=[("s0", "A", "s1")],
        initial_state="s0",
        accepting_states=[],
    )
    graph = SequenceCandidateGraph(
        np.array([[0.9, 0.1]]), 0.5, machine, labels=["A", "B"]
    )
    assert graph.count == 0


def test_graph_counts_beyond_enumeration_without_listing():
    # 20 positions over 2 labels gives 2**20 candidates if all are allowed.
    machine = StateMachine(
        transitions=[
            ("s0", "A", "s0"),
            ("s0", "B", "s0"),
        ],
        initial_state="s0",
        accepting_states=["s0"],
    )
    probs = np.full((20, 2), 0.5)
    graph = SequenceCandidateGraph(probs, 0.5, machine, labels=["A", "B"])
    assert graph.count == 2**20
    assert graph.contains(("A",) * 20)
    assert not graph.contains(("A",) * 19)

    # Lazy enumeration only materializes what is consumed.
    iterator = graph.sequences()
    first = next(iterator)
    assert first == ("A",) * 20


def test_graph_candidates_view():
    machine = make_machine()
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    graph = SequenceCandidateGraph(probs, 0.5, machine, labels=LABELS)
    candidates = graph.candidates

    assert isinstance(candidates, SequenceCandidateSet)
    assert candidates.count == graph.count
    assert candidates.count_candidates() == 1
    assert candidates.length == 3
    assert candidates.threshold == 0.5
    assert ("OPEN", "PAID", "CLOSED") in candidates
    assert list(candidates.sequences()) == [("OPEN", "PAID", "CLOSED")]


def test_graph_zero_length_sequence_depends_on_accepting_initial_state():
    # An empty sequence is a legal candidate iff the initial state accepts.
    accepting_initial = StateMachine(
        transitions=[("s0", "A", "s1")],
        initial_state="s0",
        accepting_states=["s0"],
    )
    graph = SequenceCandidateGraph(
        np.zeros((0, 2)), 0.5, accepting_initial, labels=["A", "B"]
    )
    assert graph.count == 1
    assert list(graph.sequences()) == [()]
    assert graph.contains(())

    non_accepting_initial = StateMachine(
        transitions=[("s0", "A", "s1")],
        initial_state="s0",
        accepting_states=["s1"],
    )
    graph = SequenceCandidateGraph(
        np.zeros((0, 2)), 0.5, non_accepting_initial, labels=["A", "B"]
    )
    assert graph.count == 0
    assert list(graph.sequences()) == []


def test_graph_requires_state_machine():
    with pytest.raises(TypeError):
        SequenceCandidateGraph(np.full((2, 2), 0.5), 0.5, None)


def test_graph_rejects_invalid_threshold():
    machine = make_machine()
    probs = np.full((2, 3), 1 / 3)
    with pytest.raises(TypeError):
        SequenceCandidateGraph(probs, None, machine, labels=LABELS)
    with pytest.raises(ValueError):
        SequenceCandidateGraph(probs, float("nan"), machine, labels=LABELS)
    with pytest.raises(ValueError):
        SequenceCandidateGraph(probs, -np.inf, machine, labels=LABELS)


def test_graph_rejects_machine_labels_missing_from_mapping():
    machine = make_machine()
    with pytest.raises(ValueError):
        SequenceCandidateGraph(
            np.full((3, 2), 0.5), 0.5, machine, labels=["OPEN", "PAID"]
        )


def test_graph_rejects_non_normalized_probabilities():
    machine = make_machine()
    with pytest.raises(ValueError):
        SequenceCandidateGraph(
            np.array([[0.9, 0.9, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            0.5,
            machine,
            labels=LABELS,
        )


def test_graph_rejects_unknown_state_in_allowed_labels():
    machine = make_machine()
    graph = SequenceCandidateGraph(
        np.full((3, 3), 1 / 3), 0.5, machine, labels=LABELS
    )
    with pytest.raises(IndexError):
        graph.allowed_labels(10)
    with pytest.raises(ValueError):
        graph.allowed_labels(0, state="unknown")


def test_graph_allowed_labels_is_not_a_sequence_representation():
    machine = make_machine()
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    graph = SequenceCandidateGraph(probs, 0.5, machine, labels=LABELS)
    assert graph.allowed_labels(0) == ("OPEN",)
    assert graph.allowed_labels(0, state="start") == ("OPEN",)
    # Position-wise view alone would suggest 1 * 1 * 1 = 1; the sequence
    # count agrees here, but the API distinguishes both concepts.
    assert graph.count == 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prediction sets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_sequence_lac_set_returns_one_tuple_of_graphs():
    machine = make_machine()
    probs = [np.full((3, 3), 1 / 3) for _ in range(2)]
    result = sequence_lac_set(probs, 0.5, state_machine=machine, labels=LABELS)

    assert isinstance(result, tuple)
    assert len(result) == 1
    graphs = result[0]
    assert len(graphs) == 2
    assert all(isinstance(g, SequenceCandidateGraph) for g in graphs)


def test_sequence_lac_set_requires_state_machine():
    with pytest.raises(TypeError):
        sequence_lac_set(np.full((3, 3), 1 / 3), 0.5)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calibrator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _calibration_data(n=30, length=3, seed=0, p_true=0.9):
    """Confident predictions on the legal trajectory OPEN -> PAID -> CLOSED."""
    rng = np.random.RandomState(seed)
    probs = []
    for _ in range(n):
        matrix = np.empty((length, 3))
        for t in range(length):
            remaining = rng.random(2)
            remaining = (1 - p_true) * remaining / remaining.sum()
            row = np.empty(3)
            row[t] = p_true
            row[(t + 1) % 3] = remaining[0]
            row[(t + 2) % 3] = remaining[1]
            matrix[t] = row
        probs.append(matrix)
    truth = [["OPEN", "PAID", "CLOSED"] for _ in range(n)]
    return probs, truth


def test_calibrator_matches_native_quantile():
    probs, truth = _calibration_data()
    calibrator = SequenceLACCalibrator(make_machine(), labels=LABELS)
    calibrator.fit(y_true=truth, y_pred=probs)

    scores = calibrator.get_nonconformity_scores()
    np.testing.assert_allclose(
        scores, sequence_lac_score(probs, truth, labels=LABELS)
    )

    alpha = 0.1
    threshold = calibrator.get_threshold(alpha=alpha)

    reference = BaseCalibrator(
        nonconf_score_func=lambda yp, yt: yp,
        pred_set_func=lambda *a, **k: None,
    )
    reference._residuals = scores  # pylint: disable=protected-access
    reference._len_calib = len(scores)  # pylint: disable=protected-access
    expected = float(reference.compute_quantile(alpha=alpha))
    assert threshold == pytest.approx(expected)


def test_calibrator_calibrate_builds_graphs():
    probs, truth = _calibration_data()
    calibrator = SequenceLACCalibrator(make_machine(), labels=LABELS)
    calibrator.fit(y_true=truth, y_pred=probs)

    graphs, = calibrator.calibrate(alpha=0.1, y_pred=probs[:4])
    assert len(graphs) == 4
    for graph in graphs:
        assert graph.contains(("OPEN", "PAID", "CLOSED"))


def test_calibrator_rejects_calibration_sequences_rejected_by_machine():
    probs, _ = _calibration_data(n=5)
    # PAID before OPEN is not a legal trajectory.
    truth = [["PAID", "OPEN", "CLOSED"] for _ in range(5)]
    calibrator = SequenceLACCalibrator(make_machine(), labels=LABELS)
    with pytest.raises(ValueError):
        calibrator.fit(y_true=truth, y_pred=probs)


def test_calibrator_requires_state_machine():
    probs, truth = _calibration_data()
    calibrator = SequenceLACCalibrator(labels=LABELS)
    with pytest.raises(TypeError):
        calibrator.fit(y_true=truth, y_pred=probs)


def test_calibrator_rejects_too_few_calibration_samples():
    probs, truth = _calibration_data(n=5)
    calibrator = SequenceLACCalibrator(make_machine(), labels=LABELS)
    calibrator.fit(y_true=truth, y_pred=probs)
    with pytest.raises(ValueError):
        calibrator.compute_quantile(alpha=0.05)


def test_calibrator_calibrate_before_fit():
    calibrator = SequenceLACCalibrator(make_machine(), labels=LABELS)
    probs, _ = _calibration_data(n=2)
    with pytest.raises(RuntimeError):
        calibrator.calibrate(alpha=0.5, y_pred=probs)


def test_calibrator_mapping_must_stay_consistent():
    probs, truth = _calibration_data()
    calibrator = SequenceLACCalibrator(make_machine(), labels=LABELS)
    calibrator.fit(y_true=truth, y_pred=probs)
    with pytest.raises(ValueError):
        calibrator.calibrate(
            alpha=0.5,
            y_pred=probs[:2],
            labels=["CLOSED", "PAID", "OPEN"],
        )


def test_calibrator_accepts_state_machine_at_fit_time():
    probs, truth = _calibration_data()
    calibrator = SequenceLACCalibrator(labels=LABELS)
    calibrator.fit(
        y_true=truth, y_pred=probs, state_machine=make_machine()
    )
    assert calibrator.compute_quantile(alpha=0.5) >= 0


def test_calibrator_interface_compatible_with_split_conformal_predictor():
    # SplitConformalPredictor builds its calibrator with these keywords.
    calibrator = SequenceLACCalibrator(
        make_machine(),
        labels=LABELS,
        nonconf_score_func=sequence_lac_score,
        pred_set_func=sequence_lac_set,
        weight_func=None,
    )
    probs, truth = _calibration_data()
    calibrator.fit(y_true=truth, y_pred=probs, X=None)
    graphs, = calibrator.calibrate(alpha=0.5, y_pred=probs[:2], X=None)
    assert len(graphs) == 2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward compatibility ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_existing_lac_api_is_unchanged():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    truth = np.array([0, 2])
    scores = lac_score(probs, truth)
    np.testing.assert_allclose(scores, np.array([0.3, 0.7]))

    sets, = lac_set(probs, 0.7)
    assert sets == [[0], [1]]

    calibrator = BaseCalibrator(
        nonconf_score_func=lac_score,
        pred_set_func=lac_set,
    )
    calibrator.fit(y_pred=probs, y_true=truth)
    calib_sets, = calibrator.calibrate(alpha=0.5, y_pred=probs)
    assert calib_sets == [[0], [1]]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ High-level wrapper ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_sequence_lac_wrapper_uses_trained_model_as_is():
    from deel.puncc.classification import SequenceLAC

    class FixedPredictor:
        """Predictor that emits pre-computed probabilities, never trained."""

        def __init__(self, probs):
            self.probs = probs
            self.fit_calls = 0

        def fit(self, *args, **kwargs):
            self.fit_calls += 1

        def predict(self, X, **kwargs):
            return [self.probs[i] for i in X]

    probs, truth = _calibration_data(n=30)
    predictor = FixedPredictor(probs)

    cp = SequenceLAC(predictor, make_machine(), labels=LABELS)
    cp.fit(X_calib=list(range(30)), y_calib=truth)

    assert predictor.fit_calls == 0
    _, graphs = cp.predict(X=list(range(3)), alpha=0.1)
    assert len(graphs) == 3
    assert all(graph.contains(("OPEN", "PAID", "CLOSED")) for graph in graphs)
    assert cp.predictor is predictor
