.. _sequence:

🔗 Sequence-level prediction
=============================

Many business processes produce **trajectories**: a sample is not a single
label but a whole label sequence, and only some trajectories are legal
according to a deterministic state machine (order of events, mandatory steps,
allowed branches, etc.).

Deciding each position independently is not enough here. Positions of a given
trajectory belong to the same sample, so their LAC decisions cannot be pooled,
and intersecting position-wise prediction sets would ignore the state-machine
constraints while still describing independent labels rather than complete
trajectories. This module implements a **sequence-level** extension of LAC.

Procedure
---------

1. An already trained model provides, for every position of every sequence,
   the class probabilities. These probabilities are used **as is**: the model
   is neither retrained nor rescaled. They are nevertheless checked with
   puncc's native :func:`deel.puncc.api.utils.logit_normalization_check`, so a
   model that does not output probabilities fails explicitly.
2. For each calibration sequence, the native LAC score
   :math:`1 - p(y_t)` is computed at every position with
   :func:`deel.puncc.api.nonconformity_scores.lac_score`, and the **largest**
   score along the true label path is retained as the unique sequence-level
   score. One sequence is one sample, whatever its length.
3. The native corrected quantile (correction function, calibration-size check,
   ``+inf`` padding and empirical quantile, exactly as
   :class:`deel.puncc.api.calibration.BaseCalibrator`) of those scores gives a
   single threshold :math:`\tau` shared by all positions of all new sequences.
4. For each new sequence, a **shared candidate graph** is built. It represents
   exactly the set of complete label sequences that both follow the
   deterministic state machine (ending in an accepting state) and satisfy the
   native LAC rule :math:`p_t(l_t) \ge 1 - \tau` at every position.

The graph answers whole-sequence membership queries in linear time, returns the
exact number of candidates with a dynamic program (all strings are never
enumerated), and lazily lists the actual candidates only when requested.
Position-wise label sets are exposed as an auxiliary view only: they are *not*
a representation of the sequence set. When no legal sequence exists, the graph
truthfully reports an empty candidate set.

State machine
-------------

.. autoclass:: deel.puncc.api.state_machine.StateMachine

Sequence nonconformity score and prediction sets
------------------------------------------------

.. autofunction:: deel.puncc.api.sequence.sequence_lac_score

.. autofunction:: deel.puncc.api.sequence.sequence_lac_set

Candidate graph
---------------

.. autoclass:: deel.puncc.api.sequence.SequenceCandidateGraph
   :members: count, n_candidates, is_empty, contains, sequences, allowed_labels

.. autoclass:: deel.puncc.api.sequence.SequenceCandidateSet

Calibrator
----------

.. autoclass:: deel.puncc.api.sequence.SequenceLACCalibrator
   :members: fit, compute_quantile, calibrate, get_nonconformity_scores

High-level wrapper
------------------

.. autoclass:: deel.puncc.classification.SequenceLAC

Example
-------

.. code-block:: python

    import numpy as np

    from deel.puncc.api.state_machine import StateMachine
    from deel.puncc.classification import SequenceLAC

    # Legal trajectory: OPEN -> PAID -> CLOSED
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

    # `model.predict(X)` returns, for each sequence of X, an array of shape
    # (n_positions, n_classes) holding the class probabilities.
    sequence_lac = SequenceLAC(model, machine, labels=labels)
    sequence_lac.fit(X_calib=X_calib, y_calib=y_calib)

    y_pred, graphs = sequence_lac.predict(X_test, alpha=0.1)

    graph = graphs[0]
    graph.count                            # exact number of legal candidates
    ("OPEN", "PAID", "CLOSED") in graph    # whole-sequence membership
    list(graph.sequences())                # lazy enumeration, if needed
