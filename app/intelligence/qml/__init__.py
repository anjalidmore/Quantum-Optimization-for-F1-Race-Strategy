"""
Quantum machine learning (PennyLane) on the Task 5 feature matrix.

Three models, all on the ``default.qubit`` noiseless simulator:

* ``vqc`` — variational quantum classifier for the pit decision
* ``vqr`` — variational quantum regressor for lap time
* ``kernel_svm`` — fidelity-kernel SVM for the pit decision

``data`` prepares circuit inputs using Task 6's own splits, ``baselines`` adds
the fair classical comparisons, and ``pipeline.run_all()`` runs everything and
writes the artifacts. No quantum hardware is involved and no quantum advantage
is claimed anywhere in this package.
"""
