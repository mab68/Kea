Statistic Functions
-------------------

Currently **KEA** offers the following lag statistics functions:

* Correlation function
* Structure function

Both of these functions rely on the same backend calculation of generating "lag-shifts" of a given array. 

Correlation Functions
^^^^^^^^^^^^^^^^^^^^^^

The autocorrelation function is estimated via:

.. math::
   R[\mathbf{m}] = \frac{1}{\widetilde{N}[\mathbf{m}]} \sum_\mathbf{n} y[\mathbf{n}]  y[\mathbf{n} + \mathbf{m}]

where :math:`\widetilde{N}[\mathbf{m}]` is a normalization by the number of pixels, and :math:`\mathbf{m}` is a lag-distance in real-space pixels, and the summation is over the *available* data (i.e., we ignore NaN values).

There are typically two standard ways of normalizing :math:`R[\mathbf{m}]`, either: as a *biased* normalization that is constant and dependent on the total data (:math:`\widetilde{N}[\mathbf{m}] = N`), or "*unbiased*" and changes depending on the number of pairs of data-points that are summed over (:math:`\widetilde{N}[\mathbf{m}] = N - m - 1`).

Structure Functions
^^^^^^^^^^^^^^^^^^^

The structure function follows the same procedures as the ACF, except as:

.. math::
   S^{(p)}[\mathbf{m}] = \frac{1}{\widetilde{N}[\mathbf{m}]} \sum_\mathbf{n} | y[\mathbf{n}] - y[\mathbf{n} + \mathbf{m}] |^{p}

with :math:`\widetilde{N}[\mathbf{m}] = N - m - 1`.

Module contents
^^^^^^^^^^^^^^^

.. automodule:: kea.statistics.statfunc
   :members:
   :show-inheritance:
   :undoc-members:


