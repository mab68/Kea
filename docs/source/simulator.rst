Simulations
-----------

One of the primary goals of **kea** is to provide tools to test spectral estimates in various conditions. Thus, we generate synthetic fields with known functional forms of the power spectrum. Therefore, we can test whether the power spectrum method is obtaining the correct form. **kea** also provides methods to generate random gaps in the data which can be used in conjunction with the synthetic fields to test spectral estimation performance on gapped data (which is common in astrophysical observations).

Generally, we can synthesize a field from the following description in Fourier-space:

.. math::
   \widehat{f}(\boldsymbol{k}) = \sqrt{E(\boldsymbol{k})} \delta(\boldsymbol{k})

where, e.g., for a Gaussian field:

.. math::
   \delta(\boldsymbol{k}) = a(\boldsymbol{k}) + i b(\boldsymbol{k})

with :math:`a(\boldsymbol{k}),b(\boldsymbol{k}) \sim \mathcal{N}(0,1)` independently samples from a Gaussian distribution for each :math:`\boldsymbol{k}`. The Fourier transform of :math:`\widehat{f}(\boldsymbol{k})` results in our simulated fluctuation field with :math:`E(\boldsymbol{k})` typically describing the correlations. 

Module contents
^^^^^^^^^^^^^^^

.. automodule:: kea.simulator
   :members:
   :show-inheritance:
   :undoc-members:
