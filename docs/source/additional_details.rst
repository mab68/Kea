Additional Details
------------------

Here we detail some additional technical details.


Normalization Consistency
^^^^^^^^^^^^^^^^^^^^^^^^^

Our elected Fourier transform convention utilises the :math:`2\pi` normalization on the forwards transform and that wavevector :math:`\mathbf{k}` contains the :math:`2\pi` factor. In other words, :math:`\mathbf{k}` is the angular wavevector and its relation to the linear wavevector :math:`\mathbf{q}` and linear wavelength :math:`\lambda` is given by:

.. math::
    \mathbf{k} = 2\pi \mathbf{q}, \quad |\mathbf{k}| = \frac{2\pi}{\lambda}.

Of course, this convention is not the only one that is frequently used and different fields may have a preference for technical or theoretical reasons. More generally, the Fourier transform pair may be defined using two arbitary constants :math:`a` and :math:`b` as

.. math::
    \widehat{\psi}_{(a,b)}(\mathbf{\kappa}_{b}) = \left( \frac{|b|}{\left(2\pi \right)^{1-a}} \right)^{D/2} \int \psi_{(a,b)}(\mathbf{\xi}) e^{i b \mathbf{\kappa}_{b} \cdot \mathbf{\xi}} \mathrm{d}^{D} \mathbf{\xi},

.. math::
    \psi_{(a,b)}(\mathbf{\xi}) = \left( \frac{|b|}{\left(2\pi \right)^{1+a}} \right)^{D/2} \int \widehat{\psi}_{(a,b)}(\mathbf{\kappa}_{b}) e^{-i b \mathbf{\kappa}_{b} \cdot \mathbf{\xi}} \mathrm{d}^{D} \mathbf{\kappa}_b,

where :math:`\mathbf{\kappa}_b` can represent a linear wavevector (using :math:`b=- 2\pi`; :math:`\mathbf{\kappa}_{-2\pi} \equiv \mathbf{q}`), or an angular wavevector (using :math:`b=- 1`; :math:`\mathbf{\kappa}_{-1} \equiv \mathbf{k}`). Common alternative conventions that are relevant for this thesis, which we loosely classify as associated with the applied mathematics, cosmology, ISM, ICM, and turbulence literature are


.. math::
    \text{Applied Mathematics}: (0, -1) = \begin{cases}
        \widehat{A}(\mathbf{k}) = \left( 2\pi \right)^{-D/2} \int A(\mathbf{x}) e^{-i \mathbf{k} \cdot \mathbf{x}} \mathrm{d}^{D} \mathbf{x},\\
        A(\mathbf{x}) = \left( 2\pi \right)^{-D/2} \int \widehat{A}(\mathbf{k}) e^{i\mathbf{k} \cdot \mathbf{x}} \mathrm{d}^D \mathbf{k},
    \end{cases}

.. math::
    \mathrm{Cosmology}: (1, -1) = \begin{cases}
        \widehat{C}(\mathbf{k}) = \int C(\mathbf{x}) e^{-i \mathbf{k} \cdot \mathbf{x}} \mathrm{d}^{D} \mathbf{x},\\
        C(\mathbf{x}) = \frac{1}{(2\pi)^D} \int \widehat{C}(\mathbf{k}) e^{i \mathbf{k} \cdot \mathbf{x}} \mathrm{d}^{D} \mathbf{k},
    \end{cases}

.. math::
    \mathrm{ISM/ICM}: (0, -2\pi) = \begin{cases}
        \widehat{I}(\mathbf{q}) = \int I(\mathbf{x}) e^{-2 \pi i \mathbf{q} \cdot \mathbf{x}} \mathrm{d}^{D} \mathbf{x},\\
        I(\mathbf{x}) = \int \widehat{I}(\mathbf{q}) e^{2\pi i \mathbf{q} \cdot \mathbf{x}} \mathrm{d}^{D} \mathbf{q},
    \end{cases}

.. math::
    \mathrm{Turbulence}: (-1, -1) = \begin{cases}
        \widehat{T}(\mathbf{k}) = \frac{1}{(2\pi)^{D}} \int T(\mathbf{x}) e^{-i \mathbf{k} \cdot \mathbf{x}} \mathrm{d}^{D}\mathbf{x},\\
        T(\mathbf{x}) = \int \widehat{T}(\mathbf{k}) e^{i \mathbf{k} \cdot \mathbf{x}} \mathrm{d}^{D}\mathbf{k},
    \end{cases}

where our elected convention is consistent with turbulence literature (with :math:`a=-1` and :math:`b=-1`).

With the integral representation of the Dirac delta function

.. math::
    \int_{-\infty}^{\infty} e^{i b \mathbf{\kappa}_{b} \cdot \mathbf{\xi}} \mathrm{d}^{D} \mathbf{\xi} = \left( \frac{2\pi}{|b|} \right)^{D} \delta \left( \mathbf{\kappa}_{b} \right),

the modal spectrum is

.. math::
    \left< \widehat{\psi}_{(a,b)}(\mathbf{\kappa}_{b}) \widehat{\psi}_{(a,b)}^{\ast}(\mathbf{\kappa}_{b}^\prime) \right> = \left( \frac{\left( 2\pi \right)^{1+a}}{|b|} \right)^{D/2} \delta\left(\mathbf{\kappa}_{b} - \mathbf{\kappa}_{b}^\prime \right) E_{(a,b)}(\mathbf{\kappa}_{b}),

which describes the total energy, :math:`E`, via

.. math::
    E = \left(\frac{|b|}{\left( 2\pi \right)^{1+a}} \right)^{D/2} \int E_{(a,b)}(\mathbf{\kappa}_{b}) \mathrm{d}^{D} \mathbf{\kappa}_{b}.

The ISM/ICM and applied mathematics conventions are symmetric and unitary operators that are automatically length preserving, making them desirable for formal proofs and signal processing. However, the applied mathematics convention introduces non-unity pre-factors when computing the total energy from the spectrum (\autoref{eqn:general_convention_spectrum}). While the ISM/ICM convention resolves this energy scaling issue and yields the clean integral

.. math::
    E = \int E_{(0,-2\pi)}(\mathbf{\kappa}_{-2\pi}) \mathrm{d}^{D} \mathbf{\kappa}_{-2\pi},

it does so utilizing the linear wavevector :math:`\mathbf{\kappa}_{-2\pi}`. In differential equations, linear wavevectors introduce explicit scaling factors into spatial derivatives:

.. math::
    \nabla \rightarrow - i b \mathbf{\kappa}_{b} \quad \implies \quad \nabla \xrightarrow{b = -2\pi} 2\pi i \, \mathbf{\kappa}_{-2\pi}.

Furthermore, whether a linear wavevector or an angular wavevector provides a more physically faithful representation of characteristic scales in a turbulent field is an open and nuanced debate.

Alternatively, the turbulence convention also has fewer mathematical pre-factors when considering the energy spectrum and physical fields themselves, but is not a symmetric operation. By placing the :math:`(2\pi)^{-D}` normalization entirely on the forward transform, the inverse transform remains free of scaling constants. Consequently, integrating the spectral energy density over all wavevectors directly yields the total physical energy without extra geometric factors,i.e.,

.. math::
    E = \int E_{(-1,-1)}(\mathbf{\kappa}_{-1}) \mathrm{d}^{D} \mathbf{\kappa}_{-1}.

This allows for a more intuitive physical interpretation of the spectrum as the exact energy per unit (angular) wavenumber.

Lastly, the cosmology convention avoids pre-factors in differential equations, while its zero-frequency mode (:math:`\mathbf{k} = 0`) directly yields the total spatial integral of the field:

.. math::
    \widehat{C}(\mathbf{\kappa}_{-1}=0) = \int C(\mathbf{x}) \mathrm{d}^{D} \mathbf{x}.

Furthermore, its power spectrum is defined via the unweighted forward transform of the two-point correlation function (though the covariance retains a non-unity pre-factor).

Ultimately, because these distinct conventions define energy and wavevector scalings differently, they produce non-identical spectral representations. No single convention is objectively superior; rather, each subfield adopts the convention that offers the most convenient mathematical framework for its primary problems of interest.

In this thesis, we consider and often compare these conventions from the branches of literature so it is necessary to transform between them for consistent representation of energy. To transform from convention :math:`(a,b)` to :math:`(a^\prime, b^\prime)`, the frequency variable must rescale as

.. math::
    \mathbf{\kappa}_{b} = \frac{b^\prime}{b} \mathbf{\kappa}_{b^\prime},

and the Fourier transformed functions as

.. math::
        \widehat{\psi}_{(a,b)}(\mathbf{\kappa})|_{\mathbf{\kappa}_{b} = b^\prime \mathbf{\kappa}_{b^\prime} / b} = \left(\frac{|b^\prime|}{|b|} \left(2\pi\right)^{a^\prime - a} \right)^{-D/2} \widehat{\psi}_{(a^\prime,b^\prime)}(\mathbf{\kappa}_{b^\prime}),

.. math::
        E_{(a,b)}(\mathbf{\kappa}_{b})|_{\mathbf{\kappa}_b = b^\prime \mathbf{\kappa}_{b^\prime}/b} = \left(\frac{|b^\prime|}{|b|} \left(2\pi\right)^{a^\prime - a}\right)^{-D/2} E_{(a^\prime,b^\prime)}(\mathbf{\kappa}_{b^\prime}),

.. math::
        \mathcal{E}_{(a,b)}(\kappa_{b})|_{\kappa_b = b^\prime \kappa_{b^\prime}/b} = \left(\frac{b^\prime}{b}\right)^{D-1} \left(\frac{|b^\prime|}{|b|} \left(2\pi\right)^{a^\prime - a}\right)^{-D/2} \mathcal{E}_{(a^\prime,b^\prime)}(\kappa_{b^\prime}),

.. math::
        \mathcal{A}_{(a,b)}(\kappa_{b})|_{\kappa_b = b^\prime \kappa_{b^\prime}/b} = \left(\frac{b^\prime}{b}\right)^{D/2} \left(\frac{|b^\prime|}{|b|} \left(2\pi\right)^{a^\prime - a}\right)^{-D/4} \mathcal{A}_{(a^\prime,b^\prime)}(\kappa_{b^\prime}).

Thus, the :math:`T`, :math:`I`, and :math:`C` conventions translate as:

.. math::
    E^{II}(\mathbf{q}) = \left(2\pi\right)^{D} E^{TT}(\mathbf{k})\big|_{\mathbf{k}=2\pi\mathbf{q}} = E^{CC}(\mathbf{k})\big|_{\mathbf{k} = 2\pi\mathbf{q}},

.. math::
    \mathcal{E}^{II}(q) = 2\pi\mathcal{E}^{TT}(k)\big|_{k=2\pi q} = \left(2\pi\right)^{1-D} \mathcal{E}^{CC}(k) \big|_{k = 2\pi q},

.. math::
    \mathcal{A}^{II}(q) = \mathcal{A}^{TT}(k)\big|_{k= 2\pi q} = \left(2\pi\right)^{-D/2} \mathcal{A}^{CC}(k)\big|_{k= 2\pi q}.

Note that there is no difference in the amplitude spectra for the :math:`T` and :math:`I` conventions, only a shift associated with linear wavenumbers, :math:`q`, and angular wavenumbers, :math:`k`.

Binning
^^^^^^^

Since the autocorrelation function (ACF), structure function (SF), and modal spectral (FFT and BT) estimates provide :math:`D`-dimensional functions, it is useful to bin them to reduce the dimensionality. `Kea` provides a general binning routine that applies to the spectral estimates as well as the lag-functions.

With bin-width :math:`\Delta b = x_+ - x_-` and :math:`b \in \mathscr{R}`,

.. math::
   :label: eqn:binning

   F[b] = \widetilde{\sum}_{x_- \le \left| \mathbf{n} \Delta x \right| < x_+ } f[\mathbf{n}],

where the following options are provided as parameters for the routine:

- Provide a binning function :math:`\widetilde{\sum}` which is typically e.g., an average, or a summation, or a standard error to describe the variation within the bin.
- Whether to ignore the range of values that are outside the circular or spherical shell (since the data is represented on a square/cubic grid).
- Defining specific minimum and maximum :math:`b`: :math:`b_\mathrm{min}, b_\mathrm{max}`.
- The location of the number that represents the bin e.g., the center of the bin (:math:`b = \left( x_+ + x_- \right)/2`), or the left most value (:math:`b = x_-`).
- Whether to apply a normalization of the bin width: :math:`\Delta b^{-1}`.
- Bin with log-spaced widths. In which case, the bins are uniformly spaced in log-space.
- Specify a fixed specific number of bins, :math:`N_\mathrm{bins}`.


Following the continuous definitions the ACF and SF are typically represented via averaging circular/spherical shells\footnote{In the case of 1D data, there are exactly 2 values for each :math:`\left| \mathbf{k} \right|`: the positive and negative value.}. Whereas, the angle-integrated (:math:`\mathcal{E}[b]`) and angle-averaged (:math:`\overline{E}[b]`) spectra are found when :math:`\widetilde{\sum}` in :eq:`eqn:binning` is the summation and average, respectively:

.. math::
    \mathcal{E}[b] = \frac{1}{\Delta b} \sum_{k_- \le \left| \mathbf{k} \right| < k_+ } E[\mathbf{m}] \left( \Delta k \right)^D,

.. math::
    \overline{E}[b] = \frac{1}{K_{D}[b]} \sum_{k_- \le \left| \mathbf{k} \right| < k_+ } E[\mathbf{m}] \left( \Delta k \right)^{D}.

where :math:`K_{D}[b]` is the number of pixels within the bin-range (thus defining the function that averages) and we apply an additional normalization by the bin-width :math:`\Delta b^{-1}` (which ensures unit consistency and that :math:`\sum \mathcal{E}[b] \Delta b` is the total energy). Their relation is found via: :math:`\mathcal{E}[b] = K_{D}[b] \overline{E}[b] / \Delta b` (assuming the remaining binning properties are kept the same). The bin-shell volume coefficients for centered (:math:`b_1 = b-\Delta b / 2`, :math:`b_2 = b+\Delta b/2`), or left-edge bins (:math:`b_1 = b`, :math:`b_2 = b+\Delta b`) are:

.. math::
    K_D[b] = \frac{V(b_1) - V(b_2)}{\left( \Delta k \right)^{D}}

where

.. math::
    V(r) = \frac{r^{D} \pi^{D/2}}{\Gamma\left( D/2 + 1 \right)}

is the volume of a :math:`D`-dimensional hypersphere.

