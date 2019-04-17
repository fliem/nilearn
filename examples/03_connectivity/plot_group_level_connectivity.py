"""
Functional connectivity matrices for group analysis of connectomes
==================================================================

Functional connectomes share a lot of common structure across people,
similarly to genetics, where the majority of  DNA is similar in all humans,
and individual differences are encoded in a small fraction of DNA.
Some *population-regularized* functional connectivity estimators take these
commonalities into
account by informing the connectivity estimation of a single subject with
connectivity data from a group of subjects. Since functional connectivity
is extracted from noisy data, population-regularized connectivity estimation
is more robust and enables more accurate predictions
based on connectivity.


First, this example extracts functional connectivity with different
estimators and compares the resulting connectivity matrices. Some estimators
do not regard group-level information (correlation, partial correlation),
others estimated population-regularized connectivity
(**tangent**, and **population shrinkage of covariance estimator
(PoSCE)**).

Second, we use the resulting connectivity coefficients to
discriminate children from adults. In general, the population-regularized
estimators  **tangent and PoSCE outperform** the standard correlations:
see `Dadi et al. 2019
<https://www.sciencedirect.com/science/article/pii/S1053811919301594>`_
and
`Rahim et al. 2019
<https://hal.inria.fr/hal-02068389>`_
for two systematic studies on multiple large samples.
"""

###############################################################################
# Load data and extract signal
# ----------------------------
# We will be working with the brain development fMRI dataset and only use 30
# subjects, to save computation time.
from nilearn import datasets

rest_data = datasets.fetch_development_fmri(n_subjects=30)

###############################################################################
# We use probabilistic regions of interest (ROIs) from the MSDL atlas.
msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords
n_regions = len(msdl_coords)
print('MSDL has {0} ROIs, part of the following networks :\n{1}.'.format(
    n_regions, msdl_data.networks))

###############################################################################
# Region signals extraction
# -------------------------
# To extract regions time series, we instantiate a
# :class:`nilearn.input_data.NiftiMapsMasker` object and pass it the file name
# of the atlas, as well as filtering band-width and detrending option.
from nilearn import input_data

masker = input_data.NiftiMapsMasker(
    msdl_data.maps, resampling_target="data", t_r=2.5, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1)

###############################################################################
# Then we compute region signals and extract useful phenotypic information.
children = []
pooled_subjects = []
groups = []  # child or adult
for func_file, confound_file, phenotypic in zip(
        rest_data.func, rest_data.confounds, rest_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    is_child = phenotypic['Child_Adult'] == 'child'
    if is_child:
        children.append(time_series)

    groups.append(phenotypic['Child_Adult'])

print('Data has {0} children.'.format(len(children)))



###############################################################################
# Different ways to represent connectivity
# ----------------------------------------
# First, we create a helper function to plot matrices
import numpy as np
import matplotlib.pylab as plt


def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        # Set diagonal to zero, for better visualization
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{0}, subject {1}'.format(matrix_kind, n_subject)
        plotting.plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                             title=title, figure=fig, colorbar=False)



###############################################################################
# ROI-to-ROI correlations of children
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The simplest and most commonly used kind of connectivity is correlation. It
# models the full (marginal) connectivity between pairwise ROIs. We can
# estimate it using :class:`nilearn.connectome.ConnectivityMeasure`.
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')

###############################################################################
# From the list of ROIs time-series for children, the
# `correlation_measure` computes individual correlation matrices.
correlation_matrices = correlation_measure.fit_transform(children)

###############################################################################
# The individual coefficients are stored in 2D matrices and are stacked into a
# 3D array.
print('Correlations of children are stacked in an array of shape {0}'
      .format(correlation_matrices.shape))

###############################################################################
# We also can see the average correlation across all fitted subjects.
mean_correlation_matrix = correlation_measure.mean_
print('Mean correlation has shape {0}.'.format(mean_correlation_matrix.shape))

###############################################################################
# We display the connectome matrices of the first 4 children.
from nilearn import plotting

plot_matrices(correlation_matrices[:4], 'correlation')
###############################################################################
# Function networks can be seen as blocks of connectivity.

###############################################################################
# Now we display the mean correlation matrix over all children as a connectome.
plotting.plot_connectome(mean_correlation_matrix, msdl_coords,
                         title='mean correlation over all children')

###############################################################################
# Studying partial correlations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can also study **direct connections**, revealed by partial correlation
# coefficients. We just change the `ConnectivityMeasure` kind
partial_correlation_measure = ConnectivityMeasure(kind='partial correlation')

###############################################################################
# and repeat the previous operation.
partial_correlation_matrices = partial_correlation_measure.fit_transform(
    children)

###############################################################################
# Most of direct connections are weaker than full connections:
plot_matrices(partial_correlation_matrices[:4], 'partial')

###############################################################################
# Compared to a connectome computed on correlations, the connectome graph
# with partial correlations is sparser:
plotting.plot_connectome(
    partial_correlation_measure.mean_, msdl_coords,
    title='mean partial correlation over all children')

###############################################################################
# Extract connectivity with tangent embedding
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can use **both** correlations and partial correlations to capture
# reproducible connectivity patterns at the group-level and build a **robust**
# **group connectivity matrix**. This is done by the **tangent** kind.
tangent_measure = ConnectivityMeasure(kind='tangent')

###############################################################################
# We fit our children group and get the group connectivity matrix stored as
# in `tangent_measure.mean_`, and individual deviation matrices of each subject
# from it.
tangent_matrices = tangent_measure.fit_transform(children)

###############################################################################
# `tangent_matrices` model individual connectivities as
# **perturbations** of the group connectivity matrix `tangent_measure.mean_` .
# Keep in mind that these subjects-to-group variability matrices do not
# straightforwardly reflect individual brain connections. For instance negative
# coefficients can not be interpreted as anticorrelated regions.
plot_matrices(tangent_matrices[:4], 'tangent variability')

###############################################################################
# The average tangent matrix cannot be interpreted, as the average
# variation is expected to be zero

###############################################################################
# Extract connectivity via PoSCE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Next, we use the population shrinkage of covariance estimator
# :class:`nilearn.connectome.PopulationShrunkCovariance` .
# It uses the correlation of a group to better estimate connectivity of a
# single subject.
from nilearn.connectome import PopulationShrunkCovariance

posce_measure = PopulationShrunkCovariance(shrinkage=1e-2)
posce_matrices = posce_measure.fit_transform(children)

plot_matrices(posce_matrices[:4], 'PoSCE')

###############################################################################
# Extract biomarkers and use them for  classification?
# --------------------------------------------------------------
# *ConnectivityMeasure* can output the estimated subjects coefficients
# as 1D arrays through the parameter *vectorize*.
connectivity_biomarkers = {}
kinds = ['correlation', 'partial correlation', 'tangent', 'PoSCE']
for kind in kinds:
    if kind == 'PoSCE':
        posce_measure = PopulationShrunkCovariance(shrinkage=1e-2,
                                                  vectorize=True)
        connectivity_biomarkers[kind] = posce_measure.\
            fit_transform(pooled_subjects)
    else:
        conn_measure = ConnectivityMeasure(kind=kind, vectorize=True)
        connectivity_biomarkers[kind] = conn_measure.fit_transform(pooled_subjects)

# For each kind, all individual coefficients are stacked in a 2D
# matrix (subject x connectivity features). This will be the input matrix for
# the classifier.
print('Correlation biomarker features for all subject of shape {0}'.format(
    connectivity_biomarkers['correlation'].shape))

###############################################################################
# Note that we use the **pooled groups** that includes data from children
# and adults.


###############################################################################
# We now aim to predict the group label ('child' or 'adult') from the
# connectivity data using a support vector classifier (SVC). To evaluate
# the models' performance, we use stratified 3-fold-cross-validation,
# which keeps the ratio of children and adults constant in all folds.
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=3)

mean_scores = []
for kind in kinds:
    svc = LinearSVC(random_state=0)
    cv_scores = cross_val_score(svc,
                                connectivity_biomarkers[kind],
                                y=groups,
                                cv=cv,
                                groups=groups,
                                scoring='accuracy',
                                )
    mean_scores.append(cv_scores.mean())

###############################################################################
# Finally, we can display the classification scores.
plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds)) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05)
yticks = [kind.replace(' ', '\n') for kind in kinds]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.grid(True)
plt.tight_layout()

###############################################################################
# While the comparison is not fully conclusive on this small dataset,
# `Dadi et al 2019
# <https://www.sciencedirect.com/science/article/pii/S1053811919301594>`_
# and `Rahim et al. 2019
# <https://hal.inria.fr/hal-02068389>`_ across many cohorts and clinical
# questions, the tangent and PoSCE estimators should be preferred.
