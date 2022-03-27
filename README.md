This is a repository for using semi-supervised learning to classify neutron diffraction data.  The work is described in a paper currently under review

Files included:
* Data
  * BravaisLattice_Data.pt: A binary file that contains 10 random simulated powder diffraction patterns from the testing dataset. Each piece of data is labeled with the correct Bravais Lattice.
  * SpaceGroup_Data.pt: A binary file that contains 10 random simulated powder diffraction patterns from the testing dataset. Each piece of data is labeled with the correct Space Group.
  * mapping.pt: A binary file that is used to convert categorical labels to text.
  * Data.py: Defines the DiffractionDataset class, a Pytorch Dataset that can use the BravaisLatticeData.pt and the SpaceGroupData.pt files.
  * ICSD_IDs.txt: A text file containing the ICSD IDs used in this work.
* Models
  * Generator.py: Defines the Generator model used in the Semi-Supervised Generative Adversarial Network (SGAN).
  * ResNet.py: Defines the ResNet classifier that is used as a supervised classifier and as the Discriminator of the SGAN.
  * BravaisModels.pth: Supervised and Semi-supervised Bravais Lattice classifiers
  * SpaceGroupModels.pth: Supervised and Semi-supervised space group classifiers
* Notebooks
  * PlotData.ipynb: Loads data from the BravaisLattice_Data.pt and SpaceGroup_Data.pt files as a 2&theta; vs. normalized intensity graph.
  * LoadBravaisLatticeModels.ipynb: Evaluates the accuracy of the supervised and semi-supervised Bravais Lattice models.
  * LoadSpaceGroupModels.ipynb: Evaluates the accuracy of the supervised and semi-supervised space group models. This notebook also includes a comparison of the top-5 accuracies of each model.