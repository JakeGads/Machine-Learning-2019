"""
Dataset: pda_data_no_dups.csv
Dataset Description: This is a protein data set retrieved from Research Collaboratory for Structural Bioinformatics
(RCSB) Protein Data Bank (PDB). The PDB archive is a repository of atomic coordinates and other information
describing proteins and other important biological macromolecules. Structural biologists use methods such as X-ray
crystallography, NMR spectroscopy, and cryo-electron microscopy to determine the location of each atom relative to
each other in the molecule. They then deposit this information, which is then annotated and publicly released into the
archive by the wwPDB. The constantly growing PDB reflects the research that is happening in laboratories across the
world. This can make it both exciting and challenging to use the database in research and education.

Problem description: Which of these columns best predicts the “classification” of a protein: macromoleculeType,
residueCount, resolution, structureMolecularWeight, densityMatthews, densityPercentSol, phValue. Be mindful of
overfitting and underfitting.
"""