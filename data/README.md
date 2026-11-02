# Test sets

The test sets of the ChefNMR2 paper, as used in the study: one JSON Lines file per set, one molecule per line.

## Files

Each file is released under the licence named in its block; data derived from a source also remain subject to the
source's own licence. Please cite the sources of the files you use.

### `nmrexp.jsonl` (500 molecules)

Held-out test set of NMRexp: experimental peak lists parsed from published articles.

- Structures and peak lists: NMRexp, version 2 (https://doi.org/10.5281/zenodo.17296666; CC BY 4.0).
- Licence of this file: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
- Cite: Wang, J.-J. et al. NMRexp: A database of 3.3 million experimental NMR spectra. Scientific Data 12, 1954
  (2025). https://doi.org/10.1038/s41597-025-06245-5

### `spectranp+.jsonl` (500 molecules)

Held-out test set of SpectraNP+: natural products with simulated peak lists.

- Structures: COCONUT, release of October 2024 (https://doi.org/10.5281/zenodo.13897048; CC BY 4.0); NPAtlas,
  release of September 2024 (https://doi.org/10.5281/zenodo.13756408; CC BY-NC 4.0); NP-MRD (https://np-mrd.org;
  CC BY-NC 4.0).
- Peak lists: simulated by the authors with Mnova NMRPredict.
- Licence of this file: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- Cite:
  - the ChefNMR2 preprint ([Citation](../README.md#citation))
  - Chandrasekhar, V. et al. COCONUT 2.0: a comprehensive overhaul and curation of the collection of open natural
    products database. Nucleic Acids Research 53, D634–D643 (2025). https://doi.org/10.1093/nar/gkae1063
  - Poynton, E. F. et al. The Natural Products Atlas 3.0: extending the database of microbially derived natural
    products. Nucleic Acids Research 53, D691–D699 (2025). https://doi.org/10.1093/nar/gkae1093
  - Wishart, D. S. et al. The Natural Products Magnetic Resonance Database (NP-MRD) for 2025. Nucleic Acids Research
    53, D700–D708 (2025). https://doi.org/10.1093/nar/gkae1067

### `uspto.jsonl` (500 molecules)

Held-out test set of USPTO: molecules from US patents with simulated peak lists.

- Structures: Lowe, reactions from US patents (https://doi.org/10.6084/m9.figshare.5104873; CC0 1.0).
- Peak lists: the simulated spectra of Alberts et al. (https://doi.org/10.5281/zenodo.11611178; CDLA-Sharing-1.0).
- Licence of this file: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
- Cite: Alberts, M., Schilter, O., Zipoli, F., Hartrampf, N. & Laino, T. Unraveling Molecular Structure: A
  Multimodal Spectroscopic Dataset for Chemistry. Advances in Neural Information Processing Systems 37,
  125780–125808 (2024). https://doi.org/10.52202/079017-3996

### `specteach.jsonl` (216 molecules)

Simple compounds with peak lists picked automatically from recorded spectra.

- Structures and recorded spectra: Van Bramer and Bastin (https://doi.org/10.6084/m9.figshare.23739492; CC BY 4.0).
- Peak lists: picked by the authors from the recorded spectra with Mnova.
- Licence of this file: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
- Cite: Van Bramer, S. E. & Bastin, L. D. Spectroscopy Data for Undergraduate Teaching. Journal of Chemical
  Education 100, 3897–3902 (2023). https://doi.org/10.1021/acs.jchemed.3c00046

### `nmrshiftdb2.jsonl` (315 molecules)

Assigned experimental peak lists from the nmrshiftdb2 database.

- Structures and peak lists: nmrshiftdb2 (https://nmrshiftdb.nmr.uni-koeln.de).
- Licence of this file: [nmrshiftdb2 Database License](nmrshiftdb2.license.txt), as published by nmrshiftdb2.
- Cite:
  - Kuhn, S. & Schlörer, N. E. Facilitating quality control for spectra assignments of small organic molecules:
    nmrshiftdb2 – a free in‐house NMR database with integrated LIMS for academic service laboratories. Magnetic
    Resonance in Chemistry 53, 582–589 (2015). https://doi.org/10.1002/mrc.4263
  - Steinbeck, C. & Kuhn, S. NMRShiftDB – compound identification and structure elucidation support through a free
    community-built web database. Phytochemistry 65, 2711–2717 (2004).
    https://doi.org/10.1016/j.phytochem.2004.08.027

### `known_natural_products.jsonl` (9 molecules)

Well-known natural products.

- Structures, and peak lists from the supporting information, of these articles:
  - artemisinin: Turconi, J. et al. Semisynthetic Artemisinin, the Chemical Path to Industrial Production. Organic
    Process Research & Development 18, 417–422 (2014). https://doi.org/10.1021/op4003196
  - ginkgolide C: Hébert, M., Bellavance, G. & Barriault, L. Total Synthesis of Ginkgolide C and Formal Syntheses of
    Ginkgolides A and B. Journal of the American Chemical Society 144, 17792–17796 (2022).
    https://doi.org/10.1021/jacs.2c08351
  - ingenol: Jørgensen, L. et al. 14-Step Synthesis of (+)-Ingenol from (+)-3-Carene. Science 341, 878–882 (2013).
    https://doi.org/10.1126/science.1241606
  - lysergic acid: Liu, Q., Zhang, Y.-A., Xu, P. & Jia, Y. Total Synthesis of (+)-Lysergic Acid. The Journal of
    Organic Chemistry 78, 10885–10893 (2013). https://doi.org/10.1021/jo4018777
  - morphine: Chu, S., Münster, N., Balan, T. & Smith, M. D. A Cascade Strategy Enables a Total Synthesis of
    (±)‐Morphine. Angewandte Chemie International Edition 55, 14306–14309 (2016).
    https://doi.org/10.1002/anie.201608526
  - rapamycin: McAlpine, J. B., Swanson, S. J., Jackson, M. & Whittern, D. N. Revised NMR assignments for rapamycin.
    The Journal of Antibiotics 44, 688–690 (1991). https://doi.org/10.7164/antibiotics.44.688
  - taxol: Nicolaou, K. C. et al. Total Synthesis of Taxol. 1. Retrosynthesis, Degradation, and Reconstitution.
    Journal of the American Chemical Society 117, 624–633 (1995). https://doi.org/10.1021/ja00107a006
  - vancomycin aglycone: Thompson, C., Ge, M. & Kahne, D. Synthesis of Vancomycin from the Aglycon. Journal of the
    American Chemical Society 121, 1237–1244 (1999). https://doi.org/10.1021/ja983504u
- Penicillin G: peak lists recorded by the authors.
- Licence of this file: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
- Cite:
  - the article of each compound above
  - penicillin G: the ChefNMR2 preprint ([Citation](../README.md#citation))

### `taxanes.jsonl` (7 molecules)

Taxane diterpenoids from a study of taxol biosynthesis.

- Structures: McClune et al.; peak lists: the assignment tables of its supporting information.
- Licence of this file: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
- Cite: McClune, C. J. et al. Discovery of FoTO1 and Taxol genes enables biosynthesis of baccatin III. Nature 643,
  582–592 (2025). https://doi.org/10.1038/s41586-025-09090-z

### `novel_natural_products.jsonl` (4 molecules)

Natural products whose structures the paper reports.

- Structures: the ChefNMR2 paper; peak lists: recorded by the authors.
- Licence of this file: [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
- Cite: the ChefNMR2 preprint ([Citation](../README.md#citation))

## Fields

| field | meaning |
|---|---|
| `id` | the molecule's id in this release |
| `name` | the compound's name (natural-product files only) |
| `formula` | the molecular formula, with hydrogens |
| `smiles` | the reported structure |
| `c` | the 13C shifts in ppm, as the model read them |
| `h` | the 1H peaks as the model read them: (shift in ppm, number of protons, multiplicity) |
| `c_text`, `h_text` | the peak lists as text, as the language models of the paper read them |
