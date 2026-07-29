# Data: provenance and citation

`svd_boost_features.csv` is **not** raw audio. It holds acoustic features derived
from the Saarbrücken Voice Database (SVD), one row per recording:

| | |
|---|---|
| Rows | 15,873 recordings (15,555 after de-duplication in `main.py`) |
| Speakers | 1,119 |
| Labels | 8,928 healthy (`0`), 6,945 pathological (`1`) |
| Columns | 90: the 88 eGeMAPS v02 functionals, plus `speaker` and `label` |

`speaker` is the SVD session identifier, carried through so that `GroupKFold` can
keep one person's recordings out of both the training and the test side. `label`
is the organic-pathology-vs-healthy distinction, **not** a Parkinson's label, and
not a diagnosis of any specific condition.

Features were extracted per recording with openSMILE (eGeMAPS v02, Functionals
level): pitch, jitter, shimmer, harmonics-to-noise ratio, formants, and spectral
shape. No clinical questionnaire scores are included.

## This data is borrowed, not ours

**None of the underlying data originated with this project.** It is a derivative of
the Saarbrücken Voice Database (SVD), collected at the former Institut für Phonetik,
Universität des Saarlandes, and now hosted by Essen University Hospital. This
repository contributes only the modelling code; the recordings, the speakers, and
the clinical labels are the database's work, reused here under its license.

Get the original (38 GB of audio, which this repository does not mirror):

- Database home: <https://stimmdb.coli.uni-saarland.de/>
- Full dataset: <https://doi.org/10.5281/zenodo.16874898>

## License of the source data

The SVD is released under **Creative Commons Attribution 4.0 International
(CC BY 4.0)**: <https://creativecommons.org/licenses/by/4.0/>

CC BY 4.0 permits redistribution, adaptation, and commercial use, *provided the
creators are credited, the license is named and linked, and changes are indicated.*
That is what this file exists to do. Note that this license covers the **data
only**; the code in this repository is under its own separate LICENSE, and neither
license extends to the other.

### Required attribution

> "Saarbruecken Voice Database" by Manfred Pützer and William J. Barry, hosted by
> Essen University Hospital, is licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
> Source: <https://doi.org/10.5281/zenodo.16874898>
>
> **Changes made:** the audio is not redistributed here. The 88 eGeMAPS v02
> functionals were extracted from each recording with openSMILE and stored as one
> CSV row per recording; rows were de-duplicated, and the SVD session identifier
> was retained as the `speaker` column so evaluation can be grouped by speaker.
> No recordings, transcripts, or clinical records are included.

If you redistribute `svd_boost_features.csv` or anything derived from it, that
attribution, including the statement of changes, has to travel with it.

## How to cite

If you use this data, cite the database and the feature-extraction work:

```bibtex
@dataset{svd,
  title     = {Saarbruecken Voice Database},
  author    = {P{\"u}tzer, Manfred and Barry, William J.},
  year      = {2008},
  version   = {v2},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.16874898},
  url       = {https://doi.org/10.5281/zenodo.16874898},
  note      = {Licensed CC BY 4.0}
}

@article{gemaps,
  title   = {The {Geneva} Minimalistic Acoustic Parameter Set ({GeMAPS}) for
             Voice Research and Affective Computing},
  author  = {Eyben, Florian and Scherer, Klaus R. and Schuller, Bj{\"o}rn W. and
             Sundberg, Johan and Andr{\'e}, Elisabeth and Busso, Carlos and
             Devillers, Laurence Y. and Epps, Julien and Laukka, Petri and
             Narayanan, Shrikanth S. and Truong, Khiet P.},
  journal = {IEEE Transactions on Affective Computing},
  volume  = {7},
  number  = {2},
  pages   = {190--202},
  year    = {2016}
}

@inproceedings{opensmile,
  title     = {openSMILE -- The {Munich} Versatile and Fast Open-Source Audio
               Feature Extractor},
  author    = {Eyben, Florian and W{\"o}llmer, Martin and Schuller, Bj{\"o}rn},
  booktitle = {Proceedings of ACM Multimedia},
  pages     = {1459--1462},
  year      = {2010}
}
```