# Multiplicative Filter Networks

This repository contains a PyTorch MFN implementation and code to perform & reproduce experiments from the ICLR 2021 paper [**Multiplicative Filter Networks**](https://openreview.net/forum?id=OmtmcPkkhT) by Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter.

## Requirements

* `pytorch 1.7.0`
* `torchvision 0.8.1`
* `numpy 1.18.1`
* `pillow 6.2.1`
* `scikit-image 0.16.2` 

## Contents

The file `mfn/mfn.py` contains implementations of our two instantiations of multiplicative filter networks: FourierNet (Section 3.1) and GaborNet (Section 3.2). It also contains an MFN base class into which any filter may be plugged in (see documentation for details). 

The `experiments` directory contains scripts that correspond to experiments from the paper. Currently, this has:

* the cameraman image representation experiment from Section 4.1 (`image_rep.py`), and
* the cat video representation experiment from Section 4.1 (`video_rep.py`); see the paper [supplement](https://openreview.net/attachment?id=OmtmcPkkhT&name=supplementary_material) for details on the particular video used

Scripts to reproduce more experiments from the paper will be added soon!

## License 
"Multiplicative Filter Networks" is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.
