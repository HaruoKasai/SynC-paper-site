# Third-party software

The custom analysis code in this repository is released under the MIT License (see [LICENSE](LICENSE)). It was run alongside, and in some cases consumes the output of, the following third-party packages, which are not covered by that licence and remain under their own terms:

| Package | Version used | Licence |
| --- | --- | --- |
| [Suite2p](https://github.com/MouseLand/suite2p/tree/v0.14.5) | 0.14.5 | GPL-3.0 |
| [OASIS](https://github.com/j-friedrich/OASIS) (deconvolution, via Suite2p) | as bundled with Suite2p 0.14.5 | GPL-3.0 |
| [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/tree/v2.3.10) | 2.3.10 | LGPL-3.0 |
| [Fiji / ImageJ](https://imagej.net/licensing/) | 1.54p | GPL (Fiji; component-specific exceptions) / public domain (original ImageJ core) |
| [Python](https://docs.python.org/release/3.9.13/license.html) | 3.9.13 | PSF License |
| [MATLAB](https://www.mathworks.com/products/matlab.html) | R2024b | Proprietary (MathWorks); a MATLAB licence is required to run the .m scripts |
| [NI-DAQmx](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html) | — | Proprietary (National Instruments); used for data acquisition only |

This repository does not redistribute these packages. Users must obtain them from their original distributors and comply with their respective licences. Licence identifiers above summarize the principal upstream terms for the versions used; bundled components may carry additional notices, and the original distributions remain authoritative.
