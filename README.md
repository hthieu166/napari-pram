# napari-pram

[![License](https://img.shields.io/pypi/l/napari-pram.svg?color=green)](https://github.com/hthieu166/napari-pram/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-pram.svg?color=green)](https://pypi.org/project/napari-pram)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-pram.svg?color=green)](https://python.org)
[![tests](https://github.com/hthieu166/napari-pram/workflows/tests/badge.svg)](https://github.com/hthieu166/napari-pram/actions)
[![codecov](https://codecov.io/gh/hthieu166/napari-pram/branch/main/graph/badge.svg)](https://codecov.io/gh/hthieu166/napari-pram)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-pram)](https://napari-hub.org/plugins/napari-pram)

Plugin for PRAM data annotation and processing.

![PRAM Demo](docs/figs/demo.jpg)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Usage

### Open `napari-pram` toolbox:

On the toolbar, select ``[Plugins] > napari-pram: Open PRAM's toolbox``

## Installation
You will need a python environment. I recommend [Conda](https://docs.conda.io/en/latest/miniconda.html). Create a new environment, for example:
    
    conda create --name napari-env python=3.7 pip 

Activate the new environment:

    conda activate napari-env 

Install [napari](https://napari.org/tutorials/fundamentals/installation) via [pip]:

    pip install napari[all]

Then you can finally install our plugin `napari-pram` via [pip]:

    pip install napari-pram

Alternatively, the plugin can be installed using napari-GUI

``[Plugins] > Install/Uninstall Plugins`` and search for `napari-pram`

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-pram" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
