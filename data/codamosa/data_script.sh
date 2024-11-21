#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# From Pynguin eval
# checking out a later version of apimd so we don't have to mess with the loader code
cd test-apps && git clone https://github.com/KmolYuan/apimd.git && cd apimd && git checkout f32841b && cd ../..
cd test-apps && git clone https://github.com/realpython/codetiming.git && cd codetiming && git checkout a7ad85a && cd ../..
cd test-apps && git clone https://github.com/lidatong/dataclasses-json.git && cd dataclasses-json && git checkout 3dc59e01 && cd ../..
cd test-apps && git clone https://github.com/rr-/docstring_parser.git && cd docstring_parser && git checkout a5dc2cd77 && cd ../..
cd test-apps && git clone https://github.com/huzecong/flutes.git && cd flutes && git checkout 49647e4b && cd ../..
cd test-apps && git clone https://gitlab.com/finite-loop/flutils.git && cd flutils && git checkout df0f84e1 && cd ../..
cd test-apps && git clone https://github.com/httpie/httpie.git && cd httpie && git checkout bb36897 && cd ../..
cd test-apps && git clone https://github.com/PyCQA/isort.git && cd isort && git checkout a6222a8 && cd ../..
cd test-apps && git clone https://github.com/lk-geimfari/mimesis.git && cd mimesis && git checkout 310092ce && cd ../..
cd test-apps && git clone https://github.com/nvbn/py-backwards.git && cd py-backwards && git checkout 8be3c4430 && cd ../..
cd test-apps && git clone https://github.com/przemyslawjanpietrzak/pyMonet.git && cd pyMonet && git checkout f132cfa && cd ../..
cd test-apps && git clone https://github.com/vst/pypara.git && cd pypara && git checkout 7d705a54 && cd ../..
cd test-apps && git clone https://github.com/relekang/python-semantic-release.git && cd python-semantic-release && git checkout 3689157c2 && cd ../..
cd test-apps && git clone https://github.com/daveoncode/python-string-utils.git && cd python-string-utils && git checkout d903db3c2 && cd ../..
cd test-apps && git clone https://github.com/akatrevorjay/pytutils.git && cd pytutils && git checkout 9813bb3 && cd ../..
cd test-apps && git clone https://github.com/sanic-org/sanic.git && cd sanic && git checkout 93a0246 && cd ../..
cd test-apps && git clone https://github.com/feluxe/sty.git && cd sty && git checkout f99e9186 && cd ../..
cd test-apps && git clone https://github.com/thonny/thonny.git && cd thonny && git checkout fb389f4 && cd ../..
cd test-apps && git clone https://github.com/encode/typesystem.git && cd typesystem && git checkout 6a9590c125 && cd ../..
cd test-apps && git clone https://github.com/psf/black.git && cd black && git checkout 23541263 && cd ../..
# From BugsinPy
cd test-apps && git clone https://github.com/cool-RR/PySnooper.git && cd PySnooper && git checkout 31bfc63 && cd ../..
cd test-apps && git clone https://github.com/ansible/ansible.git && cd ansible && git checkout f00f123 && cd ../..
cd test-apps && git clone https://github.com/cookiecutter/cookiecutter.git && cd cookiecutter && git checkout 1c0b5b11 && cd ../..
cd test-apps && git clone https://github.com/tiangolo/fastapi.git && cd fastapi && git checkout 864643e && cd ../..
cd test-apps && git clone https://github.com/keras-team/keras.git && cd keras && git checkout 2c48a3b3 && cd ../..
cd test-apps && git clone https://github.com/spotify/luigi.git && cd luigi && git checkout f2f631b && cd ../..
cd test-apps && git clone https://github.com/matplotlib/matplotlib.git && cd matplotlib && git checkout 9765379 && cd ../..
cd test-apps && git clone https://github.com/pandas-dev/pandas.git && cd pandas && git checkout 945c9ed && cd ../..
cd test-apps && git clone https://github.com/scrapy/scrapy.git && cd scrapy && git checkout 61130c8 && cd ../..
cd test-apps && git clone https://github.com/explosion/spaCy.git && cd spaCy && git checkout 800737b && cd ../..
cd test-apps && git clone https://github.com/nvbn/thefuck.git && cd thefuck && git checkout 0949d2e && cd ../..
cd test-apps && git clone https://github.com/tornadoweb/tornado.git && cd tornado && git checkout 2047e7a && cd ../..
cd test-apps && git clone https://github.com/tqdm/tqdm.git && cd tqdm && git checkout 18d7aa4 && cd ../..
cd test-apps && git clone https://github.com/ytdl-org/youtube-dl.git && cd youtube-dl && git checkout b224cf3 && cd ../..


$SCRIPT_DIR/get_dependent_packages.sh $SCRIPT_DIR/source_dirs.txt

$SCRIPT_DIR/get_potential_modules.sh $SCRIPT_DIR/source_dirs.txt $SCRIPT_DIR/../test-apps/all_potential_modules.csv

$SCRIPT_DIR/module_post_processing.sh

