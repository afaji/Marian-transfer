Marian regression tests
=======================

<b>Marian</b> is an efficient Neural Machine Translation framework written in
pure C++ with minimal dependencies.

This repository contains the regression test framework for the main development
repository: `https://github.com/marian-nmt/marian-dev`.


## Structure

Directories:

* `tests` - regression tests
* `tools` - scripts and repositories
* `models` - models used in regression tests
* `data` - data used in training or decoding tests

Each test consists of:

* `test_*.sh` file
* `setup.sh` (optional)
* `teardown.sh` (optional)


## Usage

Downloading required data and tools:

    make install

Running regression tests:

    MARIAN=/path/to/marian-dev/build ./run_mrt.sh

Enabling multi-GPU tests:

    CUDA_VISIBLE_DEVICES=0,1 ./run_mrt.sh

More invocation examples:

    ./run_mrt.sh tests/training/basics
    ./run_mrt.sh tests/training/basics/test_valid_script.sh
    ./run_mrt.sh previous.log

where `previous.log` contains a list of test files, one test per line.  This
file is automatically generated each time `./run_mrt.sh` finishes running.

Cleaning test artifacts:

    make clean


## Debugging failed tests

Failed tests are displayed at the end of testing or in `previous.log`, e.g.:

    Failed:
    - tests/training/restoring/multi-gpu/test_async.sh
    - tests/training/embeddings/test_custom_embeddings.sh
    ---------------------
    Ran 145 tests in 00:48:48.210s, 143 passed, 0 skipped, 2 failed

Logging messages are in files ending with _.sh.log_ suffix:

    less tests/training/restoring/multi-gpu/test_async.sh.log

The last command in most tests is an execution of a custom `diff` tool, which
prints the exact invocation commands with absolute paths. It can be used to
display the differences that cause the test fails.


## Adding new tests

Use templates provided in `tests/_template`.

Please follow these recommendations:

* Test one thing at a time
* For comparing outputs with numbers, please use float-friendly
  `tools/diff-nums.py` instead of GNU `diff`
* Make your tests deterministic using `--no-shuffle --seed 1111` or similar
* Make training execution time as short as possible, for instance, by reducing
  the size of the network and the number of iterations
* Do not run decoding or scoring on files longer than ca. 10-100 lines
* If your tests require downloading and running a custom model, please keep it
  as small as possible, and contact me (Roman) to upload it into our storage


## Acknowledgements

The development of Marian received funding from the European Union's
_Horizon 2020 Research and Innovation Programme_ under grant agreements
688139 ([SUMMA](http://www.summa-project.eu); 2016-2019),
645487 ([Modern MT](http://www.modernmt.eu); 2015-2017),
644333 ([TraMOOC](http://tramooc.eu/); 2015-2017),
644402 ([HiML](http://www.himl.eu/); 2015-2017),
the Amazon Academic Research Awards program,
the World Intellectual Property Organization,
and is based upon work supported in part by the Office of the Director of
National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
(IARPA), via contract #FA8650-17-C-9117.

This software contains source code provided by NVIDIA Corporation.

