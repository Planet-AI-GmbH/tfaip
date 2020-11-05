# XLSX Experimenter

This tool is designed for scheduling/logging of experiments by using an xlsx file.
Pass an xlsx file to parse `--xlsx`, the desired `--gpus` for scheduling, optionally the python executable of the target virtual env via `--python`.
Launch in the "Working"-Dir

**Notes:** 
* This tool currently requires python >=3.7

Examples:
* Run on gpus 1 and 3, `tfaip-experimenter --xlsx demo.xlsx --gpus 1 3`
* Run on cpu only `tfaip-experimenter --xlsx demo.xlsx`

## XLSX file

First row:
* USER (optional): Arbitrary user parameters
* RESULT (optional): Arbitrary result of the experiment (must be manually inserted)
* SKIP (required): Skip this experiment if the xlsx file is parsed
* SCRIPT (required): The script to run (expected to be in the scripts dir)
* ID (required): ID of the run (best practice: simply use a running integer)
* CLEANUP (required): Cleanup the output directory (not implemented!)
* PARAMS (requied): Tag to mark the begin of parameters passed to the "SCRIPT"

Second row (for PARAMS), defines the paramter groups:
* if `default`: no group
* if `NAME` (e.g. `optimizer_params`): pass the following parameters to the `--optimizer_params` flag
* if `EMPTY`: use the previous group definition

Third row (PARAM labels, setup):
* specify the parameter name, e.g. `epochs` (without dashes)
* optionally pass additional flags (e.g. `post_distortions:split:empty`):
  * `:split` split the value of this parameter into individual args
  * `:empty` allow this parameter to be empty when added to the list (note, by default, empty `NULL` values are omitted)
  
 

