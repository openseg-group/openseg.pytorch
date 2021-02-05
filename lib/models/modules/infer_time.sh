#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile

export PYTHONPATH="../../../":$PYTHONPATH
export eval_os_8=1
export bn_type="inplace_abn"

# ${PYTHON} -m lib.models.modules.psp_block
# ${PYTHON} -m lib.models.modules.aspp_block
# ${PYTHON} -m lib.models.modules.base_oc_block
export isa_type="base_oc"
${PYTHON} -m lib.models.modules.isa_block
${PYTHON} -m lib.models.modules.spatial_ocr_block
