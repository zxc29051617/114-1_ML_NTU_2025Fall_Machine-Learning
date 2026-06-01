#!/bin/bash

DIR=test_sl
rm -rf ${DIR}
mkdir -p ${DIR}/model/
touch ${DIR}/Training.log
PYTHONPATH=. python -u strength_detection/code/train_sl.py go ${DIR} test.cfg 2>&1 | tee ${DIR}/op.log