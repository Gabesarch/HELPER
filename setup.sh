#!/bin/sh
pip install gdown
gdown https://drive.google.com/uc?id=1KFUxxL8KU4H8dxBpjhp1SGAf3qnTtEBM
tar -xf messup.tar.gz
mv messup/ data/
rm messup.tar.gz