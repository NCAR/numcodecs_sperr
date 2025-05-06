#!/bin/bash

rm -rf build
python setup.py bdist_wheel

/glade/derecho/scratch/haiyingx/patchelf/install_casper/bin/patchelf --set-rpath "\$ORIGIN/../lib64" lib64/libSPERR.so
#/glade/derecho/scratch/haiyingx/patchelf/src/patchelf --set-rpath "\$ORIGIN/../lib64" lib64/libzstd.so
/glade/derecho/scratch/haiyingx/patchelf/install_casper/bin/patchelf --set-rpath "\$ORIGIN/../lib64" lib64/libMURaMKit.so

mkdir build/bdist.linux-x86_64/wheel
cp -r lib64/ build/bdist.linux-x86_64/wheel/
ls -al build/bdist.linux-x86_64/wheel/

/glade/derecho/scratch/haiyingx/patchelf/install_casper/bin/patchelf --set-rpath "\$ORIGIN/../lib64" build/lib.linux-x86_64-cpython-39/numcodecs/sperr.cpython-39-x86_64-linux-gnu.so
/glade/derecho/scratch/haiyingx/patchelf/install_casper/bin/patchelf --set-rpath "\$ORIGIN/../lib64" build/lib.linux-x86_64-cpython-39/numcodecs/prefilter.cpython-39-x86_64-linux-gnu.so
python setup.py bdist_wheel
