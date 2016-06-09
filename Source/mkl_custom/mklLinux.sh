#!/bin/bash

# Stop on error, trace commands
set -e -x

# Enter directory the script is located in
cd "$( dirname "${BASH_SOURCE[0]}" )"

MKLROOT=/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl
MKLBUILDERROOT=$MKLROOT/tools/builder
CURRENTVER=$(cat version.txt)

rm -rf Publish

mkdir Publish{,/$CURRENTVER{,/x64}}

for THREADING in parallel sequential
do
    LIBBASENAME=libmkl_cntk_$(echo $THREADING | cut -c 1)
    make -f $MKLBUILDERROOT/makefile libintel64 export=cntklist.txt threading=$THREADING name=$LIBBASENAME MKLROOT=$MKLROOT
    mkdir Publish/$CURRENTVER/x64/$THREADING
    mv $LIBBASENAME.so Publish/$CURRENTVER/x64/$THREADING
done

cp -p $MKLROOT/../compiler/lib/intel64_lin/libiomp5.so Publish/$CURRENTVER/x64/parallel

rsync -av --files-from cntkheaderlist.txt $MKLROOT/include Publish/$CURRENTVER/include
