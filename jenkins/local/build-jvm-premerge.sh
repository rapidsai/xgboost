#!/bin/bash
##
#
# Script to build xgboost jar files.
#
# Source tree is supposed to be ready by Jenkins
# before starting this script.
#
###
set -e
gcc --version

#BUILD_ARG="-Dmaven.repo.local=$WORKSPACE/.m2 -Dcudf.classifier=cuda10-centos7"
BUILD_ARG="-Dmaven.repo.local=$WORKSPACE/.m2 -Dcudf.classifier=cuda10"

cd jvm-packages
. /opt/tools/to_cuda10.0.sh
rm -rf ../build
mvn -B $BUILD_ARG clean package
cd ..
