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
echo "Sign Jar?: $1"

# Should be called under 'jvm-packages'
stashJars(){
    MODULE_OUT=$OUT/$1
    mkdir -p $MODULE_OUT
    cp -ft $MODULE_OUT $1/target/*.jar
}

###### Preparation before build ######
SIGN_FILE=$1
BUILD_ARG="-DskipTests -Dmaven.repo.local=$WORKSPACE/.m2"

if [ "$SIGN_FILE" == true ]; then
    # Build javadoc and sources only when SIGN_JAR
    BUILD_ARG="$BUILD_ARG -s settings.xml -Prelease-to-sonatype,sonatype-stage"
fi

###### Build jar and its libraries ######
cd jvm-packages
## 1) Build libxgboost4j.so for CUDA9.2 CUDA10.1
rm -rf ../build
. /opt/tools/to_cuda9.2.sh
./create_jni.py cuda9.2
rm -rf ../build
. /opt/tools/to_cuda10.1.sh
./create_jni.py cuda10.1
## 2) Build libxgboost4j.so for CUDA10.0 and the jar file
rm -rf ../build
. /opt/tools/to_cuda10.0.sh
mvn -B clean package $BUILD_ARG

###### Stash jar files for modules ######
stashJars xgboost4j
stashJars xgboost4j-spark

cd ..
