#!/usr/bin/env bash
WORKSPACE="/Users/hal0taso/.ghq/github.com/hal0taso/ahc019"
cd $WORKSPACE
g++ main.cpp -std=c++17 -O2 -o test

mkdir -p "${WORKSPACE}/out/"
mkdir -p "${WORKSPACE}/err/"

for input in `ls ${WORKSPACE}/in`
do
echo $input
"${WORKSPACE}/"test < "${WORKSPACE}/in/${input}" 1> "${WORKSPACE}/out/${input}" 2> "${WORKSPACE}/err/${input}"
done