#!/bin/bash

for  sd in simulations/*
do
    id=`basename $sd`
    #echo "$sd -> $id"
    RE="$id "`cat $sd/results.dat| grep -v \# `
    echo $RE
    #exit
done

