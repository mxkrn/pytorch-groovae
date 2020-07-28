#!/bin/bash

python setup.py sdist

VERSION=$1
GDRIVE=/mnt/c/Users/maxkr/Google\ Drive/Research/rhythmflow/
FILE=`find /home/max/repos/rhythmflow/dist/rhythmflow-$VERSION.tar.gz`

if [ -z $FILE ]
then
	echo "file not found"
else
	rsync -t $FILE /mnt/c/Users/maxkr/Google\ Drive/Research/rhythmflow && echo "uploaded $FILE"
fi
