#!/bin/sh
# Filter the entire index file containing image metadata for just the images in the input directory.
# This shell script takes one argument:
#
# Argument 1:   Directory to the folder containing input images
# Usage:        Directory as a string

if [ $# -eq 1 ]; then
    echo "Filtering index files in:"
    pwd
else
    echo "Usage: bash $0 ['path']"
    echo "['path'] = Path to input imagery directory?"
fi

# Loop through all index files
for i in *INDEX*; do
    base=`basename $i .TAB`
    new="$base.filtered.TAB"
    
    # Creat new index
    touch $new
    echo "touch $new"
    
    # Get product names from input directory
    images=`ls $1`

    # Loop through all images
    for image in $images; do
        name=`basename $image '.tif'`
        echo $name
        grep $name $i >> $new
    done
done