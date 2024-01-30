#!/bin/bash
mkdir downloads
cd downloads
rm -f *.tsv
cd ..
for var in name.basics title.akas title.basics title.crew title.episode title.principals title.ratings
do
    curl https://datasets.imdbws.com/$var.tsv.gz | zcat > ./downloads/$var.tsv
done
