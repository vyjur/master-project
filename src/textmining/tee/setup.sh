#! /bin/sh

# Install heideltime
cd heideltime

./install_heideltime_standalone.sh

cd heideltime-standalone/treetagger

wget --no-verbose https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/norwegian.par.gz
cp tree_tagger/norwegian.par tree_tagger/auto-norwegian.par
cp tree_tagger/norwegian-abbreviations tree_tagger/auto-norwegian-abbreviations

cd ..
cp heideltime-standalone/treetagger/lib/norwegian.par heideltime-standalone/treetagger/lib/auto-norwegian.par 
cp heideltime-standalone/treetagger/lib/auto-norwegian-abbreviations heideltime-standalone/treetagger/lib/auto-norwegian-abbreviations
