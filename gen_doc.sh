#!/usr/bin/env bash

docsrepo="../docs"
# whether docs repository exist
if [ ! -d  "$docsrepo" ]; then
  echo 'docs not exist. please make sure you clone the repository'
  exit 0
fi

echo 'start to generate document'
pdoc3 --html beta_rec
echo 'generate finish, copy to docs ...'
cp -r html/beta_rec/* ../docs
echo 'done! Please swith to docs and commit changes'

