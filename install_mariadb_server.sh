#!/bin/bash
su -c "
mkdir db
cd $PWD/db
rm -r *
mariadb-install-db --user=$USER --basedir=/usr --datadir=$PWD/db/data
chown -cR $USER $PWD
chmod -R 777 $PWD
"
