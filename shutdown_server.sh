#!/bin/bash
echo "SHUTDOWN;" | mariadb
su -c "
cd $PWD/db
chown -cR $USER $PWD
chmod -R 777 $PWD
"
