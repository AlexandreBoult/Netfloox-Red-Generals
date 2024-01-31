#!/bin/bash
echo "SHUTDOWN;" | mariadb
sh -c "
cd $PWD/db
chown -cR $USER $PWD
chmod -R 777 $PWD
"
