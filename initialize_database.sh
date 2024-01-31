#!/bin/bash
mariadb < init.sql
su -c "
cd $PWD/db
chown -cR $USER $PWD
chmod -R 777 $PWD
"
