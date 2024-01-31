#!/bin/bash
su -c "
cd /var/lib/mysql
rm -r *
mysql_install_db --user=mysql --basedir=/usr --datadir=/var/lib/mysql
"
