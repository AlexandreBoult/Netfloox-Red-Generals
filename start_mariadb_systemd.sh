#!/bin/bash
su -c "
systemctl start mysqld
systemctl start mysql.service
systemctl start mariadb
"
echo server started
