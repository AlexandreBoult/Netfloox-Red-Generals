#!/bin/bash
./create_env_linux.sh
source env/bin/activate
gunicorn --bind 0.0.0.0:5000 wsgi:app
#python -m flask --app webapp.py run --debug --host=0.0.0.0 --with-threads
