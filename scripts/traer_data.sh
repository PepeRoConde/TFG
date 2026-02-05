#!/usr/bin/env bash

DIR="${1:-data/runs}"

rsync -avz --progress \
  -e "ssh -o ProxyJump=j.rconde@enterprise.dc.fi.udc.es" \
  "j.rconde@10.56.33.66:~/repo/${DIR}/" \
  "./${DIR}/"
