rsync -avz --progress \
  --exclude-from=.rsyncignore \
  -e "ssh -o ProxyJump=j.rconde@enterprise.dc.fi.udc.es" \
  ./ \
  j.rconde@10.56.33.66:~/repo/
