Hola. Te comparto a continuación los datos de tu cuenta para conectarte a nuestro servidor y las instrucciones para acceder.

Tienes que conectarte primero a la subred del grupo VARPA a través del dominio "enterprise.dc.fi.udc.es". Para ello, necesitas estar en la red de la universidad, ya sea porque estás conectado en un edificio de la universidad o porque estás utilizando la VPN. Una vez accedas a nuestra subred, aparecerás en una primera máquina desde la cual podrás conectarte a la máquina en la que realizarás los experimentos. Esta última máquina tiene la IP 10.56.33.66. Para ambas máquinas el usuario es "j.rconde", como en tu correo de la UDC, y la contraseña "pepe.1". Tienes que cambiar la contraseña la primera vez que te conectes.

Resumen:
Username: j.rconde
Contraseña: pepe.1
Máquina de acceso a la subred del grupo VARPA: enterprise.dc.fi.udc.es (cambia contraseña con yppasswd)
Máquina final para ejecutar el código: 10.56.33.66 (cambia contraseña con passwd)

Acceso y transferencia de ficheros:
Una vez hayas accedido a las máquinas por primera vez y hayas cambiado las contraseñas, es posible usar ProxyJump (tanto con ssh como scp) para acceder directamente a la máquina final o transferir ficheros directamente a/desde la máquina final. Por ejemplo:

Para acceder a la máquina:   ssh -J j.rconde@enterprise.dc.fi.udc.es j.rconde@10.56.33.66

Para transferir un "fichero" hacia tu "home" en la máquina:   scp -o ProxyJump=j.rconde@enterprise.dc.fi.udc.es "fichero" j.rconde@10.56.33.66:~/
