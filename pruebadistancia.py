from djitellopy import Tello
import time

me = Tello()
me.connect()
print(me.get_battery())

me.streamoff()
me.streamon()

me.takeoff()

# Ajusta los valores de velocidad y duración según sea necesario
# Estos valores pueden requerir ajustes para lograr el movimiento deseado
speed = 20  # Velocidad de desplazamiento hacia la derecha (ajusta según sea necesario)
duration = 2  # Duración en segundos (ajusta según sea necesario)

# Controlar el movimiento hacia la derecha

me.send_rc_control(0, 0, 0, 0)
time.sleep(1)
me.send_rc_control(speed, 0, 0, 0)
time.sleep(2.2)

# Detener el movimiento
me.send_rc_control(0, 0, 0, 0)
time.sleep(1)

# Aterrizar el dron
me.land()

# Cierra la conexión
me.end()