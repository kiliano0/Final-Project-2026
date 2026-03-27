import RPi.GPIO as GPIO
import time

SOLENOID_PIN = 23  # GPIO23

GPIO.setmode(GPIO.BCM)
GPIO.setup(SOLENOID_PIN, GPIO.OUT)

try:
    print("Solenoid ON")
    GPIO.output(SOLENOID_PIN, GPIO.HIGH)
    time.sleep(1)

    print("Solenoid OFF")
    GPIO.output(SOLENOID_PIN, GPIO.LOW)
    time.sleep(1)

finally:
    GPIO.cleanup()
