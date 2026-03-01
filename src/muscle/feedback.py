try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

from src.core.config import (
    GPIO_LED_YELLOW, GPIO_LED_RED, 
    GPIO_BUZZER, GPIO_VIBRATION
)

class FeedbackHAL:
    def __init__(self):
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self.pins = [GPIO_LED_YELLOW, GPIO_LED_RED, GPIO_BUZZER, GPIO_VIBRATION]
            for pin in self.pins:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

    def trigger(self, level: int):
        if not GPIO_AVAILABLE:
            return

        if level == 0:
            self._set_pins(yellow=False, red=False, buzzer=False, vibration=False)
        elif level == 1:
            self._set_pins(yellow=True, red=False, buzzer=False, vibration=False)
        elif level == 2:
            self._set_pins(yellow=False, red=True, buzzer=True, vibration=True)

    def _set_pins(self, yellow: bool, red: bool, buzzer: bool, vibration: bool):
        GPIO.output(GPIO_LED_YELLOW, GPIO.HIGH if yellow else GPIO.LOW)
        GPIO.output(GPIO_LED_RED, GPIO.HIGH if red else GPIO.LOW)
        GPIO.output(GPIO_BUZZER, GPIO.HIGH if buzzer else GPIO.LOW)
        GPIO.output(GPIO_VIBRATION, GPIO.HIGH if vibration else GPIO.LOW)

    def cleanup(self):
        if GPIO_AVAILABLE:
            GPIO.cleanup()
