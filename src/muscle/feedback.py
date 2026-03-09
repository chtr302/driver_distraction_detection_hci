try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

from src.core.config import (
    GPIO_LED_RED, GPIO_LED_GREEN, 
    GPIO_LED_BLUE, GPIO_BUZZER
)

class FeedbackHAL:
    def __init__(self):
        self.current_level = -1 # Để theo dõi sự thay đổi
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self.pins = [GPIO_LED_RED, GPIO_LED_GREEN, GPIO_LED_BLUE, GPIO_BUZZER]
            for pin in self.pins:
                # TẤT CẢ Active Low: Khởi tạo HIGH để TẮT
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)

    def set_calibration_mode(self, active: bool):
        if not GPIO_AVAILABLE: return
        if active:
            self._set_pins(red=False, green=False, blue=True, buzzer=False)
        else:
            self.trigger(0)

    def trigger(self, level: int):
        if not GPIO_AVAILABLE: return
        # Chỉ ghi đè GPIO nếu level thay đổi để tiết kiệm tài nguyên
        if level == self.current_level:
            return
            
        self.current_level = level
        
        if level == 0:
            # Safe: Green ON
            self._set_pins(red=False, green=True, blue=False, buzzer=False)
        elif level == 1:
            # Warning: Red + Green = Yellow
            self._set_pins(red=True, green=True, blue=False, buzzer=False)
        elif level == 2:
            # Critical: Red + Buzzer
            self._set_pins(red=True, green=False, blue=False, buzzer=True)

    def _set_pins(self, red: bool, green: bool, blue: bool, buzzer: bool):
        """
        Logic Active Low cho toàn bộ linh kiện:
        True (Bật) -> LOW (0V)
        False (Tắt) -> HIGH (3.3V)
        """
        GPIO.output(GPIO_LED_RED, GPIO.LOW if red else GPIO.HIGH)
        GPIO.output(GPIO_LED_GREEN, GPIO.LOW if green else GPIO.HIGH)
        GPIO.output(GPIO_LED_BLUE, GPIO.LOW if blue else GPIO.HIGH)
        GPIO.output(GPIO_BUZZER, GPIO.LOW if buzzer else GPIO.HIGH)

    def cleanup(self):
        if GPIO_AVAILABLE:
            for pin in self.pins:
                GPIO.output(pin, GPIO.HIGH)
            GPIO.cleanup()
