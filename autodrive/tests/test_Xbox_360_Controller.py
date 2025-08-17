from time import sleep
from vgamepad import VX360Gamepad, XUSB_BUTTON

def main():
    pad = VX360Gamepad()
    print("[test] Virtual Xbox 360 gamepad created.")
    print("[test] Open 'joy.cpl' → select Xbox 360 Controller → Properties to watch axes.")
    print("[test] Press Ctrl+C to exit (script will reset the gamepad).")

    try:
        # 一次按键闪烁，方便在游戏里看到按钮事件
        pad.press_button(button=XUSB_BUTTON.XUSB_GAMEPAD_A); pad.update(); sleep(0.2)
        pad.release_button(button=XUSB_BUTTON.XUSB_GAMEPAD_A); pad.update(); sleep(0.5)

        t = 0
        while True:
            # 让右扳机 0..255..0 循环
            trig = int((1 + __import__("math").sin(t)) * 127.5)
            pad.right_trigger(value=trig)

            # 左摇杆左右扫动：-32768..32767..-32768
            stick = int(__import__("math").sin(t) * 32767)
            pad.left_joystick(x_value=stick, y_value=0)

            pad.update()
            sleep(0.02)
            t += 0.08
    except KeyboardInterrupt:
        pass
    finally:
        # 退出前复位
        pad.reset()
        pad.update()
        print("\n[test] Gamepad reset. Bye.")

if __name__ == "__main__":
    main()
