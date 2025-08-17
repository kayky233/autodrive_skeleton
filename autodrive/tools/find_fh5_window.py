# tools/find_fh5_window.py
import ctypes, ctypes.wintypes as wt

def get_foreground_window_rect():
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    rect = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wt.POINT(rect.left, rect.top)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    left, top = pt.x, pt.y
    width, height = rect.right - rect.left, rect.bottom - rect.top
    return {"left": left, "top": top, "width": width, "height": height}

if __name__ == "__main__":
    print(get_foreground_window_rect())
