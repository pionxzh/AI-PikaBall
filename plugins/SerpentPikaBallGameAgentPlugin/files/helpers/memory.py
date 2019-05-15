from ctypes import *
from ctypes.wintypes import *
import win32api, win32ui, win32process
import ctypes

def readInfo ():
    OpenProcess = windll.kernel32.OpenProcess
    ReadProcessMemory = windll.kernel32.ReadProcessMemory
    FindWindowA = windll.user32.FindWindowA
    GetWindowThreadProcessId = windll.user32.GetWindowThreadProcessId

    PROCESS_ALL_ACCESS = 0x1F0FFF
    try:
        HWND = win32ui.FindWindow(None, u"Pikachu VolleyBall-Serpentai").GetSafeHwnd()
    except:
        HWND = win32ui.FindWindow(None, u"Pikachu VolleyBall-Serpentai ( Paused! )").GetSafeHwnd()
    PID = win32process.GetWindowThreadProcessId(HWND)[1]
    processHandle = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS,False,PID)

    # Open process for reading & writing:
    BaseAddress = win32process.EnumProcessModules(processHandle)[0]

    #print(f"HWND: {HWND}")
    #print(f"PID: {PID}")
    #print(f"PROCESS: {processHandle}")
    #print(f'BaseAddress: {BaseAddress}')

    

    # Read out game info:
    numRead = c_int()
    Score = c_int()
    Ball = c_int()
    Ai = c_int()
    AiX = c_int()
    AiY = c_int()
    ComX = c_int()
    ComY = c_int()
    BallX = c_int()
    BallY = c_int()
    ScoreL = c_int()
    ScoreR = c_int()
    CollisionValue = c_int()
    CollisionX = c_int()
    CollisionY = c_int()
    ReadProcessMemory(processHandle, BaseAddress + 0x1348C, byref(Score), 4, byref(numRead))
    #print(Score.value)
    ReadProcessMemory(processHandle, Score.value + 0xA8, byref(Score), 4, byref(numRead))
    #print(Score.value)
    ReadProcessMemory(processHandle, Score.value + 0x3C, byref(ScoreL), 4, byref(numRead))
    ReadProcessMemory(processHandle, Score.value + 0x40, byref(ScoreR), 4, byref(numRead))
    #print(f'L Score: {ScoreL.value}')
    #print(f'R Score: {ScoreR.value}')
    ReadProcessMemory(processHandle, Score.value + 0x108, byref(ComX), 4, byref(numRead))
    ReadProcessMemory(processHandle, Score.value + 0x10C, byref(ComY), 4, byref(numRead))
    #print(f'COM:      ({ComX.value}, {ComY.value})')
    ReadProcessMemory(processHandle, Score.value + 0x14, byref(Ball), 4, byref(numRead))
    ReadProcessMemory(processHandle, Ball.value + 0x30, byref(BallX), 4, byref(numRead))
    ReadProcessMemory(processHandle, Ball.value + 0x34, byref(BallY), 4, byref(numRead))

    ReadProcessMemory(processHandle, Ball.value + 0x4C, byref(CollisionValue), 4, byref(numRead))
    ReadProcessMemory(processHandle, Ball.value + 0x50, byref(CollisionX), 4, byref(numRead))
    ReadProcessMemory(processHandle, Ball.value + 0x54, byref(CollisionY), 4, byref(numRead))
    
    ReadProcessMemory(processHandle, Score.value + 0x10, byref(Ai), 4, byref(numRead))
    ReadProcessMemory(processHandle, Ai.value + 0xA8, byref(AiX), 4, byref(numRead))
    ReadProcessMemory(processHandle, Ai.value + 0xAC, byref(AiY), 4, byref(numRead))
    #print(f'AI:        ({AiX.value}, {AiY.value})')
    #print(f'BALL:      ({BallX.value}, {BallY.value})')
    #print(f'Collision: ({CollisionValue.value}, {CollisionX.value}, {CollisionY.value})')
    return(ComX.value, ComY.value, AiX.value, AiY.value, BallX.value, BallY.value, ScoreL.value, ScoreR.value, CollisionValue.value, CollisionX.value, CollisionY.value)

print('Memory Info: ', readInfo())
