import multiprocessing
import time


def my_function():
    import torch  # 在子进程中使用torch
    # 模拟一个需要时间运行的函数
    while True:
        time.sleep(1)
        print(torch.__version__)


if __name__ == "__main__":
    p = multiprocessing.Process(target=my_function)
    p.start()
    import torch  # 在主进程中使用torch
    print(torch.__version__)
    import os
    os.system("pip install torch==2.4.0")
    print("---", torch.__version__)
    p.join()
