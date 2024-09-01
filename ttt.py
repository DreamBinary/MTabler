import multiprocessing
import logging
logger = multiprocessing.get_logger()


def worker():
    logger.info("Hello from the subprocess!")


if __name__ == "__main__":
    # 配置日志记录
    multiprocessing.log_to_stderr(logging.INFO)

    # 创建并启动子进程
    p = multiprocessing.Process(target=worker)
    p.start()

    # 等待子进程完成
    p.join()
