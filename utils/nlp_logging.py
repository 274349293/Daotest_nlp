import logging
from logging.handlers import TimedRotatingFileHandler


class CustomLogger:
    def __init__(self, name=None, write_to_file=True):
        """
        初始化日志模块
        :param name: 日志器的名称，默认为None，将会自动设置为当前模块名
        :param write_to_file: 是否将日志写入文件，默认为True
        """
        # Create logger
        self.logger = logging.getLogger(name if name else __name__)
        self.logger.setLevel(logging.INFO)

        # Formatter for log messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Handlers
        if write_to_file:
            # File handler for logging to a file, rotates at midnight, keeps 7 backups
            file_handler = TimedRotatingFileHandler(
                filename='./utils/log/app.log',
                when='midnight',
                interval=1,
                backupCount=7,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Console handler for logging to the console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        else:
            # Only console handler if not writing to file
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def info(self, message):
        """记录信息级别的日志"""
        self.logger.info(message)

    def debug(self, message):
        """记录调试级别的日志"""
        self.logger.debug(message)

    def warning(self, message):
        """记录警告级别的日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误级别的日志"""
        self.logger.error(message)

    def critical(self, message):
        """记录严重错误级别的日志"""
        self.logger.critical(message)


# 示例用法
if __name__ == "__main__":
    logger = CustomLogger("DaoTest_nlp_Logger", write_to_file=False)  # 设置为False以打印日志
    logger.info("这是一条信息日志")
    logger.debug("这是一条调试日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.critical("这是一条严重错误日志")
