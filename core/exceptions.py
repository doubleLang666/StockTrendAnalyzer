"""项目统一异常体系。

所有模块对外抛出的异常都应继承 StockAnalyzerError，
便于 UI 层集中捕获并转换为用户可读的提示信息。
"""

from __future__ import annotations


class StockAnalyzerError(Exception):
    """项目异常基类。"""

    #: 面向用户的简短错误类别名。
    label: str = "程序错误"

    def __init__(self, message: str = "", *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def user_message(self) -> str:
        """返回适合直接展示给用户的单行文本。"""
        if self.detail:
            return f"{self.message}（{self.detail}）"
        return self.message


class ConfigurationError(StockAnalyzerError):
    """配置项不合法或缺失。"""

    label = "配置错误"


class DataImportError(StockAnalyzerError):
    """数据导入失败基类。"""

    label = "数据导入失败"


class EncodingDetectionError(DataImportError):
    """无法确定文件编码。"""

    label = "编码识别失败"


class ParseError(DataImportError):
    """文件结构或字段解析失败。"""

    label = "文件解析失败"


class ValidationError(StockAnalyzerError):
    """数据校验未通过。"""

    label = "数据校验未通过"


class ProcessingError(StockAnalyzerError):
    """数据处理失败。"""

    label = "数据处理失败"


class InsufficientDataError(ProcessingError):
    """有效数据量不足以完成计算。"""

    label = "有效数据不足"


class AlgorithmError(StockAnalyzerError):
    """相似度算法计算失败。"""

    label = "算法计算失败"


class UnknownAlgorithmError(AlgorithmError):
    """请求的算法未注册。"""

    label = "未知算法"


class ServiceNotRegisteredError(StockAnalyzerError):
    """通过服务注册表请求了尚未注册的服务。"""

    label = "服务不可用"


class ExportError(StockAnalyzerError):
    """分析结果导出失败。"""

    label = "结果导出失败"
