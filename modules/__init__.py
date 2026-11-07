"""业务功能模块层。

每个子包对应设计文档中的一个功能模块，彼此只通过 ``core`` 中的
数据结构与接口协作，不允许相互循环依赖：

    data_import      数据导入（不依赖趋势分析）
    data_processing  数据处理（只依赖 core）
    trend_analysis   趋势分析与历史检索（可依赖相似度模块提供的服务接口）
    similarity       相似度算法（不依赖任何 UI 与文件解析）
    export           结果导出
"""

from __future__ import annotations

__all__: list[str] = []
