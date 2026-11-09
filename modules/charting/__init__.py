"""专业行情图表模块（设计文档第二节「图表模块重构」）。

与 ``modules/trend_analysis`` 的分工：

* ``trend_analysis`` 负责趋势曲线、多周期比较与历史区间检索（折线图，保持原样）；
* ``charting`` 负责券商风格的 K 线图：蜡烛、成交量、均线与技术指标副图。

模块内部按职责拆分，互不依赖 Qt：

* :mod:`~modules.charting.chart_config` —— 外观与绘制参数；
* :mod:`~modules.charting.candle_data` —— 行情整理与校验（只读原始数据）；
* :mod:`~modules.charting.indicators` —— 均线与 MACD / RSI / BOLL 计算（本地计算，可单测）；
* :mod:`~modules.charting.candlestick_chart` —— 主图蜡烛与均线；
* :mod:`~modules.charting.volume_chart` —— 成交量副图；
* :mod:`~modules.charting.indicator_chart` —— MACD / RSI 副图与 BOLL 主图叠加；
* :mod:`~modules.charting.chart_figure` —— 组装整张图并返回艺术家引用；
* :mod:`~modules.charting.chart_interaction` —— 十字光标、滚轮缩放、拖动平移与数据提示；
* :mod:`~modules.charting.comparison_chart` —— 历史区间对比（双图并排 / 归一化走势）。
"""

from __future__ import annotations

from modules.charting.candle_data import CandleData, prepare_candle_data
from modules.charting.candlestick_chart import CandleArtists, draw_candlesticks, update_candlesticks
from modules.charting.chart_config import ChartConfig
from modules.charting.chart_figure import CandleFigure, build_candlestick_figure
from modules.charting.chart_interaction import (
    CandleChartController,
    CursorInfo,
    WindowInfo,
)
from modules.charting.comparison_chart import (
    DualCandleFigure,
    NORMALIZED_NOTE,
    build_dual_candle_figure,
    build_normalized_figure,
    normalize_to_100,
)
from modules.charting.indicator_chart import (
    BollArtists,
    MacdArtists,
    RsiArtists,
    draw_boll,
    draw_macd,
    draw_rsi,
    update_boll,
    update_macd,
    update_rsi,
)
from modules.charting.indicators import (
    BollData,
    MacdData,
    build_boll,
    build_macd,
    build_moving_averages,
    build_rsi,
    build_volume_moving_averages,
    macd,
    rsi,
    sma,
)
from modules.charting.volume_chart import VolumeArtists, draw_volume, update_volume

__all__ = [
    "ChartConfig",
    "CandleData",
    "CandleFigure",
    "CandleArtists",
    "VolumeArtists",
    "MacdArtists",
    "RsiArtists",
    "BollArtists",
    "MacdData",
    "BollData",
    "CandleChartController",
    "CursorInfo",
    "WindowInfo",
    "prepare_candle_data",
    "build_moving_averages",
    "build_volume_moving_averages",
    "build_macd",
    "build_rsi",
    "build_boll",
    "sma",
    "macd",
    "rsi",
    "draw_candlesticks",
    "update_candlesticks",
    "draw_volume",
    "update_volume",
    "draw_macd",
    "update_macd",
    "draw_rsi",
    "update_rsi",
    "draw_boll",
    "update_boll",
    "build_candlestick_figure",
    "DualCandleFigure",
    "NORMALIZED_NOTE",
    "build_dual_candle_figure",
    "build_normalized_figure",
    "normalize_to_100",
]
