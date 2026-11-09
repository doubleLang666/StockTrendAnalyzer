"""K 线图配置（设计文档 4.5 / 6）。

本模块只描述"图怎么画"，不依赖 Qt，也不依赖具体的绘图调用，
因此界面设置面板、离屏测试与后续阶段（十字光标、指标开关）可以共用同一份配置。

约定（对应设计文档第 5 节视觉要求）：

* 红涨绿跌：``up_color`` 为红、``down_color`` 为绿、``flat_color`` 用于平盘十字星；
* 深色行情界面：图表区域深灰底色、低对比网格；
* 所有颜色集中在此，绘制代码不出现颜色字面量。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final, Iterable

__all__ = [
    "ChartConfig",
    "DEFAULT_BOLL_K",
    "DEFAULT_BOLL_WINDOW",
    "DEFAULT_MA_WINDOWS",
    "DEFAULT_MACD_PARAMS",
    "DEFAULT_RSI_WINDOWS",
    "DEFAULT_VOLUME_MA_WINDOWS",
    "MA_COLORS",
    "RSI_COLORS",
]

#: 阶段一固定展示的三条均线；30/60 日留给阶段三的设置面板。
DEFAULT_MA_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20)
DEFAULT_VOLUME_MA_WINDOWS: Final[tuple[int, ...]] = (5, 10)
DEFAULT_MACD_PARAMS: Final[tuple[int, int, int]] = (12, 26, 9)
DEFAULT_RSI_WINDOWS: Final[tuple[int, ...]] = (6, 12, 24)
DEFAULT_BOLL_WINDOW: Final[int] = 20
DEFAULT_BOLL_K: Final[float] = 2.0

#: 均线颜色：与国内行情软件习惯接近（5 黄、10 蓝、20 紫、30 绿、60 灰）。
MA_COLORS: Final[dict[int, str]] = {
    5: "#f2c14e",
    10: "#5aa9ff",
    20: "#d977d9",
    30: "#7bd389",
    60: "#b6bfcb",
}

#: RSI 三条线按顺序取色（不按周期值取色，改周期也不会串色）。
RSI_COLORS: Final[tuple[str, ...]] = ("#f2c14e", "#5aa9ff", "#d977d9", "#7bd389")
_FALLBACK_MA_COLORS: Final[tuple[str, ...]] = (
    "#f2c14e",
    "#5aa9ff",
    "#d977d9",
    "#7bd389",
    "#b6bfcb",
    "#ff9f6b",
)


def clean_windows(values: Iterable[Any] | None) -> tuple[int, ...]:
    """规范化均线窗口：去重、升序、只保留正整数。"""
    if not values:
        return ()
    windows: set[int] = set()
    for item in values:
        try:
            window = int(item)
        except (TypeError, ValueError):
            continue
        if window > 0:
            windows.add(window)
    return tuple(sorted(windows))


@dataclass(frozen=True, slots=True)
class ChartConfig:
    """一张 K 线图的全部外观与绘制参数。"""

    # --- 指标与副图 ---
    ma_windows: tuple[int, ...] = DEFAULT_MA_WINDOWS
    volume_ma_windows: tuple[int, ...] = DEFAULT_VOLUME_MA_WINDOWS
    show_volume: bool = True
    initial_bars: int = 240
    show_macd: bool = True
    show_rsi: bool = False
    show_boll: bool = False
    rsi_windows: tuple[int, ...] = DEFAULT_RSI_WINDOWS
    macd_params: tuple[int, int, int] = DEFAULT_MACD_PARAMS
    boll_window: int = DEFAULT_BOLL_WINDOW
    boll_k: float = DEFAULT_BOLL_K

    # --- 颜色 ---
    up_color: str = "#e5484d"        # 红：收盘 >= 开盘
    down_color: str = "#2fa46b"      # 绿：收盘 < 开盘
    flat_color: str = "#c9d1d9"      # 平盘十字星
    background: str = "#0f1318"      # 整张图底色
    axes_background: str = "#151a21"  # 绘图区底色
    grid_color: str = "#2a313b"
    grid_alpha: float = 0.35
    foreground: str = "#c9d1d9"      # 主文字与坐标轴
    muted: str = "#8b949e"           # 次要文字（刻度、说明）

    # --- 几何 ---
    price_axis_on_right: bool = True
    body_width_ratio: float = 0.62   # 实体宽度 / 相邻交易日间距
    flat_body_ratio: float = 0.15    # 十字星实体高度 / 当日振幅
    wick_linewidth: float = 0.9
    ma_linewidth: float = 1.2
    volume_alpha: float = 0.85
    highlight_color: str = "#f2c14e"  # 相似区间高亮
    highlight_alpha: float = 0.14

    def __post_init__(self) -> None:
        cleaned = clean_windows(self.ma_windows)
        object.__setattr__(self, "ma_windows", cleaned)
        object.__setattr__(self, "volume_ma_windows", clean_windows(self.volume_ma_windows))
        object.__setattr__(self, "rsi_windows", clean_windows(self.rsi_windows) or DEFAULT_RSI_WINDOWS)
        params = tuple(int(item) for item in self.macd_params)
        if len(params) != 3 or not 0 < params[0] < params[1] or params[2] <= 0:
            params = DEFAULT_MACD_PARAMS
        object.__setattr__(self, "macd_params", params)  # type: ignore[arg-type]
        if self.boll_window <= 0:
            object.__setattr__(self, "boll_window", DEFAULT_BOLL_WINDOW)
        if self.boll_k <= 0:
            object.__setattr__(self, "boll_k", DEFAULT_BOLL_K)
        if self.initial_bars < 10:
            object.__setattr__(self, "initial_bars", 10)
        if not 0.1 <= self.body_width_ratio <= 0.95:
            raise ValueError("K线实体宽度比例应在 0.1~0.95 之间")

    # ------------------------------------------------------------------
    # 派生
    # ------------------------------------------------------------------
    def color_for_window(self, window: int) -> str:
        """均线颜色：已知窗口用固定配色，其余按顺序取备用色。"""
        color = MA_COLORS.get(int(window))
        if color is not None:
            return color
        try:
            index = self.ma_windows.index(int(window))
        except ValueError:
            index = 0
        return _FALLBACK_MA_COLORS[index % len(_FALLBACK_MA_COLORS)]

    def with_ma(self, windows: Iterable[int]) -> "ChartConfig":
        return replace(self, ma_windows=clean_windows(windows))

    def with_volume(self, enabled: bool) -> "ChartConfig":
        return replace(self, show_volume=bool(enabled))

    def with_indicators(
        self,
        *,
        macd: bool | None = None,
        rsi: bool | None = None,
        boll: bool | None = None,
    ) -> "ChartConfig":
        """按需开关指标副图；参数为 ``None`` 表示保持原值。"""
        changes: dict[str, bool] = {}
        if macd is not None:
            changes["show_macd"] = bool(macd)
        if rsi is not None:
            changes["show_rsi"] = bool(rsi)
        if boll is not None:
            changes["show_boll"] = bool(boll)
        return replace(self, **changes)

    def with_initial_bars(self, bars: int) -> "ChartConfig":
        return replace(self, initial_bars=int(bars))

    # ------------------------------------------------------------------
    # 读取应用配置
    # ------------------------------------------------------------------
    @classmethod
    def from_app_config(cls, config: Any | None = None) -> "ChartConfig":
        """从 :class:`app.config.AppConfig` 的 ``chart`` 段读取初值。

        配置缺失或类型不符时回落到本类默认值，绝不因为配置问题让界面无法绘图。
        """
        defaults = cls()
        getter = getattr(config, "get", None)
        if getter is None:
            return defaults
        try:
            ma_windows = clean_windows(
                getter("chart.ma_windows", defaults.ma_windows)
            ) or defaults.ma_windows
            volume_windows = clean_windows(
                getter("chart.volume_ma_windows", defaults.volume_ma_windows)
            )
            show_volume = bool(getter("chart.show_volume", defaults.show_volume))
            initial_bars = int(getter("chart.initial_bars", defaults.initial_bars))
            show_macd = bool(getter("chart.show_macd", defaults.show_macd))
            show_rsi = bool(getter("chart.show_rsi", defaults.show_rsi))
            show_boll = bool(getter("chart.show_boll", defaults.show_boll))
            rsi_windows = clean_windows(
                getter("chart.rsi_windows", defaults.rsi_windows)
            ) or defaults.rsi_windows
            macd_params = tuple(
                int(item)
                for item in getter("chart.macd_params", defaults.macd_params)
            )
            boll_window = int(getter("chart.boll_window", defaults.boll_window))
            boll_k = float(getter("chart.boll_k", defaults.boll_k))
        except (TypeError, ValueError):
            return defaults
        return cls(
            ma_windows=ma_windows,
            volume_ma_windows=volume_windows,
            show_volume=show_volume,
            initial_bars=initial_bars,
            show_macd=show_macd,
            show_rsi=show_rsi,
            show_boll=show_boll,
            rsi_windows=rsi_windows,
            macd_params=macd_params,  # type: ignore[arg-type]
            boll_window=boll_window,
            boll_k=boll_k,
        )

    @classmethod
    def light(cls) -> "ChartConfig":
        """浅色主题（阶段三设置面板预留）。"""
        return cls(
            background="#ffffff",
            axes_background="#fbfcfd",
            grid_color="#dfe3e8",
            grid_alpha=0.7,
            foreground="#1f2933",
            muted="#7b8794",
        )
