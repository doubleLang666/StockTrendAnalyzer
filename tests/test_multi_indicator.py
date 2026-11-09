"""多指标联动分析：计算引擎测试（设计文档第 8 节 T1–T6 的引擎部分）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from core.exceptions import InsufficientDataError, ProcessingError
from modules.multi_indicator import (
    MultiIndicatorRequest,
    MultiSearchRequest,
    analyze_frame,
    available_indicators,
    compute_indicator_score,
    default_weights,
    get_indicator,
    indicator_series,
    normalize_weights,
    prepare_pair,
    search_history,
    select_default_indicators,
    transform_values,
)
from modules.multi_indicator.preparation import (
    TRANSFORM_BASE100,
    TRANSFORM_LOG_BASE100,
    TRANSFORM_OFFSET100,
    SeriesPair,
)
from modules.similarity.service import SimilarityService


def make_frame(count: int = 60, *, start: str = "2024-01-02", volume: bool = True, turnover: bool = False) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=count)
    base = 100.0 + np.arange(count) * 0.8 + np.sin(np.arange(count) / 3.0) * 4.0
    close = pd.Series(base, index=index)
    data = {
        "open": close * (1 - 0.004),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
    }
    if volume:
        data["volume"] = pd.Series(1e6 + np.arange(count) * 5e3, index=index)
    if turnover:
        data["turnover"] = pd.Series(1.5 + np.zeros(count), index=index)
    return pd.DataFrame(data)


def pair_for(frame: pd.DataFrame, key: str, *, same: bool = True) -> SeriesPair:
    spec = get_indicator(key)
    index = frame.index
    compare_end = index[-1] if same else index[len(index) // 2]
    return prepare_pair(frame, spec, index[0], index[-1], index[0], compare_end)


@pytest.fixture()
def service() -> SimilarityService:
    return SimilarityService()


class TestSpecs:
    def test_available_indicators_follow_columns(self):
        frame = make_frame()
        keys = [spec.key for spec in available_indicators(frame)]
        assert keys == ["close", "open", "high", "low", "volume", "pct_change"]
        # 缺换手率时，换手率不会出现在可用列表里
        assert "turnover" not in keys

    def test_default_selection_uses_document_order(self):
        frame = make_frame()
        keys = select_default_indicators(available_indicators(frame))
        assert keys == ("close", "volume", "pct_change")

    def test_turnover_becomes_available_when_column_exists(self):
        frame = make_frame(turnover=True)
        keys = select_default_indicators(available_indicators(frame))
        assert keys == ("close", "volume", "turnover", "pct_change")

    def test_unknown_indicator_raises(self):
        with pytest.raises(ProcessingError):
            get_indicator("nope")

    def test_indicator_series_computes_pct_change(self):
        frame = make_frame()
        series = indicator_series(frame, get_indicator("pct_change"))
        assert series.isna().iloc[0]
        expected = (frame["close"].iloc[1] - frame["close"].iloc[0]) / frame["close"].iloc[0] * 100
        assert series.iloc[1] == pytest.approx(expected)

    def test_missing_source_field_raises(self):
        frame = make_frame().drop(columns=["volume"])
        with pytest.raises(ProcessingError):
            indicator_series(frame, get_indicator("volume"))


class TestPreparation:
    def test_price_transform_uses_first_value_as_100(self):
        frame = make_frame()
        values = frame["close"].iloc[:10]
        transformed, kind, note = transform_values(get_indicator("close"), values)
        assert kind == TRANSFORM_BASE100
        assert transformed.iloc[0] == pytest.approx(100.0)
        assert "100" in note

    def test_rate_transform_offsets_by_100(self):
        values = pd.Series([1.5, -0.5, 0.0])
        transformed, kind, _ = transform_values(get_indicator("pct_change"), values)
        assert kind == TRANSFORM_OFFSET100
        assert transformed.tolist() == pytest.approx([101.5, 99.5, 100.0])

    def test_volume_falls_back_to_log_when_first_value_is_zero(self):
        values = pd.Series([0.0, 1e3, 2e3, 4e3])
        transformed, kind, note = transform_values(get_indicator("volume"), values)
        assert kind == TRANSFORM_LOG_BASE100
        assert np.isfinite(transformed).all()
        assert "log1p" in note

    def test_volume_uses_base100_when_positive(self):
        values = pd.Series([2e6, 3e6, 4e6])
        transformed, kind, _ = transform_values(get_indicator("volume"), values)
        assert kind == TRANSFORM_BASE100
        assert transformed.iloc[0] == pytest.approx(100.0)

    def test_pair_truncates_to_shorter_range(self):
        frame = make_frame(80)
        index = frame.index
        pair = prepare_pair(
            frame,
            get_indicator("close"),
            index[0],
            index[59],
            index[20],
            index[49],
        )
        assert pair.length == 30
        assert len(pair.target) == len(pair.compare)
        assert pair.target_dates[0] == index[0]
        assert pair.compare_dates[0] == index[20]

    def test_pair_keeps_nan_gaps_and_reports_ratio(self):
        frame = make_frame(20)
        frame.loc[frame.index[5:12], "close"] = np.nan
        pair = prepare_pair(
            frame, get_indicator("close"), frame.index[0], frame.index[-1], frame.index[0], frame.index[-1]
        )
        assert pair.valid_count == 13
        assert pair.valid_ratio == pytest.approx(13 / 20)
        assert not pair.is_valid  # 65% 低于 80% 门槛
        assert "门槛" in pair.reason

    def test_pair_rejects_relative_index(self):
        frame = make_frame(10).set_index(pd.RangeIndex(10))
        with pytest.raises(ProcessingError):
            prepare_pair(frame, get_indicator("close"), 0, 9, 0, 9)

    def test_pair_rejects_too_short_range(self):
        frame = make_frame(10)
        with pytest.raises(InsufficientDataError):
            prepare_pair(
                frame,
                get_indicator("close"),
                frame.index[0],
                frame.index[0],
                frame.index[1],
                frame.index[1],
            )


class TestScoring:
    def test_self_comparison_scores_100(self, service):
        frame = make_frame()
        pair = pair_for(frame, "close")
        score = compute_indicator_score(service, pair)
        assert score.is_valid
        assert score.raw_value("pearson") == pytest.approx(1.0)
        assert score.raw_value("rmse") == pytest.approx(0.0)
        assert score.pearson_score == pytest.approx(100.0)
        assert score.rmse_score == pytest.approx(100.0)
        assert score.mae_score == pytest.approx(100.0)
        assert score.combined_score == pytest.approx(100.0)

    def test_pearson_score_maps_minus_one_to_zero(self, service):
        frame = make_frame(40)
        spec = get_indicator("close")
        first = frame.index[:20]
        second = frame.index[20:]
        pair = prepare_pair(frame, spec, first[0], first[-1], second[0], second[-1])
        # 人为构造完全反向的两段（同一组数值倒序）
        values = np.linspace(100, 120, 20)
        mirrored = SeriesPair(
            spec=spec,
            target=pd.Series(values),
            compare=pd.Series(values[::-1]),
            target_raw=pd.Series(values),
            compare_raw=pd.Series(values[::-1]),
            target_dates=pair.target_dates,
            compare_dates=pair.compare_dates,
            transform=TRANSFORM_BASE100,
            note="",
        )
        score = compute_indicator_score(service, mirrored)
        assert score.raw_value("pearson") == pytest.approx(-1.0)
        assert score.pearson_score == pytest.approx(0.0)
        assert score.combined_score < 50

    def test_scores_are_scale_invariant(self, service):
        frame = make_frame()
        index = frame.index
        spec = get_indicator("close")
        left = frame["close"].iloc[:30].reset_index(drop=True)
        right = frame["close"].iloc[30:60].reset_index(drop=True)
        pair = SeriesPair(
            spec=spec,
            target=left,
            compare=right,
            target_raw=left,
            compare_raw=right,
            target_dates=index[:30],
            compare_dates=index[30:60],
            transform=TRANSFORM_BASE100,
            note="",
        )
        score = compute_indicator_score(service, pair)
        assert score.combined_score is not None
        # 同一组数据整体放大 1000 倍（量纲变化）不应影响 0–100 分（T6）
        scaled_pair = SeriesPair(
            spec=spec,
            target=left * 1000,
            compare=right * 1000,
            target_raw=left * 1000,
            compare_raw=right * 1000,
            target_dates=index[:30],
            compare_dates=index[30:60],
            transform=TRANSFORM_BASE100,
            note="",
        )
        scaled_score = compute_indicator_score(service, scaled_pair)
        assert scaled_score.combined_score == pytest.approx(score.combined_score, rel=1e-9)

    def test_flat_series_at_different_levels_is_identical_after_transform(self, service):
        spec = get_indicator("close")
        left_raw = pd.Series(np.full(20, 50.0))
        right_raw = pd.Series(np.full(20, 60.0))
        left, _, _ = transform_values(spec, left_raw)
        right, _, _ = transform_values(spec, right_raw)
        pair = SeriesPair(
            spec=spec,
            target=left,
            compare=right,
            target_raw=left_raw,
            compare_raw=right_raw,
            target_dates=pd.date_range("2024-01-01", periods=20, freq="B"),
            compare_dates=pd.date_range("2024-01-01", periods=20, freq="B"),
            transform=TRANSFORM_BASE100,
            note="",
        )
        score = compute_indicator_score(service, pair)
        assert score.combined_score == pytest.approx(100.0)

    def test_invalid_pair_produces_no_score(self, service):
        spec = get_indicator("close")
        single = pd.Series([100.0])
        pair = SeriesPair(
            spec=spec,
            target=single,
            compare=single.copy(),
            target_raw=single,
            compare_raw=single.copy(),
            target_dates=pd.DatetimeIndex(["2024-01-01"]),
            compare_dates=pd.DatetimeIndex(["2024-01-01"]),
            transform=TRANSFORM_BASE100,
            note="",
            min_valid_ratio=0.8,
        )
        score = compute_indicator_score(service, pair)
        assert not score.is_valid
        assert score.combined_score is None
        assert score.score_for("combined") is None

    def test_flat_identical_series_gets_full_marks(self, service):
        spec = get_indicator("close")
        flat = pd.Series(np.full(20, 50.0))
        pair = SeriesPair(
            spec=spec,
            target=flat,
            compare=flat.copy(),
            target_raw=flat,
            compare_raw=flat.copy(),
            target_dates=pd.date_range("2024-01-01", periods=20, freq="B"),
            compare_dates=pd.date_range("2024-01-01", periods=20, freq="B"),
            transform=TRANSFORM_BASE100,
            note="",
        )
        score = compute_indicator_score(service, pair)
        assert score.rmse_score == pytest.approx(100.0)
        assert score.mae_score == pytest.approx(100.0)

    def test_mode_selection_returns_matching_component(self, service):
        frame = make_frame()
        pair = pair_for(frame, "close")
        score = compute_indicator_score(service, pair)
        assert score.score_for("pearson") == pytest.approx(score.pearson_score)
        assert score.score_for("rmse") == pytest.approx(score.rmse_score)
        assert score.score_for("combined") == pytest.approx(score.combined_score)
        with pytest.raises(ProcessingError):
            compute_indicator_score(service, pair, mode="nope")


class TestWeights:
    def test_default_weights_follow_document_example(self):
        weights = default_weights(["close", "volume", "turnover", "pct_change"])
        assert weights["close"] == pytest.approx(0.4)
        assert weights["volume"] == pytest.approx(0.3)
        assert weights["turnover"] == pytest.approx(0.2)
        assert weights["pct_change"] == pytest.approx(0.1)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_default_weights_fall_back_to_equal_shares(self):
        weights = default_weights(["open", "high"])
        assert weights == pytest.approx({"open": 0.5, "high": 0.5})

    def test_normalize_weights_renormalizes_valid_subset(self):
        weights = {"close": 0.4, "volume": 0.3, "turnover": 0.2, "pct_change": 0.1}
        normalized, changed = normalize_weights(weights, ["close", "turnover", "pct_change"])
        assert changed
        assert normalized["close"] == pytest.approx(0.4 / 0.7)
        assert normalized["turnover"] == pytest.approx(0.2 / 0.7)
        assert sum(normalized.values()) == pytest.approx(1.0)

    def test_normalize_weights_keeps_weights_when_all_valid(self):
        weights = {"close": 0.6, "volume": 0.4}
        normalized, changed = normalize_weights(weights, ["close", "volume"])
        assert not changed
        assert normalized == pytest.approx(weights)

    def test_normalize_weights_returns_empty_without_valid_keys(self):
        assert normalize_weights({"close": 1.0}, []) == ({}, False)

    def test_normalize_weights_uses_equal_shares_when_all_zero(self):
        normalized, changed = normalize_weights({"close": 0.0, "volume": 0.0}, ["close", "volume"])
        assert changed
        assert normalized == pytest.approx({"close": 0.5, "volume": 0.5})


class TestAnalysis:
    def request(self, frame: pd.DataFrame, keys, **kwargs) -> MultiIndicatorRequest:
        index = frame.index
        defaults = dict(
            target_start=index[0],
            target_end=index[29],
            compare_start=index[30],
            compare_end=index[59],
        )
        defaults.update(kwargs)
        return MultiIndicatorRequest(keys=tuple(keys), **defaults)

    def test_weighted_total_matches_manual_sum(self, service):
        frame = make_frame()
        result = analyze_frame(
            frame,
            self.request(frame, ["close", "volume"], weights={"close": 0.75, "volume": 0.25}),
            similarity_service=service,
        )
        assert result.has_result
        close_score = result.score_of("close").combined_score
        volume_score = result.score_of("volume").combined_score
        assert result.total_score == pytest.approx(0.75 * close_score + 0.25 * volume_score)

    def test_invalid_indicator_does_not_contribute_zero(self, service):
        frame = make_frame(80)
        frame.loc[frame.index[30:70], "volume"] = np.nan  # 对比区间成交量全部缺失
        result = analyze_frame(
            frame,
            self.request(frame, ["close", "volume"], weights={"close": 0.6, "volume": 0.4}),
            similarity_service=service,
        )
        valid = result.valid_scores
        assert [score.spec_key for score in valid] == ["close"]
        assert result.weights == pytest.approx({"close": 1.0})
        assert result.weight_changed
        assert result.total_score == pytest.approx(result.score_of("close").combined_score)
        assert any("成交量" in label for label, _ in result.skipped)

    def test_weight_change_does_not_touch_raw_values(self, service):
        frame = make_frame()
        first = analyze_frame(
            frame,
            self.request(frame, ["close", "volume"], weights={"close": 0.5, "volume": 0.5}),
            similarity_service=service,
        )
        second = analyze_frame(
            frame,
            self.request(frame, ["close", "volume"], weights={"close": 0.9, "volume": 0.1}),
            similarity_service=service,
        )
        assert first.score_of("close").raw_value("pearson") == pytest.approx(
            second.score_of("close").raw_value("pearson")
        )
        assert first.total_score != pytest.approx(second.total_score)

    def test_single_mode_total_uses_single_scores(self, service):
        frame = make_frame()
        result = analyze_frame(
            frame,
            self.request(frame, ["close", "volume"], weights={"close": 0.5, "volume": 0.5}, score_mode="pearson"),
            similarity_service=service,
        )
        expected = 0.5 * result.score_of("close").pearson_score + 0.5 * result.score_of("volume").pearson_score
        assert result.total_score == pytest.approx(expected)

    def test_identical_ranges_give_full_marks(self, service):
        frame = make_frame()
        index = frame.index
        request = MultiIndicatorRequest(
            target_start=index[0],
            target_end=index[39],
            compare_start=index[0],
            compare_end=index[39],
            keys=("close", "volume", "pct_change"),
        )
        result = analyze_frame(frame, request, similarity_service=service)
        assert result.total_score == pytest.approx(100.0)
        assert result.grade == "高度相似"

    def test_empty_keys_raise(self, service):
        frame = make_frame()
        with pytest.raises(ProcessingError):
            analyze_frame(frame, self.request(frame, []), similarity_service=service)

    def test_relative_index_raises(self, service):
        frame = make_frame(30).set_index(pd.RangeIndex(30))
        with pytest.raises(ProcessingError):
            analyze_frame(
                frame,
                MultiIndicatorRequest(0, 10, 10, 20, ("close",)),
                similarity_service=service,
            )

    def test_summary_mentions_participants_and_mode(self, service):
        frame = make_frame()
        result = analyze_frame(
            frame, self.request(frame, ["close", "volume"]), similarity_service=service
        )
        text = result.summary_text()
        assert "综合相似度" in text
        assert "收盘价" in text
        assert "成交量" in text
        assert "综合评分" in text


class TestPage:
    def make_page(self, frame):
        from core.data_models import MarketDataset
        from ui.multi_indicator_page import MultiIndicatorPage

        dataset = MarketDataset("测试指数", frame)
        page = MultiIndicatorPage(lambda: (dataset,), lambda: SimilarityService())
        return page

    def test_refresh_selects_source_and_defaults(self, qapp):
        page = self.make_page(make_frame(120))
        assert page.combo_source.count() == 1
        assert page.combo_source.currentText() == "原始数据 · 测试指数"
        assert page.checked_keys() == ("close", "volume", "pct_change")

    def test_only_available_indicators_are_enabled(self, qapp):
        page = self.make_page(make_frame(120))
        items = {
            page.list_indicators.item(row).data(Qt.ItemDataRole.UserRole): page.list_indicators.item(row)
            for row in range(page.list_indicators.count())
        }
        assert not (items["turnover"].flags() & Qt.ItemFlag.ItemIsUserCheckable)
        assert "换手率" in page.label_hint.text()

    def test_analyze_fills_table_and_summary(self, qapp):
        page = self.make_page(make_frame(120))
        page.analyze()
        assert page.result is not None
        assert page.result.has_result
        assert page.table_results.rowCount() == len(page.result.scores)
        assert page.table_results.item(0, 0).text() == page.result.scores[0].label
        assert "均分" in page.table_results.horizontalHeaderItem(9).text()
        assert page.result.total_score is not None

    def test_weights_default_to_document_example(self, qapp):
        page = self.make_page(make_frame(120))
        weights = page.weights_fraction()
        assert set(weights) == {"close", "volume", "pct_change"}
        assert weights["close"] == pytest.approx(0.5)
        assert weights["volume"] == pytest.approx(0.375)
        assert weights["pct_change"] == pytest.approx(0.125)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert "100.0%" in page.label_weight_total.text()

    def test_weight_input_updates_total_and_request(self, qapp):
        page = self.make_page(make_frame(120))
        page._weight_inputs["close"].setValue(60.0)
        assert "110.0%" in page.label_weight_total.text()
        assert "归一化" in page.label_weight_total.text()
        page.analyze()
        assert page.result is not None
        weights = page.result.weights
        assert weights["close"] == pytest.approx(0.6 / 1.1, rel=1e-6)
        # 权重只影响综合分，不改变单指标的原始算法结果
        assert page.result.scores[0].pearson_score is not None

    def test_reset_weights_restores_defaults(self, qapp):
        page = self.make_page(make_frame(120))
        page._weight_inputs["close"].setValue(10.0)
        page._weight_inputs["volume"].setValue(10.0)
        page._reset_weights()
        weights = page.weights_fraction()
        assert weights["close"] == pytest.approx(0.5)
        assert weights["volume"] == pytest.approx(0.375)

    def test_unchecked_indicator_weight_disabled(self, qapp):
        page = self.make_page(make_frame(120))
        assert page._weight_inputs["close"].isEnabled()
        for row in range(page.list_indicators.count()):
            item = page.list_indicators.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == "close":
                item.setCheckState(Qt.CheckState.Unchecked)
        assert not page._weight_inputs["close"].isEnabled()
        assert not page._weight_inputs["turnover"].isEnabled()

    def test_analyze_requires_checked_indicator(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox

        page = self.make_page(make_frame(120))
        for row in range(page.list_indicators.count()):
            item = page.list_indicators.item(row)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Unchecked)
        messages: list = []
        monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: messages.append(a[2])))
        page.analyze()
        assert messages and "指标" in messages[0]


class TestSearch:
    """历史检索：不泄漏未来数据、按综合分排序、避免相邻重叠。"""

    @staticmethod
    def repeating_frame(count: int = 200, period: int = 40) -> pd.DataFrame:
        """构造周期性行情：目标区间在历史里能找到几乎一模一样的窗口。"""
        index = pd.bdate_range("2020-01-02", periods=count)
        phase = np.arange(count) % period
        pattern = np.sin(phase / period * 2 * np.pi) * 6.0 + phase * 0.05
        close = pd.Series(3000.0 + pattern, index=index)
        volume = pd.Series(1e6 * (1.0 + np.cos(phase / period * 2 * np.pi) * 0.3), index=index)
        return pd.DataFrame({"close": close, "volume": volume})

    def request(self, frame: pd.DataFrame, **kwargs) -> MultiSearchRequest:
        index = frame.index
        defaults = dict(
            target_start=index[-40],
            target_end=index[-1],
            scope_start=index[0],
            scope_end=index[-1],
            keys=("close", "volume"),
            top_n=5,
        )
        defaults.update(kwargs)
        return MultiSearchRequest(**defaults)

    def test_target_period_is_not_reused_as_candidate(self, service):
        frame = self.repeating_frame()
        results = search_history(frame, self.request(frame), similarity_service=service)
        target_start = frame.index[-40]
        for item in results:
            assert item.end_date < target_start

    def test_results_are_ranked_and_deduplicated(self, service):
        frame = self.repeating_frame()
        result = search_history(frame, self.request(frame, top_n=4), similarity_service=service)
        assert len(result) == 4
        scores = [item.total_score for item in result]
        assert scores == sorted(scores, reverse=True)
        assert [item.rank for item in result] == [1, 2, 3, 4]
        starts = [frame.index.get_loc(item.start_date) for item in result]
        gap = 40 * (1.0 - 0.5)
        for left, right in zip(starts, starts[1:]):
            assert abs(left - right) >= gap

    def test_repeating_pattern_gets_high_score(self, service):
        frame = self.repeating_frame()
        results = search_history(frame, self.request(frame, top_n=1), similarity_service=service)
        assert results
        assert results[0].total_score > 80.0

    def test_analysis_detail_matches_row(self, service):
        frame = self.repeating_frame()
        result = search_history(frame, self.request(frame, top_n=2), similarity_service=service)
        for item in result:
            assert item.analysis is not None
            assert item.analysis.total_score == pytest.approx(item.total_score, rel=1e-9)
            assert len(item.scores) == 2
            assert "收盘价" in item.describe()

    def test_score_mode_is_respected(self, service):
        frame = self.repeating_frame()
        pearson = search_history(
            frame, self.request(frame, score_mode="pearson"), similarity_service=service
        )
        assert pearson[0].analysis is not None
        assert pearson[0].analysis.score_mode == "pearson"

    def test_requires_keys_and_valid_mode(self, service):
        frame = self.repeating_frame()
        with pytest.raises(ProcessingError):
            search_history(frame, self.request(frame, keys=()), similarity_service=service)
        with pytest.raises(ProcessingError):
            search_history(frame, self.request(frame, score_mode="kendall"), similarity_service=service)

    def test_scope_without_room_raises(self, service):
        frame = self.repeating_frame(count=60)
        with pytest.raises(InsufficientDataError):
            search_history(frame, self.request(frame), similarity_service=service)


class TestSearchPage:
    def make_page(self, frame):
        from core.data_models import MarketDataset
        from ui.multi_indicator_page import MultiIndicatorPage

        dataset = MarketDataset("测试指数", frame)
        return MultiIndicatorPage(lambda: (dataset,), lambda: SimilarityService())

    def test_search_fills_tab_and_rows(self, qapp):
        page = self.make_page(TestSearch.repeating_frame(count=320))
        page.spin_top_n.setValue(3)
        page.search_history()
        assert page.tabs_results.currentIndex() == 1
        assert page.table_search.rowCount() == 3
        assert page.table_search.item(0, 0).text() == "1"
        assert "区间" in page.table_search.horizontalHeaderItem(1).text()

    def test_selecting_result_loads_detail(self, qapp):
        page = self.make_page(TestSearch.repeating_frame(count=320))
        page.search_history()
        before = page.result
        page._on_search_selected(0, 0)
        assert page.tabs_results.currentIndex() == 0
        assert page.result is not before
        assert page.result is not None and page.result.has_result
        assert page.table_results.rowCount() == len(page.result.scores)
        assert "已载入" in page.label_status.text()

    def test_scope_dates_default_to_full_history(self, qapp):
        frame = TestSearch.repeating_frame()
        page = self.make_page(frame)
        assert page.date_scope_start.date().toString("yyyy-MM-dd") == str(frame.index[0].date())
        assert page.date_scope_end.date().toString("yyyy-MM-dd") == str(frame.index[-1].date())
