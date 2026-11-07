"""命令行导入自检：解析一份券商 TXT 并打印导入报告。

用法（在项目根目录执行）：

    .venv\\Scripts\\python.exe StockTrendAnalyzer\\tools\\check_import.py data\\399006.txt

不传参数时默认检查 ``data/399006.txt``。该脚本只读文件，不做任何写入。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig  # noqa: E402
from modules.data_import.importer import TxtImporter  # noqa: E402


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else PROJECT_ROOT / "data" / "399006.txt"
    importer = TxtImporter(AppConfig.load())
    result = importer.import_file(path)
    report = result.report
    dataset = result.dataset

    print(f"文件      : {path}")
    print(f"名称/代码 : {report.dataset_name} / {report.code or '未识别'}")
    print(f"编码      : {report.encoding}（{report.delimiter_label}分隔，表头第 {report.header_line} 行）")
    print(f"记录      : 导入 {report.imported_rows} 行 / 扫描 {report.total_rows} 行 / 丢弃 {report.dropped_rows} 行")
    print(f"日期范围  : {dataset.frame.index[0].date()} ~ {dataset.frame.index[-1].date()}")
    print(f"字段      : {', '.join(dataset.frame.columns)}")
    print(f"字段映射  : {report.mapping}")
    if report.missing_fields:
        print(f"缺失字段  : {', '.join(report.missing_fields)}")
    if report.ignored_columns:
        print(f"已忽略列  : {', '.join(report.ignored_columns)}")
    print(f"缺失统计  : {dataset.missing_counts()}")
    print(f"耗时      : {report.elapsed_ms:.0f} ms")
    print(f"校验结论  : {report.summary_text()}")
    for issue in report.issues:
        print(f"  - [{issue.severity_label}] {issue.field_label or '-'}：{issue.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
