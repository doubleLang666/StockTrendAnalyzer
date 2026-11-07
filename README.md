# 特定时间范围内的股票趋势共性分析系统

基于历史行情数据的 Python 桌面分析软件：导入券商导出的 TXT 行情文件，
对不同时间范围的行情曲线做归一化、对齐、可视化与相似度检索。

本软件**不接入实时行情、不预测未来走势、不提供投资建议、不涉及自动交易**，
所有分析结果均由明确的数学方法与程序逻辑产生。

## 目录结构

```
F:\毕业设计\
├── .venv\                     Python 3.14 虚拟环境（已配置）
├── requirements.txt           依赖清单
├── requirements-lock.txt      冻结版本
└── StockTrendAnalyzer\        源码根目录
    ├── main.py                程序入口
    ├── StockTrendAnalyzer.spec PyInstaller 打包配置
    ├── config.json            运行配置（首次运行复制到 exe 旁）
    ├── pytest.ini             测试配置
    ├── app\                   应用协调层（配置、路径、服务注册表、装配）
    ├── core\                  核心层（数据模型、接口、常量、异常）
    ├── modules\               业务模块层（导入/处理/趋势/相似度/导出）
    ├── ui\                    界面层
    ├── tests\                 pytest 测试
    ├── packaging\             打包用版本信息
    ├── tools\                 开发辅助脚本（自检、截图、图标、打包）
    ├── resources\             图标等静态资源
    ├── data\                  用户行情数据目录
    └── exports\               分析结果默认导出目录（运行时创建）
```

依赖方向严格单向：`ui → app → modules → core`，`core` 不依赖任何项目内模块。

## 路径管理

所有路径集中在 `app/paths.py`，源码里没有任何本机绝对路径：

| 函数 | 开发时 | 打包后 |
| --- | --- | --- |
| `application_dir()` | 源码根目录 | exe 所在目录（可写） |
| `bundle_dir()` | 源码根目录 | `sys._MEIPASS`（只读） |
| `resources_dir()` / `resource_path()` | `resources\` | `_MEIPASS\resources\` |
| `config_path()` | 源码根 `config.json` | exe 旁 `config.json` |
| `data_dir()` | `data\` | exe 旁 `data\` |
| `export_dir()` | `exports\` | exe 旁 `exports\` |
| `cache_dir()` | `.cache\` | exe 旁 `.cache\`（Matplotlib 字体缓存） |

打包后 `config.json` 由 `ensure_config_file()` 从包内复制到 exe 旁，之后用户改动都写在这份可写副本里。

## 运行

```powershell
F:\毕业设计\.venv\Scripts\python.exe F:\毕业设计\StockTrendAnalyzer\main.py
```

自检（无界面构建主窗口后退出，用于验证依赖是否齐全）：

```powershell
F:\毕业设计\.venv\Scripts\python.exe F:\毕业设计\StockTrendAnalyzer\main.py --self-test
```

## 测试

```powershell
cd F:\毕业设计\StockTrendAnalyzer
F:\毕业设计\.venv\Scripts\python.exe -m pytest
```

GUI 测试通过 `QT_QPA_PLATFORM=offscreen` 在无窗口模式下运行（见 `tests/conftest.py`）。

## 数据导入（第二阶段已完成）

支持券商交易软件导出的 TXT 行情文件，典型形态：

```
                  创业板指 (399006)

      时间	开盘	最高	最低	收盘	成交量	...
2010/06/01	...	...
#数据来源:通达信
```

处理流程：

1. **编码识别**：依次尝试 UTF-8(BOM)/UTF-8，失败后按中文占比在 GB18030/GBK/BIG5 中选择；
2. **表头定位**：在前 80 行中寻找能映射出至少两个标准字段且含日期的行；
3. **字段映射**：通过 `core/constants.py` 的别名表把「时间/开盘/收盘…」映射到标准字段，
   未识别的列（MA、VOL-TDX、MACD 等）记为"已忽略列"；
4. **单元格转换**：日期按多种格式解析，非法日期整行丢弃并记录；数值非法记为 `NaN`；
5. **校验**：负价格、最高价低于最低价、最高/最低与开收盘冲突、成交量负值、
   缺失值统计、重复日期等，全部以问题清单呈现，**不删除、不填充、不修正数据**。

命令行自检（不启动界面）：

```powershell
cd F:\毕业设计\StockTrendAnalyzer
F:\毕业设计\.venv\Scripts\python.exe tools\check_import.py data\399006.txt
```

## 数据处理（第三阶段已完成）

数据处理页在**不修改原始数据**的前提下，把导入的数据集转换成后续分析用的数据视图：

1. **数据选择**：勾选字段、设置日期范围（默认取数据集自身的起止日期）；
2. **二级数据**：变化率（%）、差分、对数收益率，基于所选字段计算，
   "前一有效值"指前一个非缺失值，首条记录保持为空，前一有效值为 0 时不计算；
3. **归一化**：
   - 起始值归一化：区间内第一个有效值 → 100；
   - Min-Max 归一化：默认映射到 0~1，区间可改；取值恒定时统一映射到下界并给出说明；
4. **时间对齐**：真实交易日期，或相对交易日（起始日 = 第 0 日）；
5. **数据形态**：做了归一化或二级数据即标记为"相对数据"，否则为"绝对数据"。

归一化基准值、原始区间、日期裁剪与丢弃情况都会写入 `ProcessedDataset.provenance`，
并显示在处理结果下方的说明区，便于追溯。

## 趋势分析（第四阶段已完成）

趋势分析页负责"画出来"和"找出相似历史区间"，相似度算法由第五阶段的相似度模块提供。

**基础曲线**

1. 选择数据源（"数据处理"页签生成的处理结果，或原始数据）；
2. 勾选一个或多个字段，选择日期范围；
3. 原始数据可切换绝对/相对显示（不归一化 / 起始值归一化 / Min-Max）与时间轴
   （真实交易日期 / 相对交易日，起始日 = 第 0 日）；处理结果则直接沿用其自身设置，不重复处理；
4. 同时绘制多个字段且为绝对数值时，界面会提示量纲不同、建议归一化（文档 3.3.2）。

**多周期比较**

1. 用"添加当前区间"把若干历史区间加入比较列表；
2. 点击"绘制比较图"，所有区间按当前对齐与显示模式画在同一张图上；
3. 区间长度不同时**不会静默截断**：默认按最短区间截断并在状态栏明确说明，
   取消勾选后按原长度绘制，同样会说明长度差异（文档 3.3.3）。

**历史曲线检索（核心功能，文档 3.3.4）**

1. 在"历史区间检索"分组选择算法（Pearson / RMSE / MAE / DTW / 余弦）与返回条数（默认 3）；
2. 以当前起始/结束日期为目标区间、以第一个勾选字段为目标曲线，点击"检索相似区间"；
3. 结果表展示排名、起止日期、交易日数、算法与指标值，选中某行即可在右侧叠加
   目标曲线与该结果曲线（两者都按首个有效值归一化到 100，便于比较形态）。

检索排除了目标区间自身与高度重叠窗口（`max_overlap_ratio` 默认 0.5），
有效数据点少于 `min_valid_points` 的窗口跳过；相似度函数由注册表注入，
趋势模块内部不实现任何算法。状态栏会记录算法、字段、目标区间与全部参数。

## 相似度分析（第五阶段已完成）

**算法层（`modules/similarity`）**

- 统一接口 + 策略模式：`SimilarityAlgorithm.compare(series_a, series_b, *, min_valid_points, normalize, **params)`
  返回 `SimilarityResult`（算法名、原始值、排序方向、展示值、计算状态、说明、有效点数）；
- 五种算法：`Pearson`（越高越相似，方向 DESC）、`RMSE`、`MAE`（越低越相似，方向 ASC）、
  `DTW`（动态时间规整距离，越低越相似，**允许两条曲线长度不同**）、`余弦相似度`（越大越相似）；
- 统一处理长度不一致（`LENGTH_MISMATCH`，不截断、不补值）、有效点不足（`INSUFFICIENT_DATA`）、
  逐点成对 NaN 掩码、常数曲线（`CONSTANT_SERIES`）、除零与计算失败（`FAILED`），**任何情况都不填 0**；
- 误差类指标在 `normalize=True` 时先按起始值归一化到 100，否则不同价格水平的误差不可比。

**相似度页**

1. 曲线 A / 曲线 B 各自选择数据源（原始数据或处理结果）、字段与日期范围；
2. 长度不同时默认按较短曲线截断并在状态栏注明原长度与截断后长度，取消勾选则返回"长度不一致"；
3. 点击"计算相似度"后，表格列出五种算法的原始值、有效点数与状态说明，图中同时绘制两条归一化曲线；
4. 点击"导出结果…（CSV）"可把整张对比表（含算法、原始值、有效点、状态与说明）导出。

## 结果导出（第六阶段已完成）

**导出模块（`modules/export`）**

- `CsvExporter.export(payload, path)` 按数据类型自动分派：
  - `ProcessedDataset` → 处理后的数据表（行 = 日期或相对交易日，列 = 字段）；
  - `SearchResult` 序列 → 历史区间检索结果（排名、起止日期、交易日、字段、算法、原始值、状态）；
  - `SimilarityResult` 序列 → 相似度对比结果（算法、原始值、有效点、是否先归一化、状态说明）。
- 统一 `utf-8-sig`（带 BOM）编码，Excel 双击即可正确显示中文；
- 参数与来源（数据集、对齐方式、数据形态、归一化、目标区间、算法）写入表格前的 `#` 注释行，
  满足"结果必须可追溯"的要求，同时文件仍是可解析的 CSV；
- 缺失值导出为空单元格，**不填 0**。

**界面入口**

| 页面 | 按钮 | 导出内容 |
|---|---|---|
| 数据处理 | 导出数据…（CSV） | 当前处理结果 |
| 趋势分析 | 导出检索结果…（CSV） | 当前 Top N 相似区间 |
| 相似度分析 | 导出结果…（CSV） | 当前两条曲线的全部算法结果 |

## 打包为可执行程序（PyInstaller）

提供两种打包方式，按用途选择：

| 方式 | 命令 | 输出 | 适用场景 |
|---|---|---|---|
| 目录版 | `pwsh -File tools\build_exe.ps1` | `dist\StockTrendAnalyzer\`（约 200 MB） | 自己用，启动快 |
| 单文件版 | `pwsh -File tools\build_exe.ps1 -OneFile` | `dist-onefile\StockTrendAnalyzer.exe`（约 86 MB） | **发给别人**，只有一个文件 |

等价于直接调用 PyInstaller：

```powershell
cd F:\毕业设计\StockTrendAnalyzer
F:\毕业设计\.venv\Scripts\python.exe -m PyInstaller StockTrendAnalyzer.spec --noconfirm          # 目录版
F:\毕业设计\.venv\Scripts\python.exe -m PyInstaller StockTrendAnalyzer-onefile.spec --noconfirm   # 单文件版
```

目录版输出结构（单目录模式，约 200 MB）：

```
dist\StockTrendAnalyzer\
├── StockTrendAnalyzer.exe   主程序（带图标与版本信息，无控制台窗口）
├── _internal\               PySide6 / Matplotlib / pandas / numpy 等依赖
├── config.json              出厂默认配置（首次运行复制到可写位置）
├── resources\app.ico        窗口与任务栏图标
└── data\                    行情数据目录（运行时自动创建）
```

打包要点：

- **Matplotlib / PySide6**：两者都有官方 hook，`backend_qtagg` 与 Qt 平台插件会被自动收集；
  `StockTrendAnalyzer.spec` 里额外用 `hiddenimports` 显式声明了 Matplotlib 的延迟导入后端与项目自己的子模块，
  避免静态分析漏掉。
- **不能排除 `unittest`**：matplotlib → pyparsing → `pyparsing.testing` 会 `import unittest`，
  早期把它放进 `excludes` 会导致打包后启动即崩（`ModuleNotFoundError: No module named 'unittest'`）。
- **Matplotlib 字体缓存**：`app/paths.py` 的 `set_cache_environment()` 把 `MPLCONFIGDIR`
  指向 exe 旁 `.cache\matplotlib`，否则在只读的解包目录里会报无法写缓存。
- **自检**：`StockTrendAnalyzer.exe --self-test` 会构建主窗口、导入 `data\` 下的行情文件，
  把结果写入 `self_test_report.json` 后退出（退出码 0 表示正常）。
- **不要从压缩软件的临时目录直接运行**：360 压缩等工具双击压缩包内的 exe 时会先解压到
  `%TEMP%\360zip$Temp\360$1\...` 预览目录，`_internal` 往往没有完整解压出来，于是报
  `FileNotFoundError: ...\_internal\base_library.zip`（这是 PyInstaller 引导程序在 Python 启动前报的错，
  程序内部无法捕获）。解决方式二选一：① 先"解压到当前文件夹"再双击；
  ② 直接发**单文件版** `StockTrendAnalyzer.exe`（自解压，不依赖 `_internal` 目录）。

## 数据与结果如何保存

程序本身不保存"会话状态"，所有需要长期保留的内容都以普通文件形式落在 exe（或源码）旁边，
删除/重装程序不会影响这些数据：

| 内容 | 位置 | 说明 |
|---|---|---|
| 原始行情数据 | 用户自选目录，默认 `data\` | 程序只读不写，导入时不会修改原文件 |
| 配置文件 | `config.json` | 窗口尺寸、编码候选、算法与阈值、最近导入的文件列表 |
| 处理/检索/相似度结果 | 用户通过"导出…（CSV）"另存 | 默认建议存到 `exports\`，`utf-8-sig` 编码，Excel 可直接打开 |
| 运行期缓存 | `.cache\` | Matplotlib 字体缓存，可随时删除 |

**重启后数据是否保留**：`import.restore_recent`（默认 `true`）会在启动时读取
`config.json` 中记录的最近导入文件，文件仍存在就自动重新导入并刷新下游页面；
在数据处理页签的处理结果、趋势图属于内存中的临时视图，重新导入后按需重新生成即可。

## 开发阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 一 | 项目结构、统一数据模型、模块接口、基础 GUI、pytest | ✅ 已完成 |
| 二 | TXT 解析、字段映射、数据校验、多文件导入、预览窗口 | ✅ 已完成 |
| 三 | 字段/日期选择、二级数据、归一化、时间对齐、处理窗口 | ✅ 已完成 |
| 四 | 单/多曲线绘制、多周期比较、真实日期与相对交易日对齐 | ✅ 已完成 |
| 五 | Pearson/RMSE/MAE、滑动窗口检索、Top N 排序 | ✅ 已完成 |
| 六 | DTW、余弦相似度、结果导出（CSV） | ✅ 已完成 |

## 与设计文档的偏离记录

1. **Python 版本**：文档写 3.12.x，实际环境为 3.14.6（经用户确认沿用）。
   PySide6 6.12 / pandas 3.1 / numpy 2.5 / matplotlib 3.11 / scipy 1.18 均实测兼容。
2. **虚拟环境位置**：`.venv`、`requirements.txt` 位于 `F:\毕业设计\`（工程根），
   源码位于 `F:\毕业设计\StockTrendAnalyzer\`（文档 4.2 的 StockTrendAnalyzer 根）。
3. **编码识别**：不引入 `chardet`，改为内置候选编码探测，避免改变技术选型。
4. **误差类指标归一化**：历史检索计算 RMSE/MAE 前默认先做起始值归一化
   （`SimilarityConfig.normalize`），否则不同价格水平区间的误差不可比。
5. **重叠窗口剔除**：与目标区间重叠率超过 `max_overlap_ratio`（默认 0.5）的窗口视为自匹配，予以剔除。
6. **指标列处理**：通达信导出文件中的 MA、VOL-TDX、MACD 等列不属于设计文档要求的
   8 个基础字段，导入时记录为"已忽略列"，既不参与分析也不丢弃原始文件。
7. **数据形态自动判定**：文档 3.2.6 允许用户选择绝对/相对数据；实现上改为根据
   "是否归一化 / 是否计算二级数据"自动标记，避免出现"已归一化却标记为绝对数据"的矛盾状态，
   判定结果与理由都会写入处理结果说明。
8. **处理结果不再二次归一化**：趋势页选中"处理结果"时直接使用其数值与对齐方式，
   归一化/对齐下拉框置灰；否则会把起始值归一化再做一次，产生误导。
9. **相似度页的截断策略**：文档要求长度不一致必须明确处理，实现为"默认按较短曲线截断并在
   状态栏注明原长度与截断后长度"，用户取消勾选则不做任何截断、由算法返回 `LENGTH_MISMATCH`；
   截断只发生在界面对齐这一步，算法层永远不截断、不补值。
10. **不同算法结果不混排**：五种算法的原始值取值范围不同（相关系数 / 余弦 ∈ [−1, 1]、误差与 DTW ≥ 0），
    相似度页用同一张表分列展示、不做跨算法排序；历史检索每次只用一种算法，
    排序方向由 `ALGORITHM_DIRECTIONS` 决定（Pearson/余弦降序、RMSE/MAE/DTW 升序）。
11. **DTW 的实现取舍**：距离按"规整路径长度开方后的均方根"归一化（`sqrt(total / (n + m))`），
    使不同长度的区间之间可比；为避免 Python 双层循环，动态规划按反对角线向量化推进；
    单次计算上限 400 万个单元格（`MAX_CELLS = 4_000_000`），超出返回 `FAILED` 并提示缩短区间，
    不静默降级或截断。DTW 是唯一允许两条曲线长度不同的算法。
12. **导出格式**：文档只要求"分析结果导出"，实现为 CSV（`utf-8-sig` 带 BOM，Excel 直接可读），
    参数与来源写在表格前的 `#` 注释行，兼顾"可追溯"与"仍是标准 CSV"；缺失值导出为空单元格。
