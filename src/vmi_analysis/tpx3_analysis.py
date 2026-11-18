#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 tpx3_analysis.py  —  TPX3 离子成像数据离线分析流水线
===============================================================================

本文件把 ODL1 / ODL2 两组泵浦-探测实验的全部分析逻辑整合为一个可执行脚本，
按功能分为 10 个区块（见下）。每个区块内部函数自包含，区块之间只通过
__main__ 中的流水线（run_* 函数）串联。

-------------------------------------------------------------------------------
 区块目录
-------------------------------------------------------------------------------
  BLOCK 0  配置与常量            CONFIG / 物理换算 / TOF 峰窗口
  BLOCK 1  文件格式读取          read_header / read_frame / iter_list
  BLOCK 2  数据集扫描            scan_folder / scan_dataset
  BLOCK 3  帧序与时间轴          time_sorted / frame_offset_scan
  BLOCK 4  峰形拟合               fit_gaussian_peak / fit_erf_edge
  BLOCK 5  产率计算              compute_yield
  BLOCK 6  绘图工具              公共样式 / 保存
  BLOCK 7  任务：GMD 诊断          task_gmd_diagnostics
  BLOCK 8  任务：TOF 谱            task_tof_spectrum / task_tof_vs_position
  BLOCK 9  任务：产率曲线           task_yield_curve
  BLOCK 10 任务：上升沿拟合         task_erf_fit
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
 关键约定（务必理解，否则结果会错）
-------------------------------------------------------------------------------
 1. 文件格式：一个 .tpx3 = 一帧 = 一次触发
      偏移 0   int64   秒 (Unix)
      偏移 8   int64   纳秒 (0 <= ns < 1e9)
      偏移 16  double  GMD1 读数
      偏移 24  double  GMD2 读数
      偏移 32  uint64 数据包序列，直到 EOF（无长度前缀、无尾填充）

 2. 数据包按高 4 位分派：
      0xB 像素包：x = dcol + pix//4, y = spix + (pix&3)
                   toa_ns = spidrTime*409600 + CTOA*1.5625
                   tot_ns = TOT*25
      0x6 TDC 包：只有 (p>>56)==0x6F（TDC1 上升沿）有效
      其他       ：忽略

 3. TOF 定义：TOF = toa - tdctime1（ns），触发为时间零点
      注意 tdctime1 在 Go4 中是 ULong64_t，即截断为整数纳秒。

 4. 像素计数器回绕：像素计数器 41 位（2^30 个 spidrTime，26.84 s），
    TDC 32 位（2^32 coarse，107.37 s），像素比 TDC 多回绕 4 倍。
    Go4 的 TYYYEventSource 每个像素只加一次 2^30*25 ns（且需 while 收敛），
    当像素落后触发超过一个回绕周期时会留下负 TOF（ODL1 实测约 45% 的帧）。
    Go4 的 TYYYUnpackProc::addself() 就是"把负 TOF 一直加回正值"，正是
    正确的做法，但其三处调用在源码中被注释掉了。
    本文件的 UNWRAP='signed' 等价于 addself，是默认行为；
    UNWRAP='go4' 可复现原始（有缺陷的）单次展开行为用于对比。

 5. GMD 与相机帧错位：GMD 读数比对应相机帧晚 1 帧，
    即第 i 帧的相机数据应配对 GMD2[i-1]（帧偏移 d = -1）。
    ODL1/ODL2 共 13 个独立文件夹一致确认。由 BLOCK 3 的偏移扫描验证。

 6. 帧顺序必须按文件头时间戳排序，绝不能按文件名：
    Windows 按字符串排序会把 frame_73999 排在 frame_74000 之后。

 7. 时间换算：0.15 mm = 1 ps  =>  t[ps] = position[mm] / 0.15

-------------------------------------------------------------------------------
 用法
-------------------------------------------------------------------------------
    # 全流程（两组数据，全部任务）
    python tpx3_analysis.py all --odl1 D:\\...\\ODL1_ion --odl2 D:\\...\\ODL2_ion

    # 单项任务
    python tpx3_analysis.py gmd    --root <数据集目录> --out <输出目录>
    python tpx3_analysis.py tof    --root <数据集目录> --out <输出目录> --gmd-min 2 --gmd-max 6
    python tpx3_analysis.py yield  --root <数据集目录> --out <输出目录> --gmd-min 2 --gmd-max 6 --shift -1
    python tpx3_analysis.py erf    --csv <产率csv> --fraction 1/3

    目录命名：ODL1 形如 Tr186p4_ion（带 Tr 前缀），ODL2 形如 211p6_ion。
    两种命名都能自动识别。

依赖：numpy, scipy, matplotlib, python-pptx(可选，仅用于生成 PPT)
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# =============================================================================
# BLOCK 0 ── 配置与常量
# =============================================================================
# 所有"魔数"集中在此，方便核对与修改。数值来源见文件头注释和注释内说明。

# ---- 0.1 时间基准 -----------------------------------------------------------
PIXEL_CLK_NS     = 25.0 * 16384.0     # 409600 ns  一个 spidrTime 周期
TOA_ROLLOVER_NS  = 26843545600.0      # 2^30 * 25 ns  像素计数器回绕周期
TDC_ROLLOVER_NS  = 107374182400       # 2^32 * 25 ns  TDC 计数器回绕周期
FINE_NS          = 25.0 / 16.0        # 1.5625 ns  像素精细时间单位
TDC_FINE_NS      = 25.0 / 4096.0      # TDC 精细时间单位
PS_PER_MM        = 1.0 / 0.15         # 6.6667 ps/mm  位移台换算

# ---- 0.2 数据结构 -----------------------------------------------------------
HEADER_BYTES     = 32                 # 每帧元数据头长度
PIXEL_TYPE       = 0xB                # 像素包高 4 位
TDC_TYPE         = 0x6                # TDC 包高 4 位
TDC1_RISING      = 0x6F               # TDC1 上升沿包头字节

# ---- 0.3 TOF 峰窗口（ns）----------------------------------------------------
# 峰位会随延迟位置漂移（ODL2 尤其明显），因此默认使用"跟随窗口"：
# 每个文件夹单独拟合峰位 mu 与宽度 sigma，窗口取 mu ± K*sigma。
PEAK_XE1_GUESS   = 10545.0            # Xe 1+ 峰位初值
PEAK_XE2_GUESS   = 9155.0             # Xe 2+ 峰位初值
PEAK_XE1_RANGE   = (10150.0, 11050.0) # Xe 1+ 拟合搜索范围
PEAK_XE2_RANGE   = (8800.0, 9600.0)   # Xe 2+ 拟合搜索范围
PEAK_WINDOW_K    = 3.0                # 窗口半宽 = K * sigma

# 固定窗口（仅在需要与历史结果对齐时使用）
FIXED_XE1_WINDOW = (10300.0, 10800.0)
FIXED_XE2_WINDOW = (9050.0, 9300.0)

# ---- 0.4 分析默认参数 -------------------------------------------------------
GMD2_DEFAULT     = (2.0, 6.0)         # 默认 GMD2 窗口（nA）
GMD2_ODL2_ALT    = (1.3, 3.0)         # ODL2 推荐窗口（光强偏低）
FRAME_SHIFT      = -1                 # GMD 帧偏移，-1 表示 GMD2[i-1] 配对 frame[i]
UNWRAP_MODE      = 'signed'           # 'signed'(=addself) 或 'go4'(原始单次展开)
MAX_SHIFT_SCAN   = 5                  # 帧偏移扫描范围 ±N
TOF_PLOT_RANGE   = (6000.0, 12500.0)  # TOF 绘图范围（ns）

# ---- 0.5 健壮性阈值 ---------------------------------------------------------
MIN_PIXELS_FOR_FIT = 2000   # 拟合 Xe 峰所需的最少像素数（太少时峰形不可靠）
MIN_FRAMES_WARN    = 200    # 窗口内帧数少于该值时提示统计量偏低


class Tpx3FormatError(Exception):
    """文件格式错误（对应 Go4 中的 TGo4EventErrorException）。"""


# =============================================================================
# BLOCK 1 ── 文件格式读取
# =============================================================================
# 逐帧解析 .tpx3 文件，产出 Frame 对象。
# 本区块已通过 Go4 参考值逐位校验（见 verify_reader()）。

@dataclass
class Frame:
    """一帧相机数据，对应 Go4 中的 TYYYRawEvent（相机分支）。"""
    path: str
    seconds: int                 # Unix 秒
    nanosecond: int              # 纳秒 0..999999999
    gmd1: float                  # GMD1 读数（nA）
    gmd2: float                  # GMD2 读数（nA）
    tdctime1: int                # 整数纳秒（Go4 中为 ULong64_t）
    toa: np.ndarray              # 像素到达时间（ns，已按触发展开）
    tot: np.ndarray              # 像素过阈时间（ns）
    x: np.ndarray                # 像素列
    y: np.ndarray                # 像素行
    n_packets: int = 0
    tdc_raw_ns: float = float('nan')

    @property
    def timestamp(self) -> float:
        """帧时间戳（秒），用于排序。"""
        return self.seconds + self.nanosecond * 1e-9

    @property
    def n_hits(self) -> int:
        return int(self.toa.size)

    def tof(self) -> np.ndarray:
        """TOF = toa - tdctime1（ns），与 Go4 的 TYYYUnpackProc 一致。"""
        return self.toa - float(self.tdctime1)


def read_header(path: str | Path) -> tuple[int, int, float, float]:
    """只读 32 字节元数据头（快，用于只关心 GMD 的场合）。"""
    with open(path, 'rb') as fh:
        head = fh.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES:
        raise Tpx3FormatError(f'{path}: 文件头不足 32 字节')
    sec, nsec, gmd1, gmd2 = __import__('struct').unpack('<qqdd', head)
    if not (0 <= nsec < 1_000_000_000):
        raise Tpx3FormatError(f'{path}: 非法纳秒字段 {nsec}')
    return sec, nsec, gmd1, gmd2


def _unwrap_toa(toa_ns: np.ndarray, tdc_int: int, mode: str) -> np.ndarray:
    """BLOCK 1.1 像素计数器回绕展开。

    mode='go4'    : 复现 TYYYEventSource —— 每个像素只加一次回绕周期
    mode='signed' : 等价 TYYYUnpackProc::addself —— 加够次数使像素进入
                    [tdc-1000, ∞) 窗口（默认，推荐）
    """
    if toa_ns.size == 0:
        return toa_ns
    if mode == 'go4':
        lag = toa_ns < (tdc_int - 1000.0)
        if lag.any():
            toa_ns = toa_ns + lag.astype(np.float64) * TOA_ROLLOVER_NS
        return toa_ns
    if mode == 'signed':
        k = np.maximum(np.ceil((tdc_int - 1000.0 - toa_ns) / TOA_ROLLOVER_NS), 0.0)
        return toa_ns + k * TOA_ROLLOVER_NS
    raise ValueError(f'mode 必须是 go4 或 signed，收到 {mode!r}')


def read_frame(path: str | Path, tdctime_previous: int = 0,
               unwrap: str = UNWRAP_MODE) -> Frame:
    """BLOCK 1.2 读取一帧完整数据。

    tdctime_previous: TDC 展开状态（跨文件列表时传入上一帧的值，单帧读传 0）
    """
    path = str(path)
    sec, nsec, gmd1, gmd2 = read_header(path)
    with open(path, 'rb') as fh:
        fh.seek(HEADER_BYTES)
        body = fh.read()
    if len(body) % 8:
        raise Tpx3FormatError(f'{path}: 数据包不完整（{len(body)} 字节）')

    pkt = np.frombuffer(body, dtype='<u8')
    types = ((pkt >> np.uint64(60)) & np.uint64(0xF)) if pkt.size else np.zeros(0, np.uint8)

    # --- 像素包 ---
    pix = pkt[types == PIXEL_TYPE]
    if pix.size:
        dcol = (pix & np.uint64(0x0FE0000000000000)) >> np.uint64(52)
        spix = (pix & np.uint64(0x001F800000000000)) >> np.uint64(45)
        p3   = (pix & np.uint64(0x0000700000000000)) >> np.uint64(44)
        toa  = (pix >> np.uint64(30)) & np.uint64(0x3FFF)
        tot  = (pix >> np.uint64(20)) & np.uint64(0x3FF)
        ftoa = (pix >> np.uint64(16)) & np.uint64(0xF)
        # 注意：Go4 原文写的是 (~FTOA & 0xF)，不是 (15-FTOA)，
        # 因此 CTOA 恒为 TOA*16+15，1.5625 ns 精细插值实际未生效。
        # 此处保持与 Go4 一致；若要启用精细时间，改为 (np.uint64(15) - ftoa)。
        ctoa = (toa << np.uint64(4)) | (~ftoa & np.uint64(0xF))
        spidr = pix & np.uint64(0xFFFF)
        x = (dcol + (p3 >> np.uint64(2))).astype(np.int32)
        y = (spix + (p3 & np.uint64(3))).astype(np.int32)
        toa_ns = spidr.astype(np.float64) * PIXEL_CLK_NS + ctoa.astype(np.float64) * FINE_NS
        tot_ns = tot.astype(np.float64) * 25.0
    else:
        x = y = np.zeros(0, np.int32)
        toa_ns = tot_ns = np.zeros(0, np.float64)

    # --- TDC 包（只认 TDC1 上升沿 0x6F）---
    tdcs = pkt[types == TDC_TYPE]
    tdcs = tdcs[(tdcs >> np.uint64(56)) == TDC1_RISING]
    if tdcs.size == 0:
        raise Tpx3FormatError(f'{path}: 缺少 TDC1 上升沿包 (0x6F)')

    handle = np.uint64(int(tdcs[-1]))       # Go4 取该帧最后一个 0x6F 包
    coarse = float((handle >> np.uint64(12)) & np.uint64(0xFFFFFFFF))
    fine = int((handle >> np.uint64(5)) & np.uint64(0xF))
    fine = ((fine - 1) << 9) // 12          # 整数除法，与 C++ 一致
    trig_fine = int(handle & np.uint64(0xE00)) | (fine & 0x1FF)
    raw_tdc = coarse * 25.0 + trig_fine * TDC_FINE_NS

    tdc = np.uint64(int(raw_tdc))           # static_cast<ULong64_t>：截断为整数
    while int(tdc) < tdctime_previous:      # TDC 回绕展开
        tdc = tdc + np.uint64(TDC_ROLLOVER_NS)
    tdc_int = int(tdc)

    toa_ns = _unwrap_toa(toa_ns, tdc_int, unwrap)

    return Frame(path=path, seconds=sec, nanosecond=nsec, gmd1=gmd1, gmd2=gmd2,
                 tdctime1=tdc_int, toa=toa_ns, tot=tot_ns, x=x, y=y,
                 n_packets=int(pkt.size), tdc_raw_ns=raw_tdc)


def iter_list(list_path: str | Path, unwrap: str = UNWRAP_MODE) -> Iterable[Frame]:
    """BLOCK 1.3 按 Go4 的 .lst 列表顺序连续读取（TDC 展开状态跨文件延续）。"""
    lp = Path(list_path)
    prev = 0
    for line in lp.read_text(errors='replace').splitlines():
        name = line.strip()
        if not name or name.startswith('#'):
            continue
        p = Path(name)
        if not p.is_absolute():
            p = lp.parent / name
        fr = read_frame(p, tdctime_previous=prev, unwrap=unwrap)
        prev = fr.tdctime1
        yield fr


def verify_reader(explicit: str | Path | None = None) -> bool:
    """BLOCK 1.4 自检：对照 Go4 的参考值（camera_reader_check.cxx）。

    示例文件为小类目录下的 work/camera list first.tpx3。
    可用 --sample 指定，或在脚本同级的若干常见位置查找；找不到则跳过。
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve().parent
    candidates += [
        here / 'camera list first.tpx3',
        here.parent / 'camera list first.tpx3',
        Path('camera list first.tpx3'),
    ]
    # Go4 工程目录（可自行修改）
    for base in (Path(r'Z:\Agents_space\deepseekHerness\tpx3'
                      r'\tpx3_go4_online_bakGMD\work'),):
        candidates.append(base / 'camera list first.tpx3')
    sample = next((p for p in candidates if p.exists()), None)
    if sample is None:
        print('[verify] 跳过：未找到示例文件 camera list first.tpx3 '
              '（可用 --sample 指定路径）')
        return True
    print(f'[verify] 使用示例: {sample}')
    f = read_frame(sample)
    checks = [
        ('tdctime1', f.tdctime1, 5515451938),
        ('首像素 x', int(f.x[0]), 118),
        ('首像素 y', int(f.y[0]), 142),
        ('首像素 toa', float(f.toa[0]), 5515460207.8125),
        ('toa 最小', float(f.toa.min()), 5515460168.75),
        ('toa 最大', float(f.toa.max()), 5515462801.5625),
        ('像素数', f.n_hits, 144),
    ]
    ok = True
    for name, got, want in checks:
        good = abs(got - want) < 1e-6 if isinstance(want, float) else got == want
        ok &= good
        print(f'[verify] {name:<10} {got!r:>22} {"OK" if good else f"FAIL (want {want})"}')
    print('[verify]', 'PASS' if ok else 'FAIL')
    return ok


# =============================================================================
# BLOCK 2 ── 数据集扫描
# =============================================================================
# 从一个数据集目录（内含若干延迟位置文件夹）扫描出所有帧的轻量记录。
# 只解析文件头 + 统计像素数，速度快；需要 TOF 时才用 read_frame 完整解析。

_FOLDER_ODL1 = re.compile(r'^Tr(\d+)(?:p(\d+))?(?:_\d+)?_ion$')
_FOLDER_ODL2 = re.compile(r'^(\d+)(?:p(\d+))?(?:_\d+)?_ion$')
_POS_IN_NAME = re.compile(r'odl_pos_([0-9.]+)_frame_(\d+)')

SKIP_DIRS = {'tpx3_py', 'results', '__pycache__', 'tpx3_analysis'}


def parse_delay_from_name(name: str) -> float | None:
    """BLOCK 2.1 从文件夹名解析延迟位置（mm）。

    Tr186p4_ion -> 186.4   (ODL1，带 Tr 前缀)
    211p6_ion   -> 211.6   (ODL2，无前缀)
    220_ion     -> 220.0
    Tr204p4_2_ion -> 204.4 (重复测量后缀 _2 被忽略)
    """
    m = _FOLDER_ODL1.match(name) or _FOLDER_ODL2.match(name)
    if not m:
        return None
    whole, frac = m.group(1), m.group(2)
    return float(whole) if frac is None else float(f'{whole}.{frac}')


@dataclass
class FrameRecord:
    """帧的轻量记录（不含 TOF 数组）。"""
    path: Path
    name: str
    t: float            # 时间戳（秒）
    gmd1: float         # nA
    gmd2: float         # nA
    npix: int
    size: int           # 文件字节数
    header_pos: float   # 文件名中的 odl_pos 值（mm），无则为 nan


def scan_folder(folder: Path) -> list[FrameRecord]:
    """BLOCK 2.2 扫描一个延迟位置文件夹，返回按时间戳排序的记录列表。"""
    return scan_files(folder.glob('*.tpx3'))


def scan_files(paths: Iterable[Path]) -> list[FrameRecord]:
    recs: list[FrameRecord] = []
    for f in paths:
        try:
            size = f.stat().st_size
            raw = f.read_bytes()
        except OSError:
            continue
        if len(raw) < 40 or (len(raw) - HEADER_BYTES) % 8:
            continue
        try:
            sec, nsec, g1, g2 = read_header(f)
        except Tpx3FormatError:
            continue
        pkt = np.frombuffer(raw[HEADER_BYTES:], dtype='<u8')
        npix = int((((pkt >> np.uint64(60)) & np.uint64(0xF)) == PIXEL_TYPE).sum())
        m = _POS_IN_NAME.search(f.name)
        recs.append(FrameRecord(
            path=f, name=f.name, t=sec + nsec * 1e-9,
            gmd1=g1 * 1e9, gmd2=g2 * 1e9, npix=npix, size=size,
            header_pos=float(m.group(1)) if m else float('nan')))
    recs.sort(key=lambda r: r.t)        # 必须按时间戳，不能按文件名
    return recs


@dataclass
class Position:
    """一个延迟位置（= 一个文件夹）的完整信息。"""
    folder: Path
    name: str
    delay: float                       # 权威位置（mm）
    records: list[FrameRecord]

    @property
    def n_frames(self) -> int:
        return len(self.records)

    @property
    def t_ps(self) -> float:
        return self.delay * PS_PER_MM

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (gmd1, gmd2, npix) 数组，均已按时间排序。"""
        return (np.array([r.gmd1 for r in self.records]),
                np.array([r.gmd2 for r in self.records]),
                np.array([r.npix for r in self.records], dtype=float))


def scan_dataset(root: Path) -> list[Position]:
    """BLOCK 2.3 扫描整个数据集目录，返回按延迟位置排序的 Position 列表。"""
    out: list[Position] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in SKIP_DIRS:
            continue
        name_pos = parse_delay_from_name(d.name)
        if name_pos is None:
            print(f'[scan] 跳过无法解析名称的目录: {d.name}')
            continue
        recs = scan_folder(d)
        if not recs:
            print(f'[scan] 跳过空目录: {d.name}')
            continue
        # 优先用文件名中的 odl_pos 中位数作为权威位置
        hp = np.array([r.header_pos for r in recs])
        hp = hp[np.isfinite(hp)]
        pos = float(np.median(hp)) if hp.size else name_pos
        out.append(Position(folder=d, name=d.name, delay=pos, records=recs))
    out.sort(key=lambda p: p.delay)
    return out


def ensure_tof(pos: Position, unwrap: str = UNWRAP_MODE) -> list[np.ndarray]:
    """BLOCK 2.4 为位置内每一帧计算 TOF 数组（耗时的完整解析）。"""
    return [read_record_tof(r, unwrap) for r in pos.records]


def read_record_tof(record: FrameRecord, unwrap: str = UNWRAP_MODE) -> np.ndarray:
    try:
        return read_frame(record.path, unwrap=unwrap).tof()
    except Tpx3FormatError:
        return np.zeros(0)


# =============================================================================
# BLOCK 3 ── 帧序与时间轴
# =============================================================================

def frame_offset_scan(gmd2: np.ndarray, npix: np.ndarray,
                      max_shift: int = MAX_SHIFT_SCAN) -> list[dict]:
    """BLOCK 3.1 扫描 GMD 与相机帧的错位，找出相关性最强的偏移 d。

    配对方式：corr(GMD2[i], npix[i+d])。若采集时间戳错位一帧，
    则某个非零 d 的相关性会显著高于 d=0。

    返回 [{'d':, 'r':, 'rho':, 'n':}, ...]，按 d 升序。
    """
    from scipy.stats import pearsonr, spearmanr
    rows = []
    n = gmd2.size
    for d in range(-max_shift, max_shift + 1):
        if d >= 0:
            a, b = gmd2[:n - d], npix[d:]
        else:
            a, b = gmd2[-d:], npix[:n + d]
        ok = (a > 0) & (b > 0)
        if ok.sum() < 10:
            rows.append(dict(d=d, r=np.nan, rho=np.nan, n=int(ok.sum())))
            continue
        rows.append(dict(d=d,
                         r=float(pearsonr(a[ok], b[ok])[0]),
                         rho=float(spearmanr(a[ok], b[ok])[0]),
                         n=int(ok.sum())))
    return rows


def pair_counts(gmd2: np.ndarray, tofs: Sequence[np.ndarray],
                gmd_lo: float, gmd_hi: float, shift: int,
                window) -> tuple[int, int, int]:
    """BLOCK 3.2 按 GMD 窗口和帧偏移统计帧数与 Xe 计数。

    语义：GMD2[j] 管辖第 (j - shift) 帧的像素。
      shift = -1 -> GMD2[j] 配对 frame[j+1]（即 frame[i] 用 GMD2[i-1]）

    返回 (窗口内帧数, Xe1 计数, Xe2 计数)。窗口为 (lo, hi) 或 None（不筛选）。
    """
    n = gmd2.size
    n_win = xe1 = xe2 = 0
    for j in range(n):
        if window is not None:
            if not (gmd_lo <= gmd2[j] <= gmd_hi):
                continue
        target = j - shift
        if not (0 <= target < n):
            continue
        n_win += 1
        t = tofs[target]
        if t.size == 0:
            continue
        if window is not None:
            (lo1, hi1), (lo2, hi2) = window
            xe1 += int(((t >= lo1) & (t < hi1)).sum())
            xe2 += int(((t >= lo2) & (t < hi2)).sum())
    return n_win, xe1, xe2


# =============================================================================
# BLOCK 4 ── 峰形拟合
# =============================================================================

def _gauss(x, a, mu, sig):
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def fit_gaussian_peak(x: np.ndarray, lo: float, hi: float,
                      guess: float, nbins: int = 60) -> dict | None:
    """BLOCK 4.1 在 [lo,hi] 内拟合高斯峰，返回 mu / sigma / 窗口。

    返回 None 表示统计量不足或拟合失败。
    """
    from scipy.optimize import curve_fit
    sub = x[(x >= lo) & (x <= hi)]
    if sub.size < 200:
        return None
    counts, edges = np.histogram(sub, bins=nbins)
    if counts.sum() < 200:
        return None
    cen = 0.5 * (edges[1:] + edges[:-1])
    try:
        p, _ = curve_fit(_gauss, cen, counts, p0=[counts.max(), guess, 70.0],
                         maxfev=40000)
    except Exception:
        return None
    mu, sig = float(p[1]), abs(float(p[2]))
    if sig <= 0 or not np.isfinite(sig):
        return None
    return dict(mu=mu, sigma=sig,
                window=(mu - PEAK_WINDOW_K * sig, mu + PEAK_WINDOW_K * sig),
                n_in_range=int(sub.size))


def fit_xe_window(tof_all: np.ndarray) -> tuple[tuple, tuple, dict]:
    """BLOCK 4.2 对全部像素拟合 Xe1+/Xe2+ 峰，返回两个跟随窗口与峰参数。"""
    f1 = fit_gaussian_peak(tof_all, *PEAK_XE1_RANGE, PEAK_XE1_GUESS)
    f2 = fit_gaussian_peak(tof_all, *PEAK_XE2_RANGE, PEAK_XE2_GUESS)
    if f1 is None or f2 is None:
        raise RuntimeError('Xe 峰拟合失败：像素数不足或峰形异常')
    return f1['window'], f2['window'], dict(xe1=f1, xe2=f2)


def _erf_step(t, b, a, t0, tau):
    """带偏移的误差函数阶跃：b + a/2 * [1 + erf((t-t0)/(sqrt2*tau))]。"""
    from scipy.special import erf
    return b + a * 0.5 * (1.0 + erf((t - t0) / (np.sqrt(2.0) * tau)))


ERF_BOUNDS = ([-np.inf, 0, -1e4, 0.3], [np.inf, np.inf, 1e5, 5000.0])


def fit_erf_edge(t: np.ndarray, y: np.ndarray, err: np.ndarray | None = None,
                 p0: Sequence[float] | None = None) -> dict:
    """BLOCK 4.3 误差函数上升沿拟合，返回 b/a/t0/tau 及其误差。

    tau 为高斯时间常数；10-90% 上升时间 = 2.5631*tau。fraction=1.0 可复现原始(有缺陷的)单次展开行为用于对比。
    """
    from scipy.optimize import curve_fit
    if p0 is None:
        b0 = float(np.median(y[:3]))
        a0 = float(max(np.median(y[3:min(10, y.size)]) - b0, 1.0))
        grad = np.gradient(y) if y.size > 3 else np.zeros_like(y)
        t00 = float(t[int(np.argmax(grad))])
        p0 = [b0, a0, t00, 10.0]
    popt, pcov = curve_fit(_erf_step, t, y, p0=p0, sigma=err,
                           absolute_sigma=err is not None,
                           bounds=ERF_BOUNDS, maxfev=200000)
    perr = np.sqrt(np.diag(pcov))
    # absolute_sigma=True 时协方差按残差 abs(resid)/sigma 的整体尺度缩放，
    # 当残差远大于 sigma 时会得到天文数字（实际即"无法确定"）。
    # 因此把下限截到协方差的"按 sigma 归一"版本：若缩放后数值超过
    # 参数本身的合理范围，就报告不可确定而非伪造一个误差。
    resid = y - _erf_step(t, *popt)
    dof = max(y.size - len(popt), 1)
    if err is not None:
        chi2 = float(np.sum((resid / err) ** 2))
        scale = chi2 / dof
        # 用 chi2/ndof 缩放得到"诚实"误差；若仍>参数量级则判为不可确定
        perr_eff = perr * np.sqrt(dof / max(chi2, 1e-12))
    else:
        chi2 = float('nan')
        scale = 1.0
        perr_eff = perr
    b, a, t0, tau = popt
    # 可确定性判定：误差 > 2 倍参数值，视为模型对该数据集不可辨识
    span = float(t.max() - t.min())
    identifiable = (np.all(np.isfinite(perr)) and
                    perr[2] < 2 * max(abs(t0), 1.0) and
                    perr[3] < 2 * abs(tau) and tau < span / 3)
    return dict(
        b=b, a=a, t0=t0, tau=tau,
        b_err=perr[0], a_err=perr[1],
        t0_err=perr[2], tau_err=perr[3],
        b_err_scaled=perr_eff[0], a_err_scaled=perr_eff[1],
        t0_err_scaled=perr_eff[2], tau_err_scaled=perr_eff[3],
        identifiable=bool(identifiable), span=span,
        rise_10_90=2.5631 * tau, fwhm=2.3548 * tau,
        chi2red=chi2 / dof if np.isfinite(chi2) else float('nan'),
        ndof=dof, resid=resid, params=tuple(popt))


def bootstrap_erf(t: np.ndarray, y: np.ndarray, err: np.ndarray,
                  p0, nboot: int = 4000, seed: int = 20261117) -> np.ndarray:
    """BLOCK 4.4 自助法重采样，返回 tau 的分布（用于给出更诚实的误差）。"""
    from scipy.optimize import curve_fit
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, t.size, t.size)
        try:
            p, _ = curve_fit(_erf_step, t[idx], y[idx], p0=p0,
                             sigma=err[idx], absolute_sigma=True,
                             bounds=ERF_BOUNDS, maxfev=50000)
            out.append(p[3])
        except Exception:
            continue
    return np.array(out)


# =============================================================================
# BLOCK 5 ── 产率计算
# =============================================================================

def compute_yield(pos: Position, gmd_window: tuple[float, float] | None,
                  shift: int = FRAME_SHIFT,
                  use_tracked: bool = True,
                  unwrap: str = UNWRAP_MODE, *, tofs: Sequence[np.ndarray] | None = None) -> dict:
    """BLOCK 5.1 计算单个延迟位置的 Xe1+/Xe2+ 产率。

    gmd_window=None 表示不筛选（分母 = 全部帧）。
    分母为"GMD2 落在窗口内的帧数"，分子为这些帧所管辖的相机帧中的 Xe 计数。
    """
    if tofs is None:
        tofs = ensure_tof(pos, unwrap=unwrap)
    all_tof = np.concatenate([t for t in tofs if t.size]) if tofs else np.zeros(0)
    if all_tof.size < MIN_PIXELS_FOR_FIT:
        return dict(valid=False, name=pos.name, delay=pos.delay,
                    reason=f'像素不足 ({all_tof.size} < {MIN_PIXELS_FOR_FIT})')

    if use_tracked:
        try:
            w1, w2, peaks = fit_xe_window(all_tof)
        except RuntimeError as exc:
            return dict(valid=False, name=pos.name, delay=pos.delay,
                        reason=str(exc))
    else:
        w1, w2 = FIXED_XE1_WINDOW, FIXED_XE2_WINDOW
        peaks = {}

    g1, g2, _ = pos.arrays()
    if gmd_window is None:
        # 不筛选：每帧都计入，不需要偏移配对
        n_win = g2.size
        xe1 = sum(int(((t >= w1[0]) & (t < w1[1])).sum()) for t in tofs if t.size)
        xe2 = sum(int(((t >= w2[0]) & (t < w2[1])).sum()) for t in tofs if t.size)
    else:
        n_win, xe1, xe2 = pair_counts(g2, tofs, gmd_window[0], gmd_window[1],
                                      shift, (w1, w2))

    if n_win == 0:
        return dict(valid=False, name=pos.name, delay=pos.delay)

    # 泊松误差：分子计数开方；比值的误差按 R*sqrt(1/N1+1/N2) 传播
    y1, y2 = xe1 / n_win, xe2 / n_win
    e1, e2 = np.sqrt(xe1) / n_win, np.sqrt(xe2) / n_win
    ratio = xe1 / max(xe2, 1e-9)
    ratio_err = ratio * np.sqrt(1 / max(xe1, 1) + 1 / max(xe2, 1))
    frac = 100.0 * xe1 / max(xe1 + xe2, 1)

    return dict(valid=True, name=pos.name, folder=str(pos.folder),
                delay=pos.delay, t_ps=pos.t_ps,
                n_frames=pos.n_frames, n_window=n_win,
                xe1=xe1, xe2=xe2, y1=y1, y2=y2, e1=e1, e2=e2,
                ratio=ratio, ratio_err=ratio_err, frac=frac,
                win1=w1, win2=w2,
                xe1_mu=peaks.get('xe1', {}).get('mu', float('nan')),
                xe2_mu=peaks.get('xe2', {}).get('mu', float('nan')),
                xe1_sig=peaks.get('xe1', {}).get('sigma', float('nan')),
                xe2_sig=peaks.get('xe2', {}).get('sigma', float('nan')))


def compute_yield_curve(root: Path, gmd_window, shift: int = FRAME_SHIFT,
                        use_tracked: bool = True,
                        unwrap: str = UNWRAP_MODE,
                        verbose: bool = True) -> list[dict]:
    """BLOCK 5.2 对数据集中所有延迟位置计算产率，返回按延迟排序的列表。"""
    positions = scan_dataset(root)
    rows = []
    for p in positions:
        r = compute_yield(p, gmd_window, shift=shift,
                          use_tracked=use_tracked, unwrap=unwrap)
        if not r.get('valid'):
            print(f'[yield] 跳过 {p.name}: {r.get("reason", "数据不足")}')
            continue
        if r['n_window'] < MIN_FRAMES_WARN:
            print(f'[yield] 提示 {p.name}: 窗口内仅 {r["n_window"]} 帧，'
                  f'统计量偏低')
        rows.append(r)
        if verbose:
            print(f'[yield] {r["name"]:<14} {r["delay"]:8.3f} mm '
                  f'{r["t_ps"]:8.1f} ps  frames={r["n_window"]:5d}  '
                  f'Xe1+={r["xe1"]:8d}  Xe2+={r["xe2"]:7d}  '
                  f'y1={r["y1"]:8.2f}  y2={r["y2"]:8.3f}')
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    """BLOCK 5.3 产率结果写 CSV。"""
    if not rows:
        print('[csv] 无数据可写')
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'[csv] wrote {path}')


def load_yield_csv(path: Path) -> dict:
    """BLOCK 5.4 读回产率 CSV，返回按延迟排序的 numpy 字典。

    兼容两种列名：本脚本输出 'name'，早期脚本输出 'folder'。
    """
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f'{path}: 空 CSV')
    rows.sort(key=lambda r: float(r['delay']))
    key = 'name' if 'name' in rows[0] else ('folder' if 'folder' in rows[0] else None)
    d = {key: np.array([r[key] for r in rows])} if key else {}
    for k in rows[0]:
        if k in ('name', 'folder', 'win1', 'win2'):
            continue
        try:
            d[k] = np.array([float(r[k]) for r in rows])
        except (ValueError, TypeError):
            pass
    return d


# =============================================================================
# BLOCK 6 ── 绘图工具
# =============================================================================

C_RED, C_BLUE, C_GREEN, C_GREY = '#d62728', '#1f77b4', '#2ca02c', '#999999'


def _mpl():
    """BLOCK 6.1 以无界面后端导入 matplotlib，并配置中文字体。

    图上大量中文标注，若不配置中文字体，matplotlib 会退化成方框，
    并抛出大量 "Font 'default' does not have a glyph" 警告。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not matplotlib.rcParams.get('font.sans-serif') or \
            'Microsoft YaHei' not in matplotlib.rcParams['font.sans-serif']:
        matplotlib.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
            'Source Han Sans SC', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    return plt


def _mm_to_ps(mm):
    return np.asarray(mm) / 0.15


def _ps_to_mm(ps):
    return np.asarray(ps) * 0.15


def add_time_axis(ax, which='top'):
    """BLOCK 6.2 在 mm 轴上叠加 ps 副坐标轴（或反向）。"""
    if which == 'top':
        sec = ax.secondary_xaxis('top', functions=(_mm_to_ps, _ps_to_mm))
        sec.set_xlabel('delay  [ps]', fontsize=9)
    else:
        sec = ax.secondary_xaxis('top', functions=(_ps_to_mm, _mm_to_ps))
        sec.set_xlabel('stage position  [mm]', fontsize=9)
    return sec


def save_fig(fig, path: Path, dpi: int = 140) -> None:
    """BLOCK 6.3 保存图并关闭。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    _mpl().close(fig)
    print(f'[fig] wrote {path}')


def _xy(rows, key):
    x = np.array([r['delay'] for r in rows])
    t = np.array([r['t_ps'] for r in rows])
    return x, t, np.array([r[key] for r in rows])


# =============================================================================
# BLOCK 7 ── 任务：GMD 诊断
# =============================================================================
# 检查 GMD 读数是否可用：GMD1/GMD2 线性关系、GMD2 分布、帧大小相关性。

def task_gmd_diagnostics(root: Path, out: Path, sample: str | None = None) -> None:
    """BLOCK 7.1 位置内 GMD 诊断（默认选帧数最多的文件夹）。"""
    plt = _mpl()
    positions = scan_dataset(root)
    if not positions:
        print('[gmd] 无数据')
        return
    if sample:
        cand = [p for p in positions if p.name == sample]
        pos = cand[0] if cand else max(positions, key=lambda p: p.n_frames)
    else:
        pos = max(positions, key=lambda p: p.n_frames)

    g1, g2, npix = pos.arrays()
    size = np.array([r.size for r in pos.records], dtype=float) / 1024.0
    ok = (g1 > 0) & (g2 > 0)
    g1, g2, npix, size = g1[ok], g2[ok], npix[ok], size[ok]

    # --- GMD1 vs GMD2 线性 ---
    from scipy.stats import pearsonr, spearmanr, linregress
    lr = linregress(g2, g1)
    r = pearsonr(g2, g1)[0]
    resid = g1 - (lr.intercept + lr.slope * g2)
    print(f'[gmd] {pos.name}: {g1.size} 帧')
    print(f'[gmd] GMD1/GMD2  斜率={lr.slope:.4f} 截距={lr.intercept:.4f} '
          f'r={r:.4f} R2={r**2:.4f} 残差RMS={resid.std():.4f} '
          f'({100*resid.std()/g1.mean():.1f}%)')

    # --- 帧大小 vs GMD2 ---
    rho_size = spearmanr(g2, size)[0]
    rho_pix = spearmanr(npix, size)[0]
    print(f'[gmd] 帧大小 vs GMD2 Spearman={rho_size:.3f}')
    print(f'[gmd] 帧大小 vs 像素数 Spearman={rho_pix:.3f} (应≈1.000)')

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.hexbin(g2, g1, gridsize=55, cmap='viridis', mincnt=1, bins='log')
    xx = np.linspace(g2.min(), g2.max(), 10)
    ax1.plot(xx, lr.intercept + lr.slope * xx, 'r-', lw=1.5,
             label=f'斜率 {lr.slope:.3f}, 截距 {lr.intercept:.3f}')
    ax1.set_xlabel('GMD2  [nA]'); ax1.set_ylabel('GMD1  [nA]')
    ax1.set_title('GMD1 vs GMD2 线性关系')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)
    ax1.text(0.98, 0.04, f'r = {r:.4f}\nN = {g1.size}', transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=9,
             bbox=dict(fc='white', ec='0.7'))

    ax2.hexbin(g2, size, gridsize=55, cmap='viridis', mincnt=1, bins='log')
    ax2.set_xlabel('GMD2  [nA]'); ax2.set_ylabel('帧大小  [kB]')
    ax2.set_title(f'帧大小 vs GMD2 (Spearman {rho_size:.3f})')
    ax2.grid(alpha=0.2)

    ax3.hexbin(npix, size, gridsize=55, cmap='viridis', mincnt=1, bins='log')
    ax3.set_xlabel('像素数/帧'); ax3.set_ylabel('帧大小  [kB]')
    ax3.set_title(f'帧大小 vs 像素数 (Spearman {rho_pix:.3f})')
    ax3.grid(alpha=0.2)

    ax4.hist(resid, bins=70, color=C_BLUE, edgecolor='white')
    ax4.set_xlabel('残差 GMD1 − 拟合  [nA]'); ax4.set_ylabel('帧数')
    ax4.set_title(f'线性残差 RMS={resid.std():.4f} nA')
    ax4.grid(alpha=0.25)

    fig.suptitle(f'GMD 诊断 — {pos.name}', fontsize=13)
    save_fig(fig, out / f'{pos.name}_GMD_diagnostics.png')

    # --- GMD2 分布 ---
    fig2, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, logy in ((a, False), (b, True)):
        ax.hist(g2, bins=80, color=C_BLUE, edgecolor='white', lw=0.4)
        ax.axvline(np.median(g2), color=C_RED, ls='--', lw=1.6,
                   label=f'中位数 {np.median(g2):.3f} nA')
        ax.axvline(g2.mean(), color=C_GREEN, ls='-.', lw=1.6,
                   label=f'均值 {g2.mean():.3f} nA')
        ax.set_xlabel('GMD2  [nA]'); ax.set_ylabel('帧数' + (' (log)' if logy else ''))
        ax.legend(frameon=False); ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale('log')
    a.set_title(f'GMD2 分布 — {pos.name} ({g1.size} 帧)')
    b.set_title('同一数据，对数纵轴')
    fig2.tight_layout()
    save_fig(fig2, out / f'{pos.name}_GMD2_hist.png')


# =============================================================================
# BLOCK 8 ── 任务：TOF 谱
# =============================================================================

def select_tof_frames(gmd2: np.ndarray, gmd_window: tuple[float, float], shift: int) -> np.ndarray:
    keep = np.zeros(gmd2.size, bool)
    for j in range(gmd2.size):
        if gmd_window[0] <= gmd2[j] <= gmd_window[1]:
            target = j - shift
            if 0 <= target < gmd2.size:
                keep[target] = True
    return keep


def task_tof_spectrum(pos: Position, out: Path,
                      gmd_window: tuple[float, float] | None = GMD2_DEFAULT,
                      shift: int = FRAME_SHIFT, bins: int = 1200) -> None:
    """BLOCK 8.1 单个位置的 TOF 谱（可带 GMD 筛选），含峰拟合标注。"""
    plt = _mpl()
    tofs = ensure_tof(pos)
    g1, g2, _ = pos.arrays()
    all_tof = np.concatenate([t for t in tofs if t.size]) if tofs else np.zeros(0)
    if all_tof.size < MIN_PIXELS_FOR_FIT:
        print(f'[tof] {pos.name}: 像素不足 ({all_tof.size})')
        return

    # 峰参数用全部像素拟合（统计量最大），筛选只影响绘制的直方图
    try:
        w1, w2, peaks = fit_xe_window(all_tof)
    except RuntimeError as exc:
        print(f'[tof] {pos.name}: {exc}')
        return

    if gmd_window is None:
        sel = all_tof
        nframes = pos.n_frames
    else:
        n_win, _, _ = pair_counts(g2, tofs, *gmd_window, shift, None)
        keep = select_tof_frames(g2, gmd_window, shift)
        if not keep.any():
            print(f'[tof] {pos.name}: GMD 窗口内没有帧，改用全部像素')
            sel = all_tof
            nframes = pos.n_frames
        else:
            sel = np.concatenate([t for t, k in zip(tofs, keep) if k and t.size]) \
                if keep.any() else np.zeros(0)
            nframes = n_win

    print(f'[tof] {pos.name}: 全部像素 {all_tof.size}, '
          f'筛选后 {sel.size} ({nframes} 帧)')
    print(f'[tof]   Xe1+ mu={peaks["xe1"]["mu"]:.1f} ns  '
          f'Xe2+ mu={peaks["xe2"]["mu"]:.1f} ns')

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True,
                                  gridspec_kw={'height_ratios': [2.2, 1]})
    lo, hi = TOF_PLOT_RANGE
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(all_tof, bins=edges, histtype='step', lw=0.9, color='#b0b0b0',
            label=f'全部 ({all_tof.size} 像素)')
    ax.hist(sel, bins=edges, histtype='stepfilled', lw=1.2, alpha=0.55,
            color=C_RED, label=f'筛选后 ({sel.size} 像素, {nframes} 帧)')
    ax.set_yscale('log'); ax.grid(alpha=0.25, which='both')
    ax.set_ylabel('像素数 / bin (log)')
    win_txt = '不筛选' if gmd_window is None else f'GMD2 {gmd_window[0]:g}-{gmd_window[1]:g} nA'
    ax.set_title(f'TOF 谱 — {pos.name}  ({win_txt}, 峰跟随窗口 μ±{PEAK_WINDOW_K:g}σ)')
    for lab, pk, col in (('Xe$^{1+}$', peaks['xe1'], C_RED),
                         ('Xe$^{2+}$', peaks['xe2'], C_BLUE)):
        ax.axvline(pk['mu'], color=col, ls='--', lw=1.0, alpha=0.8)
        ax.annotate(f'{lab}\n{pk["mu"]:.0f} ns', (pk['mu'], 0.92),
                    xycoords=('data', 'axes fraction'), ha='center',
                    fontsize=9, color=col)
    ax.legend(frameon=False, loc='upper left', fontsize=9)

    ax2.hist(sel, bins=edges, histtype='stepfilled', lw=1.0, color=C_RED, alpha=0.75)
    ax2.set_xlabel('TOF  [ns]'); ax2.set_ylabel('像素数 / bin')
    ax2.set_xlim(lo, hi); ax2.grid(alpha=0.25)
    fig.tight_layout()
    tag = 'all' if gmd_window is None else f'{gmd_window[0]:g}-{gmd_window[1]:g}nA'
    save_fig(fig, out / f'{pos.name}_TOF_{tag}.png')


def task_tof_vs_position(root: Path, out: Path,
                         gmd_window: tuple[float, float] | None = None,
                         bin_width: float = 2.0) -> None:
    """BLOCK 8.2 全部位置的 TOF 随位置变化（对数色标），另加归一化版。"""
    plt = _mpl()
    from matplotlib.colors import LogNorm
    positions = scan_dataset(root)
    if not positions:
        print('[tofpos] 无数据')
        return
    edges = np.arange(TOF_PLOT_RANGE[0], TOF_PLOT_RANGE[1] + bin_width, bin_width)
    mat, poss = [], []
    for p in positions:
        t = np.concatenate([x for x in ensure_tof(p) if x.size]) \
            if p.n_frames else np.zeros(0)
        if t.size == 0:
            continue
        h, _ = np.histogram(t, bins=edges)
        mat.append(h); poss.append(p.delay)
    if not mat:
        print('[tofpos] 无有效数据')
        return
    mat = np.array(mat, float)
    poss = np.array(poss)
    dp = np.diff(poss)
    ylo = poss[0] - (dp[0] / 2 if dp.size else 1)
    yhi = poss[-1] + (dp[-1] / 2 if dp.size else 1)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'hspace': 0.25})
    im = ax.imshow(np.ma.masked_less(mat, 0.5).T, aspect='auto', origin='lower',
                   extent=[ylo, yhi, edges[0], edges[-1]], cmap='inferno',
                   norm=LogNorm(vmin=1, vmax=max(mat.max(axis=1).max(), 2)),
                   interpolation='nearest')
    ax.set_xlabel('位移台位置  [mm]'); ax.set_ylabel('TOF  [ns]')
    ax.set_title('TOF 随位置变化 (对数色标，绝对计数)', fontsize=11)
    fig.colorbar(im, ax=ax, pad=0.012).set_label('像素数 / bin (log)')
    add_time_axis(ax)

    norm = mat / np.maximum(mat.max(axis=1, keepdims=True), 1)
    im2 = ax2.imshow(np.ma.masked_less(norm, 1e-6).T, aspect='auto', origin='lower',
                     extent=[ylo, yhi, edges[0], edges[-1]], cmap='viridis',
                     norm=LogNorm(vmin=1e-5, vmax=1.0), interpolation='nearest')
    ax2.set_xlabel('位移台位置  [mm]'); ax2.set_ylabel('TOF  [ns]')
    ax2.set_title('同一数据，每行归一化到自身最大值 (对数色标)', fontsize=11)
    fig.colorbar(im2, ax=ax2, pad=0.012).set_label('相对该位置最大值 (log)')
    add_time_axis(ax2)
    save_fig(fig, out / 'TOF_vs_position_log.png', dpi=135)


# =============================================================================
# BLOCK 9 ── 任务：产率曲线
# =============================================================================

def task_yield_curve(rows: list[dict], out: Path, label: str,
                     which: str = 'both') -> None:
    """BLOCK 9.1 画产率曲线。

    which = 'position' | 'time' | 'both'
    产率图与比值图分开画，避免 mm 与 ps 两种横轴混在同一张图上
    （混用会导致共享横轴被撑大、数据挤在一侧）。
    """
    plt = _mpl()
    rows = [r for r in rows if r.get('valid')]
    if not rows:
        print(f'[yieldplot] {label}: 无有效数据')
        return
    x, t, y1 = _xy(rows, 'y1')
    _, _, y2 = _xy(rows, 'y2')
    _, _, e1 = _xy(rows, 'e1')
    _, _, e2 = _xy(rows, 'e2')
    _, _, ratio = _xy(rows, 'ratio')
    _, _, rerr = _xy(rows, 'ratio_err')

    # ---- 产率：位置 与 时间 分成两列 ----
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5),
                             gridspec_kw={'hspace': 0.32, 'wspace': 0.22})
    (ax1, ax2), (ax3, ax4) = axes
    for ax, yv, ev, lab, col, mk in (
            (ax1, y1, e1, 'Xe$^{1+}$', C_RED, 'o'),
            (ax3, y2, e2, 'Xe$^{2+}$', C_BLUE, 's')):
        ax.errorbar(x, yv, yerr=ev, fmt=mk + '-', ms=5, lw=1.3, capsize=2, color=col)
        ax.set_yscale('log'); ax.set_xlabel('位移台位置  [mm]')
        ax.set_ylabel(f'{lab} 产率 / 帧 (log)'); ax.grid(alpha=0.25, which='both')
        ax.set_xlim(x.min() - 2.5, x.max() + 2.5)
    for ax, yv, ev, lab, col, mk in (
            (ax2, y1, e1, 'Xe$^{1+}$', C_RED, 'o'),
            (ax4, y2, e2, 'Xe$^{2+}$', C_BLUE, 's')):
        ax.errorbar(t, yv, yerr=ev, fmt=mk + '-', ms=5, lw=1.3, capsize=2, color=col)
        ax.set_yscale('log'); ax.set_xlabel('延迟时间  [ps]')
        ax.set_ylabel(f'{lab} 产率 / 帧 (log)'); ax.grid(alpha=0.25, which='both')
        ax.set_xlim(t.min() - 17, t.max() + 17)
    ax1.set_title(f'{label}：Xe$^{{1+}}$ 产率 vs 位置', fontsize=10)
    ax2.set_title(f'{label}：Xe$^{{1+}}$ 产率 vs 时间', fontsize=10)
    ax3.set_title(f'{label}：Xe$^{{2+}}$ 产率 vs 位置', fontsize=10)
    ax4.set_title(f'{label}：Xe$^{{2+}}$ 产率 vs 时间', fontsize=10)
    add_time_axis(ax2, 'top'); add_time_axis(ax4, 'top')
    save_fig(fig, out / f'{label}_yield.png')

    # ---- 比值 ----
    fig2, (ar, at) = plt.subplots(1, 2, figsize=(13, 5))
    ar.errorbar(x, ratio, yerr=rerr, fmt='^-', ms=5, lw=1.4, capsize=2, color=C_GREEN)
    ar.set_xlabel('位移台位置  [mm]'); ar.set_ylabel('Xe$^{1+}$/Xe$^{2+}$ (log)')
    ar.set_yscale('log'); ar.grid(alpha=0.25, which='both')
    ar.set_xlim(x.min() - 2.5, x.max() + 2.5)
    ar.set_title('比值 vs 位置', fontsize=10)
    at.errorbar(t, ratio, yerr=rerr, fmt='^-', ms=5, lw=1.4, capsize=2, color=C_GREEN)
    at.set_xlabel('延迟时间  [ps]'); at.set_ylabel('Xe$^{1+}$/Xe$^{2+}$ (log)')
    at.set_yscale('log'); at.grid(alpha=0.25, which='both')
    at.set_xlim(t.min() - 17, t.max() + 17)
    at.set_title('比值 vs 时间', fontsize=10)
    add_time_axis(at, 'top')
    fig2.suptitle(f'{label}：Xe$^{{1+}}$/Xe$^{{2+}}$ 比例', fontsize=12)
    save_fig(fig2, out / f'{label}_ratio.png')


# =============================================================================
# BLOCK 10 ── 任务：上升沿拟合
# =============================================================================

def _derive_ratio(d: dict) -> dict:
    """BLOCK 10.0 从 CSV 字典补全 t_ps / ratio / ratio_err（兼容早期 CSV）。

    早期脚本输出的 CSV 只有 delay、n_window、xe1、xe2，没有 t_ps 与 ratio，
    这里统一按公式补出来，保证新旧 CSV 都能直接做拟合。
    """
    if 't_ps' not in d and 'delay' in d:
        d['t_ps'] = d['delay'] * PS_PER_MM
    if 'ratio' not in d:
        xe1 = d.get('xe1', np.zeros(1))
        xe2 = d.get('xe2', np.zeros(1))
        d['ratio'] = xe1 / np.maximum(xe2, 1e-9)
    if 'ratio_err' not in d:
        xe1 = d.get('xe1', np.zeros(1))
        xe2 = d.get('xe2', np.zeros(1))
        d['ratio_err'] = d['ratio'] * np.sqrt(1.0 / np.maximum(xe1, 1)
                                              + 1.0 / np.maximum(xe2, 1))
    return d


def task_erf_fit(csv_path: Path, out: Path, fraction: str = '1/3',
                 nboot: int = 4000, label: str = '') -> dict | None:
    """BLOCK 10.1 对 Xe1+/Xe2+ 比值曲线的左侧一段做误差函数拟合。"""
    plt = _mpl()
    d = _derive_ratio(load_yield_csv(csv_path))
    t = d['t_ps']; R = d['ratio']; err = d['ratio_err']
    num, den = fraction.split('/')
    m = int(np.ceil(t.size * float(num) / float(den)))
    tf, Rf, ef = t[:m], R[:m], err[:m]
    if tf.size < 6:
        print('[erf] 拟合点太少')
        return None

    res = fit_erf_edge(tf, Rf, ef)
    label = label or csv_path.stem
    print(f'\n[erf] {label}  左侧 {fraction} ({m}/{t.size} 点, '
          f'{tf.min():.1f}-{tf.max():.1f} ps)')

    def _fmt(v, e, unit=''):
        if not np.isfinite(e) or e > 2 * max(abs(v), 1e-9):
            return f'{v:>10.3f}  (误差不可确定)'
        return f'{v:>10.3f} +- {e:.3f}{unit}'

    print(f'[erf]   b   (基线)    = {_fmt(res["b"], res["b_err"])}')
    print(f'[erf]   A   (阶跃)    = {_fmt(res["a"], res["a_err"])}')
    print(f'[erf]   t0  (50%点)   = {_fmt(res["t0"], res["t0_err"], " ps")}')
    print(f'[erf]   tau (时间常数) = {_fmt(res["tau"], res["tau_err"], " ps")}')
    print(f'[erf]   10-90% 上升   = {res["rise_10_90"]:.1f} ps')
    print(f'[erf]   chi2/ndof     = {res["chi2red"]:.1f}')

    if not res['identifiable']:
        print('[erf]   ! 警告：该数据无法约束 tau。上升沿比采样步长还窄，'
              '只有 1-2 个点跨越台阶，\n'
              '[erf]           tau 的数值只反映采样间距，不是实测的物理上升时间。')
    if np.isfinite(res['chi2red']) and res['chi2red'] > 2:
        print(f'[erf]   注意：chi2/ndof 远大于 1，泊松误差低估散布，'
              f'名义误差需膨胀约 {np.sqrt(res["chi2red"]):.1f} 倍')

    # 采样步长：应取"上升沿附近"的间距，而不是全局中位数
    # （全局中位数会被重复测量点和大跨度间隔带偏）
    near = np.abs(t - res['t0']) < max(5 * res['tau'], 50.0)
    if near.sum() >= 2:
        dt = float(np.median(np.diff(t[near])))
        dt_src = '上升沿附近'
    else:
        dt = float(np.median(np.diff(t)))
        dt_src = '全局中位'
    print(f'[erf]   采样步长（{dt_src}）= {dt:.1f} ps；'
          f'10-90% 上升时间 = {res["rise_10_90"]:.1f} ps'
          f'（{res["rise_10_90"]/dt:.2f} 个采样间隔）')
    if res['rise_10_90'] < 2 * dt:
        print('[erf]   警告：上升时间不足两个采样间隔，时间常数不可靠测定')

    taus = bootstrap_erf(tf, Rf, ef, res['params'], nboot=nboot)
    if taus.size > 1:
        print(f'[erf]   bootstrap({taus.size}): tau 中位 {np.median(taus):.2f} ps, '
              f'16-84 [{np.percentile(taus, 16):.2f}, '
              f'{np.percentile(taus, 84):.2f}] ps')

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.12})
    ax.errorbar(t, R, yerr=err, fmt='o', ms=4, capsize=2, color=C_GREY,
                label=f'全部 {t.size} 点', zorder=1)
    ax.errorbar(tf, Rf, yerr=ef, fmt='o', ms=6, capsize=2, color=C_RED,
                label=f'拟合区间 {m} 点', zorder=2)
    fine = np.linspace(tf.min() - 20, tf.max() + 20, 500)
    ax.plot(fine, _erf_step(fine, *res['params']), '-', lw=2, color=C_BLUE,
            label='erf 拟合')
    ax.axvline(res['t0'], color='k', ls='--', lw=1.1, alpha=0.7)
    ax.axhline(res['b'], color='k', ls=':', lw=0.9, alpha=0.6)
    ax.axhline(res['b'] + res['a'], color='k', ls=':', lw=0.9, alpha=0.6)
    ax.fill_betweenx([res['b'], res['b'] + res['a']],
                     res['t0'] - res['tau'], res['t0'] + res['tau'],
                     color=C_BLUE, alpha=0.15)
    ax.set_ylabel('Xe$^{1+}$/Xe$^{2+}$'); ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f'{label} — 上升沿 erf 拟合（左侧 {fraction}）', fontsize=12)
    ax.text(0.985, 0.06,
            f'τ = {res["tau"]:.1f} ± {res["tau_err"]:.1f} ps\n'
            f't₀ = {res["t0"]:.0f} ps\n'
            f'10–90% = {res["rise_10_90"]:.0f} ps',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(fc='white', ec='0.6', alpha=0.95))

    ax2.axhline(0, color='k', lw=0.8)
    ax2.errorbar(tf, res['resid'], yerr=ef, fmt='o', ms=5, capsize=2, color=C_RED)
    ax2.set_xlabel('延迟时间  [ps]'); ax2.set_ylabel('残差')
    ax2.set_title(f'残差 (χ²/ndof = {res["chi2red"]:.2f})', fontsize=10)
    ax2.grid(alpha=0.25)
    add_time_axis(ax2, 'top')
    fig.tight_layout()
    save_fig(fig, out / f'{label}_erf_fit.png')
    return res


def erf_range_sensitivity(csv_path: Path, fractions=(0.25, 1/3, 0.5, 2/3, 1.0)) -> None:
    """BLOCK 10.2 拟合范围敏感性：tau 随所取区间变化多少（重要诊断）。"""
    d = _derive_ratio(load_yield_csv(csv_path))
    t = d['t_ps']
    R = d['ratio']
    err = d['ratio_err']
    print(f'\n[erf] 范围敏感性 — {csv_path.stem}')
    print(f'{"fraction":>10}{"points":>8}{"t0[ps]":>10}{"tau[ps]":>10}'
          f'{"10-90%[ps]":>12}{"chi2/n":>9}')
    for fr in fractions:
        k = int(np.ceil(t.size * fr))
        if k < 6:
            continue
        try:
            r = fit_erf_edge(t[:k], R[:k], err[:k])
        except Exception as exc:
            print(f'{fr:>10.2f}{k:>8}   拟合失败: {exc}')
            continue
        # 退化判定：t0 落在数据范围外太远，或 tau 大于整个拟合区间，
        # 说明该区间内已经没有单个干净的上升沿，erf 模型不再适用。
        span = t[:k].max() - t[:k].min()
        degenerate = (r['t0'] < t[:k].min() - span or r['t0'] > t[:k].max() + span
                      or r['tau'] > span / 3)
        if degenerate:
            print(f'{fr:>10.2f}{k:>8}{"     —":>10}{"     —":>10}{"     —":>12}'
                  f'{r["chi2red"]:>9.1f}  <- 退化（该区间无单一上升沿）')
            continue
        print(f'{fr:>10.2f}{k:>8}{r["t0"]:>10.1f}{r["tau"]:>10.2f}'
              f'{r["rise_10_90"]:>12.1f}{r["chi2red"]:>9.1f}')


# =============================================================================
# 流水线入口
# =============================================================================

def run_gmd(root: Path, out: Path, sample: str | None = None) -> None:
    """任务组 A：GMD 诊断 + 帧偏移扫描。"""
    plt = _mpl()
    task_gmd_diagnostics(root, out, sample)

    positions = scan_dataset(root)
    if not positions:
        return
    pos = (next((p for p in positions if p.name == sample), None)
           if sample else max(positions, key=lambda p: p.n_frames))
    g1, g2, npix = pos.arrays()
    rows = frame_offset_scan(g2, npix, MAX_SHIFT_SCAN)
    print(f'\n[align] {pos.name} 帧偏移扫描 (GMD2 vs 像素数)')
    print(f'{"d":>4}{"Pearson r":>12}{"Spearman":>11}')
    for r in rows:
        mark = '  <== d=0' if r['d'] == 0 else ''
        print(f'{r["d"]:>4d}{r["r"]:>12.4f}{r["rho"]:>11.4f}{mark}')
    best = max(rows, key=lambda r: (r['r'] if np.isfinite(r['r']) else -9))
    print(f'[align] 最佳偏移 d = {best["d"]:+d} '
          f'(r={best["r"]:.4f}，d=0 为 '
          f'{[r["r"] for r in rows if r["d"] == 0][0]:.4f})')

    ds = np.array([r['d'] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ds, [r['r'] for r in rows], 'o-', color=C_BLUE, ms=6, lw=1.5,
            label='Pearson r')
    ax.plot(ds, [r['rho'] for r in rows], 's--', color=C_GREEN, ms=5, lw=1.2,
            label='Spearman ρ')
    ax.plot(best['d'], best['r'], 'r*', ms=20, label=f'最佳 d = {best["d"]:+d}')
    ax.axvline(0, color='k', ls=':', lw=1.2, alpha=0.7, label='d = 0 (标称)')
    ax.set_xlabel('帧偏移  d\n(GMD2[i] vs 像素数[i+d])')
    ax.set_ylabel('相关系数')
    ax.set_title(f'GMD 与相机帧错位扫描 — {pos.name}')
    ax.grid(alpha=0.25); ax.legend(frameon=False); ax.set_xticks(ds)
    save_fig(fig, out / f'{pos.name}_frame_alignment.png')


def run_tof(root: Path, out: Path, label: str,
            gmd_window: tuple[float, float] | None = GMD2_DEFAULT,
            shift: int = FRAME_SHIFT, sample: str | None = None) -> None:
    """任务组 B：TOF 谱 + TOF 随位置变化。"""
    positions = scan_dataset(root)
    if not positions:
        return
    if sample:
        pos = next((p for p in positions if p.name == sample), None)
        if pos is None:
            print(f'[tof] 未找到文件夹 {sample}')
            return
    else:
        # 选像素最多的位置（而不是帧数最多），保证峰拟合可靠
        pos = max(positions, key=lambda p: sum(r.npix for r in p.records))
    task_tof_spectrum(pos, out, gmd_window, shift)
    task_tof_vs_position(root, out, gmd_window)


def run_yield(root: Path, out: Path, label: str,
              gmd_window: tuple[float, float] | None = GMD2_DEFAULT,
              shift: int = FRAME_SHIFT,
              use_tracked: bool = True) -> Path | None:
    """任务组 C：产率曲线（含 CSV）。返回 CSV 路径供 erf 任务使用。"""
    rows = compute_yield_curve(root, gmd_window, shift=shift,
                               use_tracked=use_tracked)
    if not rows:
        print('[yield] 无结果')
        return None
    if gmd_window is None:
        tag = 'noGMD'
    else:
        tag = f'{gmd_window[0]:g}-{gmd_window[1]:g}nA'
    csv_p = out / f'{label}_yield_{tag}.csv'
    save_csv(rows, csv_p)
    task_yield_curve(rows, out, f'{label} ({tag})')
    return csv_p


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='TPX3 离子成像数据离线分析流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='任务: verify | gmd | tof | yield | erf | all')

    ap.add_argument('task', choices=['verify', 'gmd', 'tof', 'yield',
                                     'erf', 'all'])
    ap.add_argument('--root', help='数据集目录（内含延迟位置文件夹）')
    ap.add_argument('--out', help='输出目录（默认 <root>/analysis_out）')
    ap.add_argument('--label', default='DATA', help='输出文件名前缀')
    ap.add_argument('--sample', help='指定用于诊断的文件夹名')
    ap.add_argument('--gmd-min', type=float, default=GMD2_DEFAULT[0])
    ap.add_argument('--gmd-max', type=float, default=GMD2_DEFAULT[1])
    ap.add_argument('--no-gmd', action='store_true', help='不筛选 GMD（全部帧）')
    ap.add_argument('--shift', type=int, default=FRAME_SHIFT,
                    help='GMD 帧偏移（默认 -1）')
    ap.add_argument('--fixed-window', action='store_true',
                    help='使用固定 TOF 窗口而非峰跟随窗口')
    ap.add_argument('--unwrap', choices=['signed', 'go4'], default=UNWRAP_MODE)
    ap.add_argument('--csv', help='erf 任务使用的产率 CSV')
    ap.add_argument('--fraction', default='1/3', help='erf 拟合区间比例')
    ap.add_argument('--nboot', type=int, default=4000)
    ap.add_argument('--odl1', help='(all) ODL1 数据集目录')
    ap.add_argument('--odl2', help='(all) ODL2 数据集目录')

    a = ap.parse_args(argv)

    if a.task == 'verify':
        return 0 if verify_reader(a.sample) else 1

    if a.task == 'erf':
        if not a.csv:
            ap.error('erf 任务需要 --csv')
        p = Path(a.csv)
        out = Path(a.out) if a.out else p.parent
        task_erf_fit(p, out, a.fraction, a.nboot, a.label if a.label != 'DATA' else p.stem)
        erf_range_sensitivity(p)
        return 0

    if a.task == 'all':
        ok = True
        for lbl, r in (('ODL1', a.odl1), ('ODL2', a.odl2)):
            if not r:
                continue
            root = Path(r)
            out = Path(a.out) / lbl if a.out else root / 'analysis_out'
            print(f'\n{"="*70}\n== {lbl}: {root}\n{"="*70}')
            run_gmd(root, out, a.sample)
            run_tof(root, out, lbl)
            csv_p = run_yield(root, out, lbl)
            if csv_p:
                task_erf_fit(csv_p, out, a.fraction, a.nboot, f'{lbl}_1-3')
                erf_range_sensitivity(csv_p)
            # 不筛选版本
            run_yield(root, out, lbl, gmd_window=None)
        return 0 if ok else 1

    if not a.root:
        ap.error(f'{a.task} 任务需要 --root')
    root = Path(a.root)
    out = Path(a.out) if a.out else root / 'analysis_out'
    gmd_win = None if a.no_gmd else (a.gmd_min, a.gmd_max)

    if a.task == 'gmd':
        run_gmd(root, out, a.sample)
    elif a.task == 'tof':
        run_tof(root, out, a.label, gmd_win, a.shift, a.sample)
    elif a.task == 'yield':
        run_yield(root, out, a.label, gmd_win, a.shift,
                  use_tracked=not a.fixed_window)
    return 0


if __name__ == '__main__':
    sys.exit(main())
