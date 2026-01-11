import os
import logging
import re
from datetime import datetime, timedelta
from enum import IntFlag
from typing import Iterable
import pandas as pd
import numpy as np
import skyfield
import skyfield.sgp4lib
import skyfield.timelib
import skyfield.toposlib
from skyfield.api import wgs84, load, EarthSatellite
from apexpy import Apex
import aacgmv2


# ======== ロギング設定 ========
logging.getLogger("aacgmv2_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# ===== 定数 =====
COMMENT_DELIM_CSV = "#"
TIME_FORMAT_CSV = "%Y-%m-%dT%H:%M:%SZ"
TIME_FORMAT_FILENAME = "%Y%m%d%H%M%SZ"


# ======== DataField / Fields ========
class DataField(str):
    """
    pd.DataFrame のカラム名や属性名を表す。
    文字列として使えるが、bit, precision, value などの情報を保持する。
    """
    def __new__(cls, name, bit=None, precision=None, value=None):
        obj = str.__new__(cls, name)
        obj.bit = bit
        obj.precision = precision
        obj.value = value
        return obj

    def __repr__(self):
        return f"DataField({str(self)!r})"


class RecordBit(IntFlag):
    DATETIME = 1 << 7
    GLAT = 1 << 6
    GLON = 1 << 5
    ALT_KM = 1 << 4
    SUNLIT = 1 << 3
    MLAT = 1 << 2
    MLT = 1 << 1
    L = 1 << 0


class Fields:
    """すべての DataField を定義するクラス"""

    class Record:
        DATETIME = DataField("DATETIME", bit=RecordBit.DATETIME)
        GLAT = DataField("GLAT", bit=RecordBit.GLAT, precision=3)
        GLON = DataField("GLON", bit=RecordBit.GLON, precision=3)
        ALT_KM = DataField("ALT_KM", bit=RecordBit.ALT_KM, precision=3)
        SUNLIT = DataField("SUNLIT", bit=RecordBit.SUNLIT, precision=1)
        MLAT = DataField("MLAT", bit=RecordBit.MLAT, precision=3)
        MLT = DataField("MLT", bit=RecordBit.MLT, precision=3)
        L = DataField("L", bit=RecordBit.L, precision=3)

    class Method:
        AACGM = DataField("AACGM")
        APEX = DataField("APEX")

    class TimeLine:
        START = DataField("START")
        END = DataField("END")
        DURATION_SEC = DataField("DURATION_SEC")
        AOS = DataField("AOS")
        LOS = DataField("LOS")
        MAX_EL = DataField("MAX_EL")
        MAX_EL_DEG = DataField("MAX_EL_DEG", precision=3)
        EVENT = DataField("EVENT")

    class Attrs:
        SAT_NAME = DataField("SAT_NAME")
        BITS = DataField("BITS")
        T0 = DataField("START_TIME")
        T1 = DataField("END_TIME")
        DT = DataField("TIME_RESOLUTION")
        METHOD = DataField("METHOD")
        OBS = DataField("OBSERVATION_POINT")
        MINEL = DataField("MINIMUM_ELEVATION", value=5)

    class Event:
        PASSTIME = DataField("PASSTIME")
        SUNLIT = DataField("SUNLIT")
        EMIC = DataField("EMIC")
        LIGHTNING = DataField("LIGHTNING_WHISTLER")
        CHORUS = DataField("CHORUS")


RECORD_COLUMN_ORDER = [
    Fields.Record.DATETIME,
    Fields.Record.GLAT,
    Fields.Record.GLON,
    Fields.Record.ALT_KM,
    Fields.Record.SUNLIT,
    Fields.Record.MLAT,
    Fields.Record.MLT,
    Fields.Record.L,
]

TIMELIME_COLUMN_ORDER = [
    Fields.TimeLine.START,
    Fields.TimeLine.END,
    Fields.TimeLine.DURATION_SEC,
]


# ===== 共通ユーティリティ =====
def name_file(prefix: str, sat_name: str, t0: datetime, t1: datetime, **params) -> str:
    """共通的なファイル名生成"""
    param_str = "_".join(f"{k}={v}" for k, v in params.items())
    return (
        f"{prefix}_{sat_name}_"
        f"{t0.strftime('%Y%m%d%H%M%SZ')}_"
        f"{t1.strftime('%Y%m%d%H%M%SZ')}_"
        f"{param_str}.csv"
    )


def name_record(bits, sat_name, t0, t1, dt, method):
    return name_file(
        "record", sat_name, t0, t1,
        bits=f"{bits:03d}", dt=dt.total_seconds(), method=method
    )


def _parse_record_name(filename: str) -> dict:
    """recordファイル名を分解してメタ情報を抽出"""
    pattern = (
        r"record_(\d{3})_(.*?)_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_dt([0-9.]+)_method(.*?)\.csv"
    )
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid record filename: {filename}")
    bits, sat, t0, t1, dt, method = match.groups()
    return dict(
        bits=int(bits),
        sat_name=sat,
        t0=datetime.strptime(t0, "%Y%m%dT%H%M%SZ"),
        t1=datetime.strptime(t1, "%Y%m%dT%H%M%SZ"),
        dt=timedelta(seconds=float(dt)),
        method=method,
    )


def _name_passtime(sat_name, t0, t1, obs_point: str, minel, format="csv"):
    filename = (
        f"passtime_{sat_name}_{obs_point}_"
        f"{t0.strftime(TIME_FORMAT_FILENAME)}_"
        f"{t1.strftime(TIME_FORMAT_FILENAME)}_"
        f"minel={minel}." + format
    )
    return filename


def _calc_sunlit(geocentric):
    """衛星位置が日照下にあるかを判定"""
    eph = load("de421.bsp")
    sunlit_array = geocentric.is_sunlit(eph)
    return sunlit_array


def _calc_mag(t_array, glat_array, glon_array, alt_km_array, apex=None, method=Fields.Method.AACGM):
    """
    MLAT, MLTを計算
    t_array = (ts.utc(...)).utc_datetime()
    """
    mlat_array = np.empty_like(glat_array)
    #mlon_array = np.empty_like(glat)
    mlt_array  = np.empty_like(glat_array)

    for i, (la, lo, al, ti) in enumerate(zip(glat_array, glon_array, alt_km_array, t_array)):
        if method == Fields.Method.AACGM:
            mlat_array[i], _, mlt_array[i] = aacgmv2.get_aacgm_coord(la, lo, al, ti)
        elif method == Fields.Method.APEX and apex is not None:
            mlat_array[i], mlt_array[i] = apex.convert(la, lo, "geo", "mlt", height=al, datetime=ti)
        else:
            raise ValueError("can't calculate magnetic coordinate")

    return mlat_array, mlt_array


def _calc_L(t_array, glat_array, glon_array, alt_km_array, mlat_array, method=Fields.Method.AACGM):
    """
    L-valueの計算
    t_array = (ts.utc(...)).utc_datetime()
    """
    l_array = np.empty_like(glat_array)

    if method == Fields.Method.AACGM:
        Re = 6378  # km
        l_array = (Re + alt_km_array) / (Re * (np.cos(np.radians(mlat_array)) ** 2))
    elif method == Fields.Method.APEX:
        for i, (la, lo, al, ti) in enumerate(zip(glat_array, glon_array, alt_km_array, t_array)):
            apex = Apex(ti, al)
            alat, _ = apex.geo2apex(la, lo, al)
            aalt = apex.get_apex(alat, al)
            l_array[i] = 1.0 + aalt / apex.RE
    else:
        raise ValueError("can't calculate L value")

    return l_array


# ===== record計算 =====
def _calc_orbit_arrays(sat, bits, ts, t0, t1, dt, method) -> dict[str, np.ndarray]:
    time_points = ts.utc(
        t0.year, t0.month, t0.day, t0.hour, t0.minute,
        np.arange(t0.second, (t1 - t0).total_seconds() + t0.second, dt.total_seconds())
    )

    geocentric = sat.at(time_points)
    subpoint = wgs84.subpoint(geocentric)

    arrays = {
        Fields.Record.DATETIME: pd.to_datetime(time_points.utc_datetime(), utc=True),
        Fields.Record.GLAT: subpoint.latitude.degrees,
        Fields.Record.GLON: subpoint.longitude.degrees,
        Fields.Record.ALT_KM: subpoint.elevation.km,
    }

    if bits & RecordBit.SUNLIT:
        arrays[Fields.Record.SUNLIT] = _calc_sunlit(geocentric)

    if bits & (RecordBit.MLAT | RecordBit.MLT | RecordBit.L):
        apex = Apex(date=t0)
        mlat_array, mlt_array = _calc_mag(
            time_points.utc_datetime(),
            arrays[Fields.Record.GLAT],
            arrays[Fields.Record.GLON],
            arrays[Fields.Record.ALT_KM],
            apex,
            method=method
        )
        arrays[Fields.Record.MLAT] = mlat_array
        arrays[Fields.Record.MLT] = mlt_array

    if bits & RecordBit.L:
        arrays[Fields.Record.L] = _calc_L(
            time_points.utc_datetime(),
            arrays[Fields.Record.GLAT],
            arrays[Fields.Record.GLON],
            arrays[Fields.Record.ALT_KM],
            arrays[Fields.Record.MLAT],
            method=method
        )

    return arrays


def _build_record_df(arrays: dict, bits: int) -> pd.DataFrame:
    """DataFrame化と丸め処理"""
    df = pd.DataFrame(arrays)
    round_map = {
        field: field.precision
        for field in Fields.Record.__dict__.values()
        if isinstance(field, DataField)
        and field.precision is not None
        and field in df.columns
    }
    if round_map:
        df = df.round(round_map)
    return df


def record(
        sat: skyfield.sgp4lib.EarthSatellite,
        bits: int,
        ts: skyfield.timelib.Timescale,
        t0: datetime,
        t1: datetime,
        dt: timedelta,
        method=Fields.Method.AACGM.name,
        save_csv: bool = False
) -> pd.DataFrame:
    """
    衛星が通過する軌道の各時刻において, 地理座標, 高度, 磁気座標をpandas.DataFrameに格納して返す。
    衛星軌道上のあらゆるデータを格納するので，データ量が大きい。注意
    2.12 GB if 1 year and bits=11111111

    Args:
    sat (skyfield.sgp4lib.EarthSatellite): 衛星の変数
    bits (int): 出力項目のビット指定。0-255の値。
    ts (skyfield.timelib.Timescale): タイムスケール。skyfield.api.load.timescale()を入れる。初期値はNone
    t0: (datetime): 計算開始する日付と時刻 (UTC)。
    t1: (datetime): 計算終了する日付と時刻 (UTC)。
    注: t0とt1は, from skyfield.api import utc を使い, tzinfo=utcとする
    dt (datetime.timedelta): 時間分解能。timedelta(seconds=1)などとする
    method (str): どのライブラリで磁気座標を計算するか。"AACGM"ならaacgmv2, "APEX"ならapexpyで計算する
    save_csv (bool): data frameをcsvに保存するか

    Returns:
    pd.DataFrame: 衛星の情報が格納されたdata frame.

    """
    if t0 >= t1:
        raise ValueError("t0 must be earlier than t1")

    arrays = _calc_orbit_arrays(
        sat=sat,
        bits=bits,
        ts=ts,
        t0=t0,
        t1=t1,
        dt=dt,
        method=method,
    )

    df = _build_record_df(arrays, bits)

    cols = [c for c in RECORD_COLUMN_ORDER if c in df.columns]
    df = df[cols]

    # メタデータ
    df.attrs = {
        Fields.Attrs.SAT_NAME: sat.name,
        Fields.Attrs.BITS: bits,
        Fields.Attrs.T0: t0,
        Fields.Attrs.T1: t1,
        Fields.Attrs.DT: dt,
        Fields.Attrs.METHOD: method,
    }

    if save_csv:
        fname = name_record(bits, sat.name, t0, t1, dt, method)
        with open(fname, "w", encoding="utf-8") as f:
            f.write("# " + fname + "\n")
            df.to_csv(f, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")

    return df


def csv_of_record(
        sat: EarthSatellite,
        bits: int,
        ts: skyfield.timelib.Timescale,
        t0: datetime,
        t1: datetime,
        dt: timedelta,
        method: str=Fields.Method.AACGM,
        output_path: str | None=None,
        chunk_timedelta: timedelta=timedelta(hours=24),
) -> None:
    """
    recordを記載したcsv作成特化の関数．
    大規模なrecordに使う．
    例えば，t0からt1までの期間が長い場合や，dtが小さすぎるときに使う．

    Args:
    output_path (str): 保存先のパス．デフォルト値 None
    chunk_timedelta (timedelta): どのくらいの期間を一塊としてベクトル化された計算をするか。デフォルト値timedelta(hours=24)

    Returns:
    None: pd.Dataframeは返さない．csv保存専用関数である．
    """
    if output_path is None:
        output_path = name_record(bits, sat.name, t0, t1, dt, method, format="csv")

    # コメント行を書き込み
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{COMMENT_DELIM_CSV} {output_path}\n")

    first_chunk = True
    curr_t = t0

    while curr_t < t1:
        chunk_end = min(curr_t + chunk_timedelta, t1)

        # 境界重複防止（2チャンク目以降）
        effective_t0 = curr_t if first_chunk else curr_t + dt

        if effective_t0 >= chunk_end:
            curr_t = chunk_end
            continue

        df_chunk = record(
            sat=sat,
            bits=bits,
            ts=ts,
            t0=effective_t0,
            t1=chunk_end,
            dt=dt,
            method=method,
        )

        df_chunk.to_csv(
            output_path,
            mode="a",
            index=False,
            header=first_chunk,
            date_format=TIME_FORMAT_CSV,
        )

        first_chunk = False
        curr_t = chunk_end


def record_from_csv(
    filepath: str,
    usecols: list[str] | None = None,
    comment: str = COMMENT_DELIM_CSV,
    chunksize: int | None = None,
) -> tuple[dict, Iterable[pd.DataFrame]]:
    """
    record CSV を読み込み、attrs と DataFrame iterator を返す。
    """

    attrs = _parse_record_name(os.path.basename(filepath))

    parse_dates = None
    if usecols is None or Fields.Record.DATETIME in usecols:
        parse_dates = [Fields.Record.DATETIME]

    reader = pd.read_csv(
        filepath,
        usecols=usecols,
        parse_dates=parse_dates,
        comment=comment,
        chunksize=chunksize,
    )

    if chunksize is None:
        def _iter():
            yield reader
        df_iter = _iter()
    else:
        df_iter = reader

    return attrs, df_iter


def timeline(
    rec: pd.DataFrame,
    cond: pd.Series,
    required_cols: list[str] | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    condの条件を満たす時間帯を取り出す。

    Args:
        rec (pd.DataFrame): 入力されるrecord
        cond (pd.Series): rec["foo"]>barなどとして抽出したもの
        required_cols (List[str]): 抽出に必要とするカラム名。特に何も指定しなくていい
        save_path (str): 保存先のパス。デフォルト値None。

    Returns:
    pd.DataFrame
    """

    if required_cols is None:
        required_cols = []

    missing = [c for c in required_cols if c not in rec.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not isinstance(cond, pd.Series):
        cond = pd.Series(cond, index=rec.index)
    else:
        if not cond.index.equals(rec.index):
            raise ValueError("cond index must match rec.index")

    cond = cond.astype(bool)

    prev = cond.shift(1, fill_value=False)
    starts = cond & ~prev
    ends = ~cond & prev

    start_times = rec.loc[starts, Fields.Record.DATETIME].reset_index(drop=True)
    end_times = rec.loc[ends, Fields.Record.DATETIME].reset_index(drop=True)

    if len(end_times) < len(start_times):
        end_times = pd.concat(
            [end_times, rec[Fields.Record.DATETIME].iloc[[-1]]],
            ignore_index=True,
        )

    df = pd.DataFrame({
        Fields.TimeLine.START: start_times,
        Fields.TimeLine.END: end_times,
    })

    df[Fields.TimeLine.DURATION_SEC] = (
        df[Fields.TimeLine.END] - df[Fields.TimeLine.START]
    ).dt.total_seconds()

    df.attrs = dict(rec.attrs)
    df.attrs.pop(Fields.Attrs.BITS, None)

    if save_path:
        df.to_csv(save_path, index=False, date_format=TIME_FORMAT_CSV)

    return df


def timeline_stream(
    rec_iter,
    cond_func,
    #save_csv: bool=False
):
    """
    record DataFrame の iterator を受け取り、
    条件を満たす時間帯を逐次抽出する。
    大きなcsvからrecordを読み込んだ時はこれを使ってtimelineを抽出
    出力timelineも大きくなるかも？

    Args:
        rec_iter: Iterable[pd.DataFrame]
        cond_func: Callable[[pd.DataFrame], pd.Series]
            例: lambda df: df["foo"] < 999

    Returns:
        pd.DataFrame (timeline)
    """

    results = []

    prev_cond = False
    open_start = None

    for rec in rec_iter:
        dt_col = Fields.Record.DATETIME.name
        cond = cond_func(rec).astype(bool)

        for t, c in zip(rec[dt_col], cond):
            if c and not prev_cond:
                # False → True（開始）
                open_start = t

            elif not c and prev_cond:
                # True → False（終了）
                results.append({
                    Fields.TimeLine.START.name: open_start,
                    Fields.TimeLine.END.name: t,
                })
                open_start = None

            prev_cond = c

    # ファイル末尾まで True が続いた場合
    if open_start is not None:
        results.append({
            Fields.TimeLine.START.name: open_start,
            Fields.TimeLine.END.name: t,  # 最後に見た時刻
        })

    df = pd.DataFrame(results)

    if not df.empty:
        df[Fields.TimeLine.DURATION_SEC.name] = (
            df[Fields.TimeLine.END.name] - df[Fields.TimeLine.START.name]
        ).dt.total_seconds()

    return df


def passtime(sat: skyfield.sgp4lib.EarthSatellite,
            ts: skyfield.timelib.Timescale,
            t0: datetime, t1: datetime,
            pos: skyfield.toposlib.GeographicPosition,
            min_elevation_deg: float=Fields.Attrs.MINEL.value,
            to_timeline: bool=False,
            save_csv: bool=False) -> pd.DataFrame:
    """
    衛星のAOS, 最大仰角, LOSの時刻のdata frameを返す

    Args:
        sat (skyfield.sgp4lib.EarthSatellite): 衛星の変数
        pos (skyfield.toposlib.GeographicPosition): 観測地点。wgs84.latlon(glat_deg, glon_deg, elev_km)を入れる
        min_elevation_deg (float): 可視とみなす最小の仰角。5 deg or 10 deg ?
        t0 (datetime): どの時刻からパスを計算するか
        t1 (datetime): どの時刻までパスを計算するか
        timescale (skyfield.timelib.Timescale): タイムスケール。skyfield.api.load.timescale()を入れる。
        to_timeline (bool): timeline形式にするか。つまりAOS->START, LOS->ENDにするか
        save_csv (bool): CSVファイルを保存するか。初期値はFalse
    Returns:
        pd.DataFrame: AOS, 最大仰角, LOS時刻を記したdata frame
    """
    t0_sf = ts.utc(t0)
    t1_sf = ts.utc(t1)
    times, events = sat.find_events(pos, t0_sf, t1_sf, altitude_degrees=min_elevation_deg)

    # パスが存在しない場合
    if len(times) == 0:
        empty_df = pd.DataFrame(columns=[
            Fields.TimeLine.AOS,
            Fields.TimeLine.LOS,
            Fields.TimeLine.DURATION_SEC,
            Fields.TimeLine.MAX_EL,
            Fields.TimeLine.MAX_EL_DEG,
        ])
        empty_df.attrs = {
            Fields.Attrs.SAT_NAME: sat.name,
            Fields.Attrs.T0: t0,
            Fields.Attrs.T1: t1,
            Fields.Attrs.OBS: pos,
            Fields.Attrs.MINEL: min_elevation_deg,
        }
        return empty_df

    # 結果格納用
    aos_list = []
    los_list = []
    dur_list = []
    maxel_list = []
    maxel_deg_list = []

    for i in range(0, len(times), 3):
        if i + 3 > len(times):
            break  # 不完全なセットを無視

        t_aos, t_maxel, t_los = times[i:i+3]
        ev_aos, ev_maxel, ev_los = events[i:i+3]

        if (ev_aos, ev_maxel, ev_los) != (0, 1, 2):
            continue  # 不正な組み合わせならスキップ

        # 最大仰角の値を計算
        alt, az, distance = (sat - pos).at(t_maxel).altaz()
        max_el_deg = alt.degrees

        aos_list.append(t_aos.utc_datetime())
        los_list.append(t_los.utc_datetime())
        dur_list.append((t_los.utc_datetime() - t_aos.utc_datetime()).total_seconds())
        maxel_list.append(t_maxel.utc_datetime())
        maxel_deg_list.append(max_el_deg)

    df = pd.DataFrame({
        Fields.TimeLine.AOS: aos_list,
        Fields.TimeLine.LOS: los_list,
        Fields.TimeLine.DURATION_SEC: dur_list,
        Fields.TimeLine.MAX_EL: maxel_list,
        Fields.TimeLine.MAX_EL_DEG: maxel_deg_list,
    })

    df = df.round({
        Fields.TimeLine.MAX_EL_DEG: Fields.TimeLine.MAX_EL_DEG.precision
    })

    df.attrs = {
        Fields.Attrs.SAT_NAME: sat.name,
        Fields.Attrs.T0: t0,
        Fields.Attrs.T1: t1,
        Fields.Attrs.OBS: pos,
        Fields.Attrs.MINEL: min_elevation_deg,
    }

    if to_timeline:
        df = df.rename(columns={
                Fields.TimeLine.AOS: Fields.TimeLine.START,
                Fields.TimeLine.LOS: Fields.TimeLine.END
            })

    if save_csv:
        obs_point = f"({pos.latitude.degrees: .6f},{pos.longitude.degrees: .6f},{pos.elevation.m: 2})"

        filename = _name_passtime(sat.name, t0, t1, obs_point, min_elevation_deg, format="csv")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{COMMENT_DELIM_CSV} {filename}\n")  # コメント行
            df.to_csv(f, index=False, date_format=TIME_FORMAT_CSV)

    return df


