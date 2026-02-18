# sati: Sati is an Analysis Toolset for Impact

衛星軌道の解析のために作ったプログラムのうち，中核部分の関数をまとめた．

[TOC]

## 基本の関数
基本の関数はrecordとtimelineの２つだけで，残りは派生形であったり，利便性を高めただけのものである．

## record
recordは，衛星軌道の各時刻での情報を格納したデータを作成する関数である．
以降，関数recordで作成されたデータ自体もrecordと呼ぶことがある．
データはpd.DataFrameやCSVに格納される．
作成されるrecordの例は，record_sample.csvを参照．

関数recordの引数を説明する:
```Python
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
    bits (int): 出力項目のビット指定。0-255の値。DATETIME, GLAT, GLON, ALT_KM, SUNLIT, MLAT, MLT, Lの順番
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
```

使用例を以下に示す:
```Python
from datetime import datetime, timedelta
from skyfield.api import load, utc, EarthSatellite

sat_list = load.tle('https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle')
sat = sat_list[name]
bits = 0b11110111
ts = load.timescale()

t0 = datetime(2025, 8, 1, tzinfo=utc)
t1 = t0 + timedelta(days=1)
dt = timedelta(seconds=10)

rec = sati.record(sat, bits, ts, t0, t1, dt, method="AACGM")
```

注意点を述べる:
- skyfieldライブラリのEarthSatelliteオブジェクトを作成する方法は複数ある．sati.curr_sat()でも作れるようにしてある
- 引数bitsはいわゆるビットフラグで，1になっている部分だけ計算されて出力される．この関数でのビットフラグは左からDATETIME(日付), GLAT(地理緯度),GLON(地理経度), ALT_KM(km単位の高度), SUNLIT(太陽が当たっているかどうかのbool値), MLAT(磁気緯度), MLT(磁気地方時), L(L値)の計算をするかどうかを指定するフラグである．
- t0, t1は必ずタイムゾーン情報としてUTCを指定する．

### recordの派生関数
recordから派生した関数を示す．
- `csv_from_record`
- `record_from_csv`

---

`csv_from_record`はrecordの出力をcsvに保存することに特化した関数である．
長期間にわたる衛星軌道を計算したかったり，時間分解能dtを短くしすぎたりした時に，recordのデータ量やメモリ使用量は莫大なものになる場合がある．
また，計算時間も長くなるため非常に面倒．
そんな時にこの関数を使う．
巨大になりうるrecordをcsvに保存しておくことで，巨大なデータを再度計算する必要がないようにすることができる．

`csv_from_record`の引数を示す:
```Python
def csv_from_record(
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
```

---

`record_from_csv`は，csvに保存したrecordを読み込むための関数である．
巨大なrecordを読み込むことを前提としているため，csvはチャンクごとに読み込まれ，関数の返り値はイテレータである．(pd.read_csv()を参照)
`record_from_csv`の引数を以下に示す:
```Python
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
```

## timeline
timelineは，recordの中から条件を満たす時間帯を抜き出す関数である．
以降，関数timelineで作成されたデータ自体もtimelineと呼ぶことがある．
データはpd.DataFrameやCSVに格納される．
作成されるtimelineの例は，timeline_sample.csvを参照．

関数timelineの引数を以下に示す:
```Python
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
```



