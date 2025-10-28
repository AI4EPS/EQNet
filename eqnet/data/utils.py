# attr_cache.py
# Build three caches:
#   1) event2station.sqlite     : event_id -> JSON(list of station_ids)
#   2) event_attrs.sqlite     : event_id -> JSON(dict of event attrs)
#   3) station_attrs.lmdb/    : (event_id, station_id) -> packed binary attrs blob
#
# Input modes:
#   (A) --h5 PATH [PATH ...]                   : scan HDF5 files
#   (B) --event-csv CSV --station-csv CSV      : load from CSVs (numpy arrays serialized as strings)
# Configure which CSV columns are ndarray via --array-cols "col1,col2,..."
#
# Keys:
#   By default event_id is the HDF5 internal path (e.g., "/grp123").
#   If you have multiple files with overlapping event_ids, pass --prefix=file
#   to namespace keys as "<file_basename>:<event_id>".

import argparse, os, json, sqlite3, lmdb, struct, base64, io, csv, ast
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional
import numpy as np
import h5py
from tqdm import tqdm

# ---------- utils: safe JSON for numpy ----------
def _jsonable(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v

# ---------- pack/unpack attrs dict (supports numpy arrays + scalars) ----------
MAGIC = b"ATRB\x00\x01"

def pack_attrs(attrs: Dict[str, Any]) -> bytes:
    """Pack a dict of {name: np.ndarray or JSON-serializable scalar} into bytes."""
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<I", len(attrs)))
    for k, v in attrs.items():
        kb = k.encode("utf-8")
        buf.write(struct.pack("<I", len(kb))); buf.write(kb)
        if isinstance(v, np.ndarray):
            db = v.dtype.str.encode("ascii")
            buf.write(b"\x01")  # kind=1 ndarray
            buf.write(struct.pack("B", len(db))); buf.write(db)
            buf.write(struct.pack("B", v.ndim)); buf.write(struct.pack("<" + "I"*v.ndim, *v.shape))
            raw = v.tobytes(order="C")
            buf.write(struct.pack("<Q", len(raw))); buf.write(raw)
        else:
            jb = json.dumps(_jsonable(v), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            buf.write(b"\x00")  # kind=0 json scalar
            buf.write(struct.pack("<I", len(jb))); buf.write(jb)
    return buf.getvalue()

def unpack_attrs(b: bytes) -> Dict[str, Any]:
    mv = memoryview(b); off = 0
    assert mv[:len(MAGIC)].tobytes() == MAGIC, "bad magic"
    off += len(MAGIC)
    (n,) = struct.unpack_from("<I", mv, off); off += 4
    out = {}
    for _ in range(n):
        (klen,) = struct.unpack_from("<I", mv, off); off += 4
        k = mv[off:off+klen].tobytes().decode("utf-8"); off += klen
        kind = mv[off]; off += 1
        if kind == 1:
            dlen = mv[off]; off += 1
            dtype = mv[off:off+dlen].tobytes().decode("ascii"); off += dlen
            ndim = mv[off]; off += 1
            dims = struct.unpack_from("<" + "I"*ndim, mv, off); off += 4*ndim
            (nbytes,) = struct.unpack_from("<Q", mv, off); off += 8
            raw = mv[off:off+nbytes]; off += nbytes
            arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(dims)
            out[k] = arr
        else:
            (jlen,) = struct.unpack_from("<I", mv, off); off += 4
            jb = mv[off:off+jlen].tobytes(); off += jlen
            out[k] = json.loads(jb.decode("utf-8"))
            if isinstance(out[k], list) and all(isinstance(x, (int, float)) for x in out[k]):
                # heuristic: convert to np.ndarray
                out[k] = np.array(out[k])
            elif isinstance(out[k], list) and all(isinstance(x, str) for x in out[k]):
                # heuristic: convert to np.ndarray of strings
                out[k] = np.array(out[k], dtype=str)
    return out

# ---------- CSV ndarray decoder (configure columns via --array-cols) ----------
def decode_ndarray_from_str(s: str) -> np.ndarray:
    s = s.strip()
    if not s:
        return np.array([])
    # Try JSON list first
    if s[0] in "[{":
        return np.array(json.loads(s))
    # whitespace-separated
    if " " in s and all(c in "0123456789+-.eE []" for c in s[:200]):
        return np.fromstring(s.replace("[","").replace("]",""), sep=" ", dtype=np.float32)
    # comma-separated
    if "," in s and all(c in "0123456789+-.eE, []" for c in s[:200]):
        return np.fromstring(s.replace("[","").replace("]",""), sep=",", dtype=np.float32)
    # safe literal
    try:
        v = ast.literal_eval(s)
        return np.array(v)
    except Exception:
        raise ValueError(f"unknown ndarray encoding: {s[:64]}...")

# ---------- SQLite helpers ----------
def _sqlite_init(path: str, ddl: str):
    if Path(path).exists():
        os.remove(path)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript(ddl)
    con.commit()
    return con, cur

# ---------- LMDB key ----------
def lmdb_key(trace_id: str) -> bytes:
    return trace_id.encode("utf-8")


class EventToStations:
    """event_id -> list(station_id)"""
    def __init__(self, sqlite_path: str):
        self.con = sqlite3.connect(f"file:{sqlite_path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.cur = self.con.cursor()
    def __getitem__(self, event_id: str) -> List[str]:
        self.cur.execute("SELECT station_ids FROM map WHERE event_id=?", (event_id,))
        row = self.cur.fetchone()
        if not row: raise KeyError(event_id)
        return json.loads(row[0])
    def get(self, event_id: str, default=None):
        try: return self[event_id]
        except KeyError: return default

class EventToAttrs:
    """event_id -> dict(attrs)"""
    def __init__(self, sqlite_path: str):
        self.con = sqlite3.connect(f"file:{sqlite_path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.cur = self.con.cursor()
    def __getitem__(self, event_id: str) -> Dict:
        self.cur.execute("SELECT attrs FROM map WHERE event_id=?", (event_id,))
        row = self.cur.fetchone()
        if not row: raise KeyError(event_id)
        return json.loads(row[0])
    def get(self, event_id: str, default=None):
        try: return self[event_id]
        except KeyError: return default


class StationToAttrs:
    """(event_id/station_id) -> dict(attrs) stored in LMDB"""
    def __init__(self, lmdb_dir: str):
        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=True, max_readers=4096)
        self.txn = self.env.begin(buffers=False)
    def __getitem__(self, key: str) -> Dict:
        v = self.txn.get(lmdb_key(key))
        if v is None: raise KeyError(key)
        return unpack_attrs(v)
    def get(self, key: str, default=None):
        try: return self[key]
        except KeyError: return default

class AttrCache:
    """Convenience wrapper that holds three dict-like caches."""
    def __init__(self, cache_dir: str):
        self.event2stations = EventToStations(str(Path(cache_dir)/"event2station.sqlite"))
        self.event_attrs = EventToAttrs(str(Path(cache_dir)/"event_attrs.sqlite"))
        self.station_attrs = StationToAttrs(str(Path(cache_dir)/"station_attrs.lmdb"))
    def close(self):
        self.event2stations.con.close()
        self.event_attrs.con.close()
        self.station_attrs.env.close()

# ---------- builders ----------
def build_from_hdf5(h5_paths: List[str], out_dir: str, prefix_mode: str = "none"):
    """
    prefix_mode: 'none' | 'file'  -> if 'file', prefix event_id as '<basename>:<event_path>'
    """
    out_dir = str(Path(out_dir)); Path(out_dir).mkdir(parents=True, exist_ok=True)
    e2sta_sql = str(Path(out_dir) / "event2station.sqlite")
    e_attr_sql = str(Path(out_dir) / "event_attrs.sqlite")
    lmdb_dir = str(Path(out_dir) / "station_attrs.lmdb")

    con1, cur1 = _sqlite_init(e2sta_sql, """
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE map(event_id TEXT PRIMARY KEY, station_ids TEXT);
    """)
    con2, cur2 = _sqlite_init(e_attr_sql, """
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE map(event_id TEXT PRIMARY KEY, attrs TEXT);
    """)
    env = lmdb.open(lmdb_dir, map_size=64*(1<<30), subdir=True, readonly=False,
                    lock=True, readahead=False, map_async=True, max_dbs=1)

    for hp in h5_paths:
        with h5py.File(hp, "r", libver="latest") as f:
            event_ids = list(f.keys())
            print(f"[INFO] processing {hp} with {len(event_ids)} events")
            for event_id in tqdm(event_ids):
                event_grp = f[event_id]
                eid_raw = event_id# if event_id.startswith("/") else "/" + event_id
                eid = f"{Path(hp).name}:{eid_raw}" if prefix_mode == "file" else eid_raw
                station_ids = list(event_grp.keys())
                trace_ids = [f"{event_id}/{sid}" for sid in station_ids]
                
                # event -> station id list
                cur1.execute("INSERT OR REPLACE INTO map VALUES(?,?)",
                             (eid, json.dumps(station_ids, separators=(",",":"))))
                # event attrs (JSON)
                e_attrs = {k: _jsonable(event_grp.attrs[k]) for k in event_grp.attrs.keys()}
                cur2.execute("INSERT OR REPLACE INTO map VALUES(?,?)",
                             (eid, json.dumps(e_attrs, ensure_ascii=False, separators=(",",":"))))
                # station attrs -> LMDB
                with env.begin(write=True) as txn:
                    for sid in station_ids:
                        station = event_grp[sid]
                        if not isinstance(station, h5py.Dataset): continue
                        station_attrs = {k: (station.attrs[k] if not hasattr(station.attrs[k], "tolist") else station.attrs[k].tolist())
                                      for k in station.attrs.keys()}
                        #blob = pack_attrs({k: (np.asarray(v) if isinstance(v, list) else v) for k, v in station_attrs.items()})
                        #station_attrs = dict(station.attrs)
                        blob = pack_attrs(station_attrs)
                        txn.put(lmdb_key(f"{event_id}/{sid}"), blob)

    con1.commit(); con1.close()
    con2.commit(); con2.close()
    env.sync(); env.close()
    print(f"[OK] wrote: {e2sta_sql}, {e_attr_sql}, {lmdb_dir}")

def build_from_csv(event_csv: str, station_csv: str, out_dir: str,
                   array_cols: Optional[List[str]] = None,
                   eid_col: str = "event_id", sid_col: str = "station_id"):
    """
    event_csv columns: event_id, <attr1>, <attr2>, ...
    station_csv columns: event_id, station_id, <attr...>
    array_cols: list of column names in station_csv that encode numpy arrays (as strings)
    """
    array_cols = set(array_cols or [])
    out_dir = str(Path(out_dir)); Path(out_dir).mkdir(parents=True, exist_ok=True)
    e2sta_sql = str(Path(out_dir) / "event2station.sqlite")
    e_attr_sql = str(Path(out_dir) / "event_attrs.sqlite")
    lmdb_dir = str(Path(out_dir) / "station_attrs.lmdb")

    con1, cur1 = _sqlite_init(e2sta_sql, """
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE map(event_id TEXT PRIMARY KEY, dsets TEXT);
    """)
    con2, cur2 = _sqlite_init(e_attr_sql, """
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE map(event_id TEXT PRIMARY KEY, attrs TEXT);
    """)

    # 1) event attrs
    g_has = set()
    with open(event_csv, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            eid = row[eid_col]
            e_attrs = {k: (json.loads(v) if v and k != eid_col else v)
                      for k, v in row.items() if k != eid_col and v not in (None, "", "null")}
            cur2.execute("INSERT OR REPLACE INTO map VALUES(?,?)",
                         (eid, json.dumps(e_attrs, ensure_ascii=False, separators=(",",":"))))
            g_has.add(eid)

    # 2) station attrs -> LMDB + build event->dsets
    env = lmdb.open(lmdb_dir, map_size=64*(1<<30), subdir=True,
                    readonly=False, lock=True, readahead=False, map_async=True, max_dbs=1)
    e2sta = {}  # event_id -> list of sids (dedup)
    with open(station_csv, newline="") as f, env.begin(write=True) as txn:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr, 1):
            eid = row[eid_col]; sid = row[sid_col]
            e2sta.setdefault(eid, []).append(sid)
            # parse attrs (arrays vs scalars)
            dattrs = {}
            for k, v in row.items():
                if k in (eid_col, sid_col) or v in (None, "", "null"):
                    continue
                if k in array_cols:
                    dattrs[k] = decode_ndarray_from_str(v)
                else:
                    # try JSON scalar else keep string
                    try:
                        dattrs[k] = json.loads(v)
                    except Exception:
                        dattrs[k] = v
            blob = pack_attrs(dattrs)
            txn.put(lmdb_key(eid, sid), blob)
            if i % 2000 == 0:
                txn.commit(); txn = env.begin(write=True)

    # write event2station
    for eid, dlist in e2sta.items():
        cur1.execute("INSERT OR REPLACE INTO map VALUES(?,?)",
                     (eid, json.dumps(sorted(set(dlist)), separators=(",",":"))))

    con1.commit(); con1.close()
    con2.commit(); con2.close()
    env.sync(); env.close()
    print(f"[OK] wrote: {e2sta_sql}, {e_attr_sql}, {lmdb_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default='/work/lambda3/WZ/wanghy/quakeflow_dataset_metadata', required=True, help="output dir for three caches")
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("from-hdf5")
    a.add_argument("--h5", nargs="+", required=True)
    a.add_argument("--prefix", choices=["none","file"], default="none")

    b = sub.add_parser("from-csv")
    b.add_argument("--event-csv", required=True)
    b.add_argument("--station-csv", required=True)
    b.add_argument("--array-cols", default="", help="comma-separated columns in station_csv that are ndarray strings")
    b.add_argument("--eid-col", default="event_id")
    b.add_argument("--sid-col", default="station_id")

    args = ap.parse_args()
    if args.mode == "from-hdf5":
        print("[INFO] building from HDF5 files...")
        build_from_hdf5(args.h5, args.out, prefix_mode=args.prefix)
    else:
        print("[INFO] building from CSV files...")
        cols = [c.strip() for c in args.array_cols.split(",")] if args.array_cols else []
        build_from_csv(args.event_csv, args.station_csv, args.out, cols, args.eid_col, args.sid_col)

if __name__ == "__main__":
    main()