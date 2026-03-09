import re


def _validate(coord: float, is_lat: bool) -> float:
    limit = 90.0 if is_lat else 180.0
    name = "latitude" if is_lat else "longitude"
    if coord < -limit or coord > limit:
        raise ValueError(f"{name} out of range: {coord}")
    return float(coord)


def parse_coordinate(value: str, is_lat: bool) -> float:
    s = str(value).strip()
    if not s:
        raise ValueError("empty coordinate")

    # Fast path: decimal degrees
    try:
        return _validate(float(s), is_lat)
    except ValueError:
        pass

    raw = s.upper().replace(",", ".")
    hemi = None
    for h in ("N", "S", "E", "W"):
        if h in raw:
            hemi = h
            raw = raw.replace(h, " ")

    # Preferred dot-separated DMS format: DD.MM.SS[N|S|E|W]
    m = re.match(r"^\s*([+-]?\d+)\.(\d+)\.(\d+(?:\.\d+)?)\s*$", raw.strip())
    if m:
        deg = float(m.group(1))
        minute = float(m.group(2))
        second = float(m.group(3))
        sign = -1.0 if deg < 0 else 1.0
        dec = abs(deg) + minute / 60.0 + second / 3600.0
        if hemi in ("S", "W"):
            sign = -1.0
        elif hemi in ("N", "E"):
            sign = 1.0
        return _validate(sign * dec, is_lat)

    cleaned = re.sub(r"[D°'\"MS:]+", " ", raw)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not nums:
        raise ValueError(f"invalid coordinate: {value}")

    if len(nums) == 1:
        deg = float(nums[0])
        sign = -1.0 if deg < 0 else 1.0
        dec = abs(deg)
    else:
        deg = float(nums[0])
        minute = float(nums[1]) if len(nums) > 1 else 0.0
        second = float(nums[2]) if len(nums) > 2 else 0.0
        sign = -1.0 if deg < 0 else 1.0
        dec = abs(deg) + minute / 60.0 + second / 3600.0

    if hemi in ("S", "W"):
        sign = -1.0
    elif hemi in ("N", "E"):
        sign = 1.0

    return _validate(sign * dec, is_lat)


def decimal_to_dms(value: float, is_lat: bool) -> tuple[int, int, float, str]:
    v = _validate(float(value), is_lat)
    hemi = ("N" if v >= 0 else "S") if is_lat else ("E" if v >= 0 else "W")
    a = abs(v)
    deg = int(a)
    m_full = (a - deg) * 60.0
    minute = int(m_full)
    second = (m_full - minute) * 60.0

    # Normalize round-off overflow.
    second = round(second, 2)
    if second >= 60.0:
        second = 0.0
        minute += 1
    if minute >= 60:
        minute = 0
        deg += 1

    return deg, minute, second, hemi


def format_dms(value: float, is_lat: bool) -> str:
    deg, minute, second, hemi = decimal_to_dms(value, is_lat)
    sec_int = int(round(second))
    if sec_int >= 60:
        sec_int = 0
        minute += 1
    if minute >= 60:
        minute = 0
        deg += 1
    sec_txt = f"{sec_int:02d}"
    return f"{deg}.{minute:02d}.{sec_txt}{hemi}"


def format_lat_dms(value: float) -> str:
    return format_dms(value, is_lat=True)


def format_lon_dms(value: float) -> str:
    return format_dms(value, is_lat=False)
