import math
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="OrbitalMind AI",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-HIGH { color: #ff4b4b; font-weight: bold; }
    .risk-MED  { color: #ffa500; font-weight: bold; }
    .risk-LOW  { color: #21c45d; font-weight: bold; }
    .stDataFrame td { font-family: monospace; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# =========================
# CONSTANTS
# =========================
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Risk thresholds in km.
# Based on US Space Fence screening volumes (approximate):
#   radial ±5 km, along-track ±25 km, cross-track ±25 km → ~10 km miss distance as danger zone.
#   50 km used as outer "watch" threshold (common operational practice).
RISK_HIGH_KM = 10.0
RISK_MED_KM  = 50.0

KNOWN_CELESTRAK_GROUPS = [
    "starlink", "oneweb", "active", "stations", "visual",
    "cosmos-deb", "iridium-33-debris", "fengyun-1c-debris",
    "muos", "gps-ops", "galileo", "beidou", "glonass-ops",
]


# =========================
# HELPERS
# =========================
def resolve_path(filename: str) -> Optional[str]:
    candidates = [
        os.path.join("data", filename),
        filename,
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def norm(v) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def sub(a, b) -> Tuple:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def risk_level(min_d: Optional[float]) -> str:
    """
    Classify conjunction risk by miss distance.
    Thresholds based on common operational screening values (see RISK_HIGH_KM / RISK_MED_KM).
    """
    if min_d is None or math.isinf(min_d):
        return "N/A"
    if min_d < RISK_HIGH_KM:
        return "HIGH"
    if min_d < RISK_MED_KM:
        return "MED"
    return "LOW"


def sgp4_pos_vel_km(sat: Satrec, dt: datetime):
    jd, fr = jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute,
        dt.second + dt.microsecond / 1e6,
    )
    e, r, v = sat.sgp4(jd, fr)
    return e, r, v


# =========================
# ML — COLLISION RISK MODEL
# =========================
def extract_orbital_features(sat: Satrec) -> Dict:
    """
    Извлекает орбитальные параметры из объекта Satrec для ML модели.

    Признаки (features):
      - mean_motion     : среднее движение (об/день) — связано с высотой орбиты
      - eccentricity    : эксцентриситет — форма орбиты (0=круг, 1=парабола)
      - inclination_deg : наклонение орбиты в градусах
      - bstar           : коэффициент баллистического торможения (плотность атмосферы)
      - altitude_km     : приблизительная высота орбиты (км)
    """
    mu = 398600.4418          # гравитационный параметр Земли (км³/с²)
    mean_motion_rad_s = sat.nm  # рад/сек
    mean_motion_rev_day = sat.nm * 86400 / (2 * math.pi)

    # Большая полуось орбиты через третий закон Кеплера: a = (mu / n²)^(1/3)
    if mean_motion_rad_s > 0:
        semi_major_axis = (mu / (mean_motion_rad_s ** 2)) ** (1 / 3)
        altitude_km = semi_major_axis - 6371.0
    else:
        altitude_km = 0.0

    return {
        "mean_motion":     mean_motion_rev_day,
        "eccentricity":    sat.ecco,
        "inclination_deg": math.degrees(sat.inclo),
        "bstar":           sat.bstar,
        "altitude_km":     altitude_km,
    }


def build_ml_model() -> Tuple[Pipeline, Pipeline]:
    """
    Создаёт и обучает две ML модели на синтетических данных:

    1. risk_clf  — RandomForestClassifier (HIGH / MED / LOW)
       Классифицирует уровень риска по орбитальным параметрам.

    2. pc_reg    — GradientBoostingRegressor
       Предсказывает вероятность столкновения (Probability of Collision, Pc) в %.

    Синтетические данные генерируются на основе физических закономерностей:
      - Объекты на низких орбитах (LEO, 200–600 км) с высоким наклонением → выше риск
      - Высокий эксцентриситет → нестабильная орбита → выше риск
      - Высокий bstar → сильное торможение → непредсказуемое поведение
    """
    np.random.seed(42)
    N = 2000

    # Генерируем реалистичные орбитальные параметры
    altitude    = np.random.uniform(200, 2000, N)
    inclination = np.random.uniform(0, 98, N)
    eccentricity= np.random.uniform(0.0, 0.05, N)
    bstar       = np.random.uniform(-0.001, 0.01, N)
    mean_motion = 86400 / (2 * math.pi) * np.sqrt(398600.4418 / ((altitude + 6371) ** 3))

    # Физически обоснованная формула риска:
    # LEO (низкая орбита) + высокое наклонение + высокий эксцентриситет = выше риск
    risk_score = (
        (1 - altitude / 2000) * 0.5        # ниже орбита → выше риск
        + (inclination / 98)  * 0.25       # полярные орбиты пересекают больше плоскостей
        + eccentricity        * 3.0        # эллиптические орбиты опаснее
        + np.clip(bstar, 0, 0.01) * 20     # высокое торможение → непредсказуемость
        + np.random.normal(0, 0.05, N)     # шум
    )

    # Метки классов
    labels = np.where(risk_score > 0.75, "HIGH",
              np.where(risk_score > 0.45, "MED", "LOW"))

    # Вероятность столкновения (условная, в %)
    pc = np.clip(risk_score * 0.8 + np.random.normal(0, 0.02, N), 0, 1) * 100

    X = np.column_stack([mean_motion, eccentricity, inclination, bstar, altitude])

    # Классификатор риска
    risk_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    risk_clf.fit(X, labels)

    # Регрессор вероятности столкновения
    pc_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )),
    ])
    pc_reg.fit(X, pc)

    return risk_clf, pc_reg


def predict_risk_ml(
    sat: Satrec,
    risk_clf: Pipeline,
    pc_reg: Pipeline,
) -> Tuple[str, float, Dict]:
    """
    Предсказывает риск и вероятность столкновения для спутника.

    Возвращает:
      - ml_risk  : "HIGH" / "MED" / "LOW"
      - ml_pc    : вероятность столкновения в %
      - features : словарь с орбитальными параметрами (для отображения)
    """
    feats = extract_orbital_features(sat)
    X = np.array([[
        feats["mean_motion"],
        feats["eccentricity"],
        feats["inclination_deg"],
        feats["bstar"],
        feats["altitude_km"],
    ]])

    ml_risk = risk_clf.predict(X)[0]
    ml_pc   = float(np.clip(pc_reg.predict(X)[0], 0, 100))

    return ml_risk, ml_pc, feats


@st.cache_resource
def get_ml_models() -> Tuple[Pipeline, Pipeline]:
    """Обучает модели один раз и кэширует на всю сессию."""
    return build_ml_model()


# =========================
# TLE PARSING  (fault-tolerant)
# =========================
def parse_tle_text(text: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """
    Parse TLE text into (name, line1, line2) triples.
    Returns (valid_entries, skipped_names) instead of crashing on bad entries.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    objs: List[Tuple[str, str, str]] = []
    skipped: List[str] = []
    i = 0
    while i + 2 <= len(lines) - 1:
        name = lines[i]
        l1   = lines[i + 1]
        l2   = lines[i + 2]

        if l1.startswith("1 ") and l2.startswith("2 "):
            objs.append((name, l1, l2))
            i += 3
        else:
            # Bad entry — skip this line and try to re-sync
            skipped.append(name)
            i += 1

    return objs, skipped


# =========================
# CELESTRAK FETCH  (with error handling)
# =========================
@st.cache_data(ttl=6 * 60 * 60)
def fetch_celestrak_group(group: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (tle_text, error_message).
    On success: (text, None). On failure: (None, error).
    """
    try:
        params = {"GROUP": group, "FORMAT": "tle"}
        r = requests.get(CELESTRAK_URL, params=params, timeout=25)
        r.raise_for_status()
        text = r.text.strip()
        if not text:
            return None, f"CelesTrak returned empty response for group '{group}'."
        return text, None
    except requests.exceptions.Timeout:
        return None, "Request to CelesTrak timed out (25s). Try again later."
    except requests.exceptions.HTTPError as e:
        return None, f"CelesTrak HTTP error: {e}. Check the group name."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"


# =========================
# CONJUNCTION SEARCH  (two-pass)
# =========================
def compute_min_distance_24h(
    client_sat: Satrec,
    other_sat: Satrec,
    coarse_step_sec: int = 60,
    fine_step_sec: int   = 5,
) -> Tuple[float, Optional[datetime], Optional[float]]:
    """
    Two-pass conjunction screening:
      Pass 1 — coarse scan to find approximate minima.
      Pass 2 — fine scan ±2 minutes around each local minimum.

    Returns (min_distance_km, tca_utc, relative_velocity_km_s).
    """
    now      = datetime.utcnow()
    end_time = now + timedelta(hours=24)

    # ---- Pass 1: coarse ----
    prev_d   = float("inf")
    prev_t   = now
    minima_t = []        # timestamps of local minima detected in coarse pass

    t = now
    while t <= end_time:
        e1, r1, v1 = sgp4_pos_vel_km(client_sat, t)
        e2, r2, v2 = sgp4_pos_vel_km(other_sat, t)

        if e1 == 0 and e2 == 0:
            d = norm(sub(r1, r2))
            if d > prev_d:
                # distance started increasing → prev_t was a local minimum
                minima_t.append(prev_t)
            prev_d = d
            prev_t = t

        t += timedelta(seconds=coarse_step_sec)

    # Always check end_time as a candidate
    minima_t.append(end_time)

    # ---- Pass 2: fine scan around each candidate minimum ----
    global_min_d  = float("inf")
    global_tca    = None
    global_rel_v  = None

    refine_window = timedelta(seconds=coarse_step_sec * 2)

    for center_t in minima_t:
        window_start = max(now,      center_t - refine_window)
        window_end   = min(end_time, center_t + refine_window)

        ft = window_start
        while ft <= window_end:
            e1, r1, v1 = sgp4_pos_vel_km(client_sat, ft)
            e2, r2, v2 = sgp4_pos_vel_km(other_sat,  ft)

            if e1 == 0 and e2 == 0:
                d = norm(sub(r1, r2))
                if d < global_min_d:
                    global_min_d = d
                    global_tca   = ft
                    global_rel_v = norm(sub(v1, v2))

            ft += timedelta(seconds=fine_step_sec)

    return global_min_d, global_tca, global_rel_v


# =========================
# 3D ORBIT PLOT
# =========================
def plot_3d_scene(
    client_xyz: Tuple,
    catalog_xyz: List[Tuple],
    catalog_names: List[str],
    highlight_names: Optional[List[str]] = None,
) -> go.Figure:

    highlight_names = set(highlight_names or [])
    fig = go.Figure()

    # ---- Earth sphere ----
    earth_r = 6371
    u = np.linspace(0, 2 * math.pi, 40)
    v = np.linspace(0, math.pi, 40)
    fig.add_trace(go.Surface(
        x=earth_r * np.outer(np.cos(u), np.sin(v)),
        y=earth_r * np.outer(np.sin(u), np.sin(v)),
        z=earth_r * np.outer(np.ones(40), np.cos(v)),
        colorscale="Blues",
        opacity=0.35,
        showscale=False,
        name="Earth",
        hoverinfo="skip",
    ))

    # ---- Catalog objects ----
    normal_xyz  = [p for p, n in zip(catalog_xyz, catalog_names) if n not in highlight_names]
    threat_xyz  = [p for p, n in zip(catalog_xyz, catalog_names) if n in highlight_names]
    threat_nm   = [n for n in catalog_names if n in highlight_names]

    if normal_xyz:
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in normal_xyz],
            y=[p[1] for p in normal_xyz],
            z=[p[2] for p in normal_xyz],
            mode="markers",
            marker=dict(size=2, color="rgba(100,160,220,0.6)"),
            name="Catalog",
            hovertemplate="%{text}<extra></extra>",
            text=[n for n in catalog_names if n not in highlight_names],
        ))

    if threat_xyz:
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in threat_xyz],
            y=[p[1] for p in threat_xyz],
            z=[p[2] for p in threat_xyz],
            mode="markers",
            marker=dict(size=5, color="orange", symbol="diamond"),
            name="⚠ Threats",
            text=threat_nm,
            hovertemplate="%{text}<extra></extra>",
        ))

    # ---- Client satellite ----
    fig.add_trace(go.Scatter3d(
        x=[client_xyz[0]], y=[client_xyz[1]], z=[client_xyz[2]],
        mode="markers+text",
        text=["◉ CLIENT"],
        textfont=dict(color="red", size=11),
        marker=dict(size=7, color="red"),
        name="Client",
    ))

    fig.update_layout(
        height=680,
        paper_bgcolor="#0a0e1a",
        scene=dict(
            aspectmode="data",
            bgcolor="#0a0e1a",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        legend=dict(
            bgcolor="rgba(15,20,35,0.8)",
            bordercolor="#333",
            borderwidth=1,
            font=dict(color="#ccc"),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


# =========================
# SIDEBAR — SETTINGS
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Client Satellite")
    client_source = st.radio(
        "Source",
        ["Paste TLE manually", "First from CelesTrak group", "Load from client_tle.txt"],
        index=0,
    )

    client_tle_text = None

    if client_source == "Paste TLE manually":
        client_tle_raw = st.text_area(
            "Paste 3-line TLE (name + 2 lines)",
            height=120,
            placeholder="ISS (ZARYA)\n1 25544U ...\n2 25544 ...",
        )
        if client_tle_raw.strip():
            client_tle_text = client_tle_raw

    elif client_source == "Load from client_tle.txt":
        path = resolve_path("client_tle.txt")
        if path:
            client_tle_text = read_text_file(path)
            st.success(f"Loaded: {path}")
        else:
            st.error("client_tle.txt not found in project root or /data/")

    else:  # First from CelesTrak group
        bootstrap_group = st.text_input("Bootstrap group", value="stations")
        client_tle_text = f"__celestrak__{bootstrap_group}"  # resolved at run time

    st.divider()
    st.subheader("Catalog")
    catalog_source = st.radio(
        "Catalog source",
        ["CelesTrak group", "Load from catalog_tle.txt"],
        index=0,
    )

    catalog_group = None
    catalog_file_path = None

    if catalog_source == "CelesTrak group":
        catalog_group = st.selectbox("Group", KNOWN_CELESTRAK_GROUPS, index=0)
        custom_group = st.text_input("Or type a custom group name", value="")
        if custom_group.strip():
            catalog_group = custom_group.strip()
    else:
        catalog_file_path = resolve_path("catalog_tle.txt")
        if catalog_file_path:
            st.success(f"Loaded: {catalog_file_path}")
        else:
            st.error("catalog_tle.txt not found.")

    take_n = st.slider("Max catalog objects", 10, 2000, 200, step=10)

    st.divider()
    st.subheader("Simulation")
    coarse_step = st.slider("Coarse step (sec)", 30, 300, 60, step=10,
                             help="Step for initial 24h scan. Smaller = slower but more accurate.")
    fine_step   = st.slider("Fine step (sec)",   1,  30,   5, step=1,
                             help="Step for refinement around detected close approaches.")

    st.divider()
    st.subheader("Risk Thresholds (km)")
    r_high = st.number_input("HIGH if miss < (km)", value=RISK_HIGH_KM, min_value=0.1, step=1.0)
    r_med  = st.number_input("MED if miss < (km)",  value=RISK_MED_KM,  min_value=1.0, step=5.0)

    st.divider()
    st.subheader("🤖 AI / ML")
    use_ml = st.toggle("Enable ML Risk Prediction", value=True,
                       help="Использует RandomForest + GradientBoosting для предсказания риска по орбитальным параметрам")
    if use_ml:
        st.caption("Модель: RandomForest (классификация) + GradientBoosting (Pc%). Обучена на физически смоделированных орбитальных данных.")


# =========================
# MAIN UI
# =========================
st.title("🛰 OrbitalMind AI")
st.caption("Conjunction screening: 1 protected object vs catalog — 24-hour horizon")

col1, col2, col3 = st.columns(3)
col1.metric("Coarse step", f"{coarse_step}s")
col2.metric("Fine step",   f"{fine_step}s")
col3.metric("Catalog size", f"≤ {take_n}")

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


# =========================
# RUN
# =========================
if run_btn:

    # ---- Resolve client satellite ----
    client_sat = None

    if client_source == "First from CelesTrak group":
        text, err = fetch_celestrak_group(bootstrap_group)
        if err:
            st.error(f"Failed to fetch client group: {err}")
            st.stop()
        client_tle_text = text

    if not client_tle_text or not client_tle_text.strip():
        st.error("No client TLE provided. Please paste a TLE or load a file.")
        st.stop()

    try:
        client_objs, client_skipped = parse_tle_text(client_tle_text)
        if not client_objs:
            st.error("Could not parse any valid TLE from client source.")
            st.stop()
        client_name, cl1, cl2 = client_objs[0]
        client_sat = Satrec.twoline2rv(cl1, cl2)
        st.info(f"🛰 Client satellite: **{client_name}**")
    except Exception as ex:
        st.error(f"Error initialising client satellite: {ex}")
        st.stop()

    # ---- Resolve catalog ----
    catalog_text = None

    if catalog_source == "CelesTrak group":
        with st.spinner(f"Fetching catalog '{catalog_group}' from CelesTrak…"):
            catalog_text, err = fetch_celestrak_group(catalog_group)
        if err:
            st.error(f"Failed to fetch catalog: {err}")
            st.stop()
    else:
        if not catalog_file_path:
            st.error("catalog_tle.txt not found.")
            st.stop()
        catalog_text = read_text_file(catalog_file_path)

    catalog_objs, cat_skipped = parse_tle_text(catalog_text)
    catalog_objs = catalog_objs[:take_n]

    if cat_skipped:
        st.warning(f"Skipped {len(cat_skipped)} malformed TLE entries in catalog.")

    if not catalog_objs:
        st.error("No valid TLE objects in catalog.")
        st.stop()

    st.write(f"Screening **{len(catalog_objs)}** objects…")

    # ---- Загрузка ML моделей ----
    risk_clf, pc_reg = None, None
    if use_ml:
        with st.spinner("🤖 Загружаю ML модели…"):
            risk_clf, pc_reg = get_ml_models()
        st.success("✅ ML модели готовы (RandomForest + GradientBoosting)")

    # ---- Conjunction search ----
    results       = []
    pos_cache: Dict[str, Tuple] = {}   # reuse positions for 3D plot
    sim_time      = datetime.utcnow()

    # Pre-compute client position for 3D
    ec, r_client, _ = sgp4_pos_vel_km(client_sat, sim_time)

    progress_bar = st.progress(0, text="Analysing…")
    status_text  = st.empty()

    for idx, (name, l1, l2) in enumerate(catalog_objs):
        sat = Satrec.twoline2rv(l1, l2)

        min_d, tca, rel_v = compute_min_distance_24h(
            client_sat, sat,
            coarse_step_sec=coarse_step,
            fine_step_sec=fine_step,
        )

        # Cache current position for 3D (reuse — no double propagation)
        e2, r2, _ = sgp4_pos_vel_km(sat, sim_time)
        if e2 == 0:
            pos_cache[name] = (r2[0], r2[1], r2[2])

        # ---- ML предсказание ----
        ml_risk, ml_pc, feats = None, None, {}
        if use_ml and risk_clf is not None:
            ml_risk, ml_pc, feats = predict_risk_ml(sat, risk_clf, pc_reg)

        results.append({
            "Object":               name,
            "Min Distance (km)":    round(min_d, 3) if not math.isinf(min_d) else None,
            "TCA (UTC)":            tca.strftime("%Y-%m-%d %H:%M:%S") if tca else "N/A",
            "Rel. Velocity (km/s)": round(rel_v, 3) if rel_v else None,
            "Risk (geometry)":      risk_level(min_d),
            "Risk (ML)":            ml_risk or "—",
            "Pc % (ML)":            round(ml_pc, 2) if ml_pc is not None else None,
            "Altitude (km)":        round(feats.get("altitude_km", 0), 1) if feats else None,
            "Inclination (°)":      round(feats.get("inclination_deg", 0), 2) if feats else None,
        })

        # Update progress every 10 objects
        if idx % 10 == 0 or idx == len(catalog_objs) - 1:
            pct = (idx + 1) / len(catalog_objs)
            progress_bar.progress(pct, text=f"Analysing… {idx+1}/{len(catalog_objs)}")
            status_text.text(f"Last checked: {name}")

    progress_bar.empty()
    status_text.empty()

    # ---- Build & sort dataframe ----
    df = pd.DataFrame(results)
    df = df.sort_values("Min Distance (km)", ascending=True, na_position="last")
    df.insert(0, "№", range(1, len(df) + 1))

    # ---- Summary metrics ----
    n_high = (df["Risk (geometry)"] == "HIGH").sum()
    n_med  = (df["Risk (geometry)"] == "MED").sum()
    n_low  = (df["Risk (geometry)"] == "LOW").sum()
    n_ml_high = (df["Risk (ML)"] == "HIGH").sum() if use_ml else 0

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Total screened", len(df))
    mc2.metric("🔴 HIGH (геом.)",  n_high)
    mc3.metric("🟡 MED (геом.)",   n_med)
    mc4.metric("🟢 LOW (геом.)",   n_low)
    mc5.metric("🤖 HIGH (ML)",     n_ml_high)

    # ---- ML объяснение ----
    if use_ml:
        with st.expander("🤖 Как работает ML модель?"):
            st.markdown("""
**Две модели работают параллельно с геометрическим анализом:**

| Модель | Алгоритм | Задача |
|--------|----------|--------|
| Классификатор риска | RandomForestClassifier (150 деревьев) | HIGH / MED / LOW |
| Оценщик вероятности | GradientBoostingRegressor (100 итераций) | Pc % |

**Входные признаки (features):**
- `altitude_km` — высота орбиты: чем ниже LEO, тем плотнее трафик
- `inclination_deg` — наклонение: полярные орбиты пересекают все плоскости
- `eccentricity` — эксцентриситет: эллиптические орбиты = непредсказуемые перигеи
- `mean_motion` — среднее движение: связано с орбитальным периодом
- `bstar` — баллистический коэффициент: торможение в атмосфере

**Интерпретация:**  
`Risk (geometry)` = реальное минимальное расстояние за 24ч  
`Risk (ML)` = предсказание по орбитальным параметрам без расчёта траектории  
`Pc %` = вероятность столкновения (условная оценка модели)
            """)

    # ---- Results table ----
    st.subheader("Conjunction Results (sorted by miss distance)")

    col_config = {
        "Min Distance (km)": st.column_config.NumberColumn(format="%.3f km"),
        "Rel. Velocity (km/s)": st.column_config.NumberColumn(format="%.3f km/s"),
        "Pc % (ML)": st.column_config.ProgressColumn(
            "Pc % (ML)", min_value=0, max_value=100, format="%.2f%%"
        ) if use_ml else st.column_config.NumberColumn(),
    }

    display_cols = ["№", "Object", "Min Distance (km)", "TCA (UTC)",
                    "Rel. Velocity (km/s)", "Risk (geometry)"]
    if use_ml:
        display_cols += ["Risk (ML)", "Pc % (ML)", "Altitude (km)", "Inclination (°)"]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )

    # ---- Download button ----
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv,
        file_name=f"conjunction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    # ---- 3D Visualisation ----
    st.subheader("3D Orbital View")

    catalog_positions = list(pos_cache.values())
    catalog_names_3d  = list(pos_cache.keys())
    threat_names = df[df["Risk (geometry)"] == "HIGH"]["Object"].tolist()

    if ec == 0:
        fig = plot_3d_scene(
            (r_client[0], r_client[1], r_client[2]),
            catalog_positions,
            catalog_names_3d,
            highlight_names=threat_names,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not compute client satellite position for 3D plot (SGP4 error).")
