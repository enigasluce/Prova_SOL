import io
import re
from datetime import timezone as dt_timezone, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pvlib
from pvlib import solarposition

st.set_page_config(page_title="PV String Length", layout="wide")
st.title("PV String Length")
st.caption("Calcolo POA bifacciale, temperatura di cella e Voc Todd-Karin. V1 e Beta Voc sono modificabili per adattare il calcolo al modulo selezionato; gli altri parametri specialistici restano fissati come nel notebook originale.")

# Parametri specialistici fissati come nel codice originale
SHADE_FACTOR = -0.15
TRANSMISSION_FACTOR = -0.01
DELTA_POA = 0.04
IAM_B = 0.05
IAM_BACK = 0.98
T1 = 25.0
G1 = 1000.0
N_DIODE = 1.38
KB = 1.380649e-23
Q = 1.602176634e-19
REQUIRED_METEO = ["GHI", "DNI", "DIF", "TEMP", "WS"]


def detect_header(text):
    for row_number, line in enumerate(text.splitlines()):
        for sep in (";", ",", "\t"):
            names = {x.strip().strip('"') for x in line.split(sep)}
            if "Time" in names and ({"Date", "Day"} & names) and set(REQUIRED_METEO).issubset(names):
                return sep, row_number
    raise ValueError("Intestazione non riconosciuta. Servono Time, GHI, DNI, DIF, TEMP, WS e Date oppure Day.")


def read_utc_offset(text):
    match = re.search(r"time\s+reference\s+UTC\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", text, re.IGNORECASE)
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    return sign * (int(match.group(2)) + int(match.group(3) or 0) / 60)


@st.cache_data(show_spinner=False, max_entries=2)
def read_input(file_bytes, filename):
    text = file_bytes.decode("utf-8-sig", errors="replace")
    sep, header_row = detect_header(text)
    header = pd.read_csv(io.StringIO(text), sep=sep, skiprows=header_row, nrows=0, comment="#")
    header.columns = [str(c).strip() for c in header.columns]
    time_col = "Date" if "Date" in header.columns else "Day"
    wanted = [time_col, "Time", *REQUIRED_METEO]
    missing = [c for c in wanted if c not in header.columns]
    if missing:
        raise ValueError("Colonne mancanti: " + ", ".join(missing))

    # Carica solo le 7 colonne necessarie, non tutte le colonne Solargis.
    dtype = {"GHI": "float32", "DNI": "float32", "DIF": "float32", "TEMP": "float32", "WS": "float32"}
    df = pd.read_csv(
        io.StringIO(text), sep=sep, skiprows=header_row, comment="#",
        usecols=wanted, dtype=dtype, low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]
    metadata = {
        "filename": filename,
        "format": "TShourly (Date + Time)" if time_col == "Date" else "TMY P50 (Day + Time)",
        "separator": "TAB" if sep == "\t" else sep,
        "header_row": header_row + 1,
        "utc_offset": read_utc_offset(text),
    }
    return df, metadata


def build_index(df, metadata, timezone_input, tmy_year):
    time_text = df["Time"].astype(str).str.strip()
    if "Date" in df.columns:
        timestamp = pd.to_datetime(df["Date"].astype(str).str.strip() + " " + time_text,
                                   format="%d.%m.%Y %H:%M", errors="coerce")
    else:
        day = pd.to_numeric(df["Day"], errors="coerce")
        hm = time_text.str.extract(r"^(\d{1,2}):(\d{2})$")
        timestamp = (pd.Timestamp(int(tmy_year), 1, 1)
                     + pd.to_timedelta(day - 1, unit="D")
                     + pd.to_timedelta(pd.to_numeric(hm[0]), unit="h")
                     + pd.to_timedelta(pd.to_numeric(hm[1]), unit="m"))
    bad = int(pd.isna(timestamp).sum())
    if bad:
        raise ValueError(f"{bad} timestamp non validi.")
    if timezone_input.strip().lower() == "auto":
        offset = metadata.get("utc_offset")
        tz = dt_timezone(timedelta(hours=offset)) if offset is not None else "UTC"
    else:
        tz = timezone_input.strip()
    return pd.DatetimeIndex(timestamp).tz_localize(tz, ambiguous="infer", nonexistent="shift_forward"), str(tz)


def downsample_for_chart(df, max_points=5000):
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step]


def calculate(df, metadata, p):
    index, tz_used = build_index(df, metadata, p["timezone"], p["tmy_year"])
    out = df.copy()
    out.index = index
    for col in REQUIRED_METEO:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace(-9, np.nan).astype("float32")
    bad_rows = out[REQUIRED_METEO].isna().any(axis=1)
    dropped = int(bad_rows.sum())
    out = out.loc[~bad_rows]
    if out.empty:
        raise ValueError("Nessuna riga meteo valida.")

    ghi = out["GHI"].clip(lower=0)
    dni = out["DNI"].clip(lower=0)
    dhi = out["DIF"].clip(lower=0)
    temp_air = out["TEMP"]
    wind = out["WS"].clip(lower=0)
    solpos = solarposition.get_solarposition(out.index, p["lat"], p["lon"], altitude=p["alt"])
    dni_extra = pvlib.irradiance.get_extra_radiation(out.index)
    actual_gcr = p["gcr"] if p["gcr_manual"] else p["width"] / p["pitch"]
    if not 0 < actual_gcr <= 1:
        raise ValueError(f"GCR {actual_gcr:.3f} non valido: deve essere compreso tra 0 e 1.")

    if p["structure"] == "Tracker monoassiale":
        orient = pvlib.tracking.singleaxis(
            solpos["apparent_zenith"], solpos["azimuth"], axis_tilt=0,
            axis_azimuth=p["azimuth"], max_angle=p["max_angle"],
            backtrack=p["backtrack"], gcr=actual_gcr, cross_axis_tilt=p["cross_axis_tilt"],
        )
        surface_tilt = orient["surface_tilt"].fillna(0)
        surface_azimuth = orient["surface_azimuth"].fillna(p["azimuth"])
        aoi = orient["aoi"]
    else:
        surface_tilt = pd.Series(p["tilt"], index=out.index)
        surface_azimuth = pd.Series(p["azimuth"], index=out.index)
        aoi = pd.Series(pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                                             solpos["apparent_zenith"], solpos["azimuth"]), index=out.index)

    iam_front = pd.Series(pvlib.iam.ashrae(aoi, b=IAM_B), index=out.index).fillna(0)
    poa = pvlib.bifacial.infinite_sheds.get_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        gcr=actual_gcr, height=p["height"], pitch=p["pitch"], dni_extra=dni_extra,
        albedo=p["albedo"], dni=dni, ghi=ghi, dhi=dhi, bifaciality=p["bifaciality"],
        iam_front=iam_front, iam_back=IAM_BACK, shade_factor=SHADE_FACTOR,
        transmission_factor=TRANSMISSION_FACTOR, npoints=100, vectorize=False,
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"], model="haydavies",
    )
    poa_front = poa["poa_front"].fillna(0).clip(lower=0) * (1 - p["soiling"])
    poa_back = poa["poa_back"].fillna(0).clip(lower=0)
    poa_global = ((poa_front + poa_back) * (1 + DELTA_POA)).clip(lower=0)
    del poa, solpos

    t_faiman = pvlib.temperature.faiman(poa_global, temp_air, wind, u0=26.9, u1=6.2)
    t_pvsyst = pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind, u_c=25, u_v=1.2,
                                              module_efficiency=0.23, alpha_absorption=0.9)
    t_king = pvlib.temperature.sapm_cell(poa_global, temp_air, wind, a=-3.47, b=-0.0594,
                                         deltaT=3, irrad_ref=1000.0)
    t_cell = (t_faiman + t_pvsyst + t_king) / 3
    a_todd = N_DIODE * KB * (t_cell + 273.15) / Q
    voc = pd.Series(np.nan, index=out.index, dtype="float32")
    daylight = poa_global > 0
    v1 = p["v1"]
    beta = p["beta"]
    voc.loc[daylight] = (v1 + v1 * beta * (t_cell.loc[daylight] - T1)
                         + v1 * a_todd.loc[daylight] * np.log(poa_global.loc[daylight] / G1))

    result = pd.DataFrame({
        "GHI_Wm2": ghi.astype("float32"),
        "POA_front_Wm2": poa_front.astype("float32"),
        "POA_back_Wm2": poa_back.astype("float32"),
        "POA_global_Wm2": poa_global.astype("float32"),
        "T_amb_C": temp_air.astype("float32"),
        "T_cell_C": t_cell.astype("float32"),
        "Voc_corrected_V": voc,
    })
    valid = result.dropna(subset=["Voc_corrected_V"])
    if valid.empty:
        raise ValueError("Nessuna ora con POA maggiore di zero.")

    p995 = valid["Voc_corrected_V"].quantile(0.995)
    vmax = valid["Voc_corrected_V"].max()
    row995 = valid.iloc[np.abs(valid["Voc_corrected_V"].to_numpy() - p995).argmin()]
    rowmax = valid.loc[valid["Voc_corrected_V"].idxmax()]
    filtered = valid[valid["POA_global_Wm2"] > 150]
    tmin_pvsyst = filtered["T_cell_C"].min() if not filtered.empty else np.nan
    a_pvsyst = N_DIODE * KB * (tmin_pvsyst + 273.15) / Q if np.isfinite(tmin_pvsyst) else np.nan
    voc_pvsyst = (v1 + v1 * beta * (tmin_pvsyst - T1)
                  + v1 * a_pvsyst * np.log(1000 / G1)) if np.isfinite(tmin_pvsyst) else np.nan
    summary = pd.DataFrame({
        "Voc [V]": [p995, vmax, voc_pvsyst],
        "Temperatura cella [degC]": [row995["T_cell_C"], rowmax["T_cell_C"], tmin_pvsyst],
        "POA [W/m2]": [row995["POA_global_Wm2"], rowmax["POA_global_Wm2"], 1000],
        "Moduli teorici": [p["vdc_max"] / p995, p["vdc_max"] / vmax, p["vdc_max"] / voc_pvsyst],
        "Moduli interi max": [np.floor(p["vdc_max"] / p995), np.floor(p["vdc_max"] / vmax), np.floor(p["vdc_max"] / voc_pvsyst)],
    }, index=["P99.5", "Massimo", "PVsyst"]).round(2)
    return result, summary, actual_gcr, dropped, tz_used


with st.expander("Formati CSV supportati", expanded=False):
    st.markdown("""
- **TShourly:** `Date;Time;GHI;DNI;DIF;...;TEMP;...;WS;...`
- **TMY P50:** `Day;Time;GHI;DNI;DIF;...;TEMP;...;WS;...`

L'app cerca automaticamente l'intestazione e carica solo le colonne necessarie per ridurre la memoria.
    """)

uploaded = st.file_uploader("1. Carica il file meteo CSV", type=["csv", "txt"])

with st.sidebar:
    st.header("Posizione")
    latitude = st.number_input("Latitudine [deg]", -90.0, 90.0, 37.178550, format="%.6f")
    longitude = st.number_input("Longitudine [deg]", -180.0, 180.0, 14.650785, format="%.6f")
    altitude = st.number_input("Altitudine [m]", value=308.0)
    timezone_input = st.text_input("Timezone", "auto", help="Auto legge UTC dai metadati Solargis.")
    tmy_year = st.number_input("Anno di riferimento TMY", 1901, 2099, 2001)

    st.header("Layout")
    structure = st.radio("Configurazione", ["Tracker monoassiale", "Struttura fissa"])
    pitch = st.number_input("Pitch [m]", min_value=0.01, value=10.0)
    width = st.number_input("Larghezza fila [m]", min_value=0.01, value=4.8)
    gcr_manual = st.checkbox("Inserisci GCR direttamente", value=False)
    gcr = st.number_input("GCR [-]", 0.01, 1.0, 0.48, disabled=not gcr_manual)
    height = st.number_input("Altezza [m]", min_value=0.0, value=3.0)
    azimuth = st.number_input("Axis/surface azimuth [deg]", 0.0, 360.0, 180.0)
    if structure == "Struttura fissa":
        tilt = st.number_input("Tilt [deg]", 0.0, 90.0, 15.0)
        max_angle, backtrack, cross_axis_tilt = 55.0, False, 0.0
    else:
        max_angle = st.number_input("Max angle [deg]", 0.0, 90.0, 55.0)
        backtrack = st.checkbox("Backtracking", value=True)
        cross_axis_tilt = st.number_input("Cross-axis tilt [deg]", -90.0, 90.0, 0.0)
        tilt = 15.0

    st.header("Parametri modulo")
    v1 = st.number_input("Voc STC V1 [V]", min_value=0.01, value=49.74, step=0.01, help="Tensione a circuito aperto del modulo alle condizioni STC.")
    beta = st.number_input("Beta Voc [1/degC]", value=-0.0026, step=0.0001, format="%.6f", help="Coefficiente relativo di temperatura della Voc del modulo.")
    vdc_max = st.number_input("Tensione massima sistema [V]", min_value=1.0, value=1500.0)

    st.header("Parametri impianto essenziali")
    bifaciality = st.number_input("Bifaciality [-]", 0.0, 1.0, 0.80)
    albedo = st.number_input("Albedo [-]", 0.0, 1.0, 0.10)
    soiling = st.number_input("Soiling frontale [-]", 0.0, 1.0, 0.00)
if uploaded is None:
    st.info("Carica un CSV, imposta i parametri e premi il pulsante di calcolo.")
else:
    try:
        data, metadata = read_input(uploaded.getvalue(), uploaded.name)
        st.success(f"File riconosciuto: {metadata['format']} | {len(data):,} righe".replace(",", "."))
        st.caption("Il calcolo non parte automaticamente. Dopo aver controllato i parametri, premi il pulsante seguente.")
        run = st.button("2. CALCOLA LUNGHEZZA STRINGA E GRAFICI", type="primary", use_container_width=True)
        if run:
            params = {
                "lat": latitude, "lon": longitude, "alt": altitude, "timezone": timezone_input,
                "tmy_year": tmy_year, "structure": structure, "pitch": pitch, "width": width,
                "gcr_manual": gcr_manual, "gcr": gcr, "height": height, "azimuth": azimuth,
                "tilt": tilt, "max_angle": max_angle, "backtrack": backtrack,
                "cross_axis_tilt": cross_axis_tilt, "bifaciality": bifaciality,
                "albedo": albedo, "soiling": soiling, "v1": v1, "beta": beta, "vdc_max": vdc_max,
            }
            with st.spinner("Calcolo in corso..."):
                result, summary, actual_gcr, dropped, tz_used = calculate(data, metadata, params)
            st.success(f"Calcolo completato. GCR usato: {actual_gcr:.3f}; timezone: {tz_used}.")
            if dropped:
                st.warning(f"Escluse {dropped} righe con dati mancanti o valore Solargis -9.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Voc P99.5", f"{summary.loc['P99.5', 'Voc [V]']:.2f} V")
            c2.metric("Voc massima", f"{summary.loc['Massimo', 'Voc [V]']:.2f} V")
            c3.metric("Moduli interi max P99.5", f"{int(summary.loc['P99.5', 'Moduli interi max'])}")
            st.subheader("Lunghezza stringa")
            st.dataframe(summary, use_container_width=True)

            chart = downsample_for_chart(result)
            st.caption(f"Grafici alleggeriti a {len(chart):,} punti; il CSV scaricabile contiene tutte le {len(result):,} righe.".replace(",", "."))
            st.plotly_chart(px.line(chart.reset_index(names="timestamp"), x="timestamp", y="Voc_corrected_V",
                                      labels={"timestamp": "Timestamp", "Voc_corrected_V": "Voc [V]"},
                                      title="Voc corretta ora per ora"), use_container_width=True)
            col1, col2 = st.columns(2)
            col1.plotly_chart(px.line(chart.reset_index(names="timestamp"), x="timestamp", y="POA_global_Wm2",
                                       labels={"timestamp":"Timestamp", "POA_global_Wm2":"POA [W/m2]"}, title="POA globale"), use_container_width=True)
            col2.plotly_chart(px.line(chart.reset_index(names="timestamp"), x="timestamp", y="T_cell_C",
                                       labels={"timestamp":"Timestamp", "T_cell_C":"Temperatura [degC]"}, title="Temperatura di cella"), use_container_width=True)
            st.download_button("Scarica risultati completi CSV", result.to_csv().encode("utf-8"),
                               "risultati_voc_orari.csv", "text/csv", use_container_width=True)
    except Exception as exc:
        st.error(f"Errore: {exc}")
