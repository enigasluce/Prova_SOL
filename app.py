import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pvlib
from pvlib import solarposition

st.set_page_config(page_title="POA, temperatura cella e Voc Todd-Karin", layout="wide")
st.title("Calcolo POA, temperatura di cella e Voc Todd-Karin")
st.caption("Implementazione limitata alla sezione 'Corrected Voc: tutte le temperature' del notebook fornito.")

with st.expander("Formato CSV richiesto", expanded=True):
    st.markdown("""
Il file deve contenere una riga per intervallo temporale e le colonne seguenti:
- `Date`: data nel formato `GG.MM.AAAA`
- `Time`: ora nel formato `HH:MM`
- `GHI`, `DNI`, `DIF`: irradianza in W/m2
- `TEMP`: temperatura ambiente in degC
- `WS`: velocita del vento in m/s

Sono accettati CSV separati da `;` oppure `,`. Eventuali righe descrittive prima dell'intestazione vengono rilevate automaticamente.
    """)
    example = "Date;Time;GHI;DNI;DIF;TEMP;WS\n01.01.2025;00:00;0;0;0;7.2;2.1\n01.01.2025;12:00;520;710;95;14.8;3.4\n"
    st.code(example, language="text")
    st.download_button("Scarica CSV di esempio", example.encode("utf-8"), "formato_input.csv", "text/csv")

uploaded = st.file_uploader("Carica il file meteo CSV", type=["csv"])

with st.sidebar:
    st.header("Posizione")
    latitude = st.number_input("Latitudine [deg]", -90.0, 90.0, 37.178550, format="%.6f")
    longitude = st.number_input("Longitudine [deg]", -180.0, 180.0, 14.650785, format="%.6f")
    altitude = st.number_input("Altitudine [m]", value=308.0)
    timezone = st.text_input("Timezone IANA", "Etc/GMT-1")
    timestamp_offset = st.number_input("Offset timestamp [min]", value=30, step=1)

    st.header("Layout impianto")
    structure = st.radio("Configurazione", ["Tracker monoassiale", "Struttura fissa"])
    pitch = st.number_input("Pitch [m]", min_value=0.01, value=10.0)
    collector_width = st.number_input("Larghezza fila/collettore [m]", min_value=0.01, value=4.8)
    gcr_manual = st.checkbox("Inserisci GCR direttamente", value=False)
    gcr = st.number_input("GCR [-]", min_value=0.01, max_value=1.0, value=0.48, disabled=not gcr_manual)
    height = st.number_input("Altezza [m]", min_value=0.0, value=3.0)
    axis_azimuth = st.number_input("Axis azimuth / azimuth superficie [deg]", 0.0, 360.0, 180.0)
    if structure == "Struttura fissa":
        tilt = st.number_input("Tilt [deg]", 0.0, 90.0, 15.0)
        max_angle = None
        backtrack = False
        cross_axis_tilt = 0.0
    else:
        max_angle = st.number_input("Max angle [deg]", 0.0, 90.0, 55.0)
        backtrack = st.checkbox("Backtracking", value=True)
        cross_axis_tilt = st.number_input("Cross-axis tilt [deg]", -90.0, 90.0, 0.0)
        tilt = None

    st.header("Bifacciale e perdite")
    bifaciality = st.number_input("Bifaciality factor [-]", 0.0, 1.0, 0.80)
    albedo = st.number_input("Albedo [-]", 0.0, 1.0, 0.10)
    shade_factor = st.number_input("Shade factor [-]", -1.0, 0.0, -0.15)
    transmission_factor = st.number_input("Transmission factor [-]", -1.0, 0.0, -0.01)
    soiling = st.number_input("Soiling frontale [-]", 0.0, 1.0, 0.00)
    delta_poa = st.number_input("Correzione POA [-]", -1.0, 1.0, 0.04)
    iam_b = st.number_input("ASHRAE b [-]", min_value=0.0, value=0.05)
    iam_back = st.number_input("IAM posteriore [-]", min_value=0.0, max_value=1.0, value=0.98)

    st.header("Modulo e Todd-Karin")
    v1 = st.number_input("Voc STC V1 [V]", min_value=0.01, value=49.74)
    t1 = st.number_input("Temperatura di riferimento T1 [degC]", value=25.0)
    g1 = st.number_input("Irradianza di riferimento G1 [W/m2]", min_value=0.01, value=1000.0)
    beta = st.number_input("Beta Voc [1/degC]", value=-0.0026, format="%.6f")
    n_diode = st.number_input("Fattore di idealita n", min_value=0.01, value=1.38)
    vdc_max = st.number_input("Tensione massima sistema [V]", min_value=1.0, value=1500.0)


def read_input(file_bytes):
    text = file_bytes.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    required = {"Date", "Time", "GHI", "DNI", "DIF", "TEMP", "WS"}
    for line_no, line in enumerate(lines):
        for sep in (";", ","):
            cols = {x.strip().strip('"') for x in line.split(sep)}
            if required.issubset(cols):
                df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=line_no)
                df.columns = [str(c).strip() for c in df.columns]
                return df
    raise ValueError("Intestazione non trovata. Verificare nomi colonne e separatore.")


def calculate(df):
    required = ["Date", "Time", "GHI", "DNI", "DIF", "TEMP", "WS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Colonne mancanti: " + ", ".join(missing))
    out = df.copy()
    ts = pd.to_datetime(out["Date"].astype(str).str.strip() + " " + out["Time"].astype(str).str.strip(), format="%d.%m.%Y %H:%", errors="coerce")
    # fallback because some pandas versions reject the compact directive used above
    if ts.isna().any():
        ts = pd.to_datetime(out["Date"].astype(str).str.strip() + " " + out["Time"].astype(str).str.strip(), format="%d.%m.%Y %H:%M", errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} timestamp non validi. Usare GG.MM.AAAA e HH:MM.")
    ts = ts + pd.Timedelta(minutes=int(timestamp_offset))
    idx = pd.DatetimeIndex(ts).tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    out.index = idx
    for c in ["GHI", "DNI", "DIF", "TEMP", "WS"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[["GHI", "DNI", "DIF", "TEMP", "WS"]].isna().any().any():
        bad = out[["GHI", "DNI", "DIF", "TEMP", "WS"]].isna().sum()
        raise ValueError("Valori non numerici o mancanti: " + ", ".join(f"{k}={v}" for k, v in bad.items() if v))

    ghi, dni, dhi, temp_air, wind = out["GHI"], out["DNI"], out["DIF"], out["TEMP"], out["WS"]
    times_solar = out.index - pd.Timedelta(minutes=int(timestamp_offset))
    solpos = solarposition.get_solarposition(times_solar, latitude, longitude, altitude=altitude)
    solpos.index = out.index
    dni_extra = pvlib.irradiance.get_extra_radiation(out.index)
    actual_gcr = float(gcr if gcr_manual else collector_width / pitch)
    if not 0 < actual_gcr <= 1:
        raise ValueError(f"GCR calcolato pari a {actual_gcr:.3f}: deve essere compreso tra 0 e 1.")

    if structure == "Tracker monoassiale":
        orient = pvlib.tracking.singleaxis(
            apparent_zenith=solpos["apparent_zenith"], apparent_azimuth=solpos["azimuth"],
            axis_tilt=0.0, axis_azimuth=axis_azimuth, max_angle=max_angle,
            backtrack=backtrack, gcr=actual_gcr, cross_axis_tilt=cross_axis_tilt)
        surface_tilt = orient["surface_tilt"].fillna(0)
        surface_azimuth = orient["surface_azimuth"].fillna(axis_azimuth)
        aoi = orient["aoi"]
    else:
        surface_tilt = pd.Series(float(tilt), index=out.index)
        surface_azimuth = pd.Series(float(axis_azimuth), index=out.index)
        aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solpos["apparent_zenith"], solpos["azimuth"])

    iam_front = pvlib.iam.ashrae(aoi, b=iam_b).fillna(0)
    poa = pvlib.bifacial.infinite_sheds.get_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth, gcr=actual_gcr,
        height=height, pitch=pitch, dni_extra=dni_extra, albedo=albedo,
        dni=dni, ghi=ghi, dhi=dhi, bifaciality=bifaciality,
        iam_front=iam_front, iam_back=iam_back, shade_factor=shade_factor,
        transmission_factor=transmission_factor, npoints=200, vectorize=False,
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"], model="haydavies")
    poa_front = poa["poa_front"].fillna(0).clip(lower=0) * (1 - soiling)
    poa_back = poa["poa_back"].fillna(0).clip(lower=0)
    poa_global = ((poa_front + poa_back) * (1 + delta_poa)).clip(lower=0)

    t_faiman = pvlib.temperature.faiman(poa_global, temp_air, wind, u0=26.9, u1=6.2)
    t_pvsyst = pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind, u_c=25, u_v=1.2, module_efficiency=0.23, alpha_absorption=0.9)
    t_king = pvlib.temperature.sapm_cell(poa_global, temp_air, wind, a=-3.47, b=-0.0594, deltaT=3, irrad_ref=1000.0)
    t_cell = (t_faiman + t_pvsyst + t_king) / 3

    kb, q = 1.380649e-23, 1.602176634e-19
    a_todd = n_diode * kb * (t_cell + 273.15) / q
    voc = pd.Series(np.nan, index=out.index, dtype=float)
    day = poa_global > 0
    voc.loc[day] = v1 + v1 * beta * (t_cell.loc[day] - t1) + v1 * a_todd.loc[day] * np.log(poa_global.loc[day] / g1)

    result = pd.DataFrame({
        "GHI_Wm2": ghi, "DNI_Wm2": dni, "DIF_Wm2": dhi,
        "T_amb_C": temp_air, "WS_ms": wind,
        "POA_front_Wm2": poa_front, "POA_back_Wm2": poa_back,
        "POA_global_Wm2": poa_global,
        "T_cell_Faiman_C": t_faiman, "T_cell_PVsyst_C": t_pvsyst,
        "T_cell_King_C": t_king, "T_cell_media_C": t_cell,
        "Voc_corrected_V": voc
    })
    valid = result.dropna(subset=["Voc_corrected_V"])
    if valid.empty:
        raise ValueError("Nessuna ora con POA > 0: impossibile calcolare la Voc.")
    p995 = valid["Voc_corrected_V"].quantile(0.995)
    vmax = valid["Voc_corrected_V"].max()
    row995 = valid.iloc[(valid["Voc_corrected_V"] - p995).abs().argmin()]
    rowmax = valid.loc[valid["Voc_corrected_V"].idxmax()]
    filtered = valid[valid["POA_global_Wm2"] > 150]
    t_min_pvsyst = filtered["T_cell_media_C"].min() if not filtered.empty else np.nan
    a_pvsyst = n_diode * kb * (t_min_pvsyst + 273.15) / q if np.isfinite(t_min_pvsyst) else np.nan
    voc_pvsyst = v1 + v1 * beta * (t_min_pvsyst - t1) + v1 * a_pvsyst * np.log(1000 / g1) if np.isfinite(t_min_pvsyst) else np.nan
    summary = pd.DataFrame({
        "Voc [V]": [p995, vmax, voc_pvsyst],
        "Temp [degC]": [row995["T_cell_media_C"], rowmax["T_cell_media_C"], t_min_pvsyst],
        "POA [W/m2]": [row995["POA_global_Wm2"], rowmax["POA_global_Wm2"], 1000],
        "n_module": [vdc_max/p995, vdc_max/vmax, vdc_max/voc_pvsyst]
    }, index=["P:99.5", "P:100", "PVsyst"]).round(2)
    return result, summary, actual_gcr

if uploaded is None:
    st.info("Carica un CSV per avviare il calcolo.")
else:
    try:
        data = read_input(uploaded.getvalue())
        result, summary, actual_gcr = calculate(data)
        st.success(f"Calcolo completato su {len(result):,} righe. GCR utilizzato: {actual_gcr:.3f}".replace(",", "."))
        c1, c2, c3 = st.columns(3)
        c1.metric("Voc P99.5", f"{summary.loc['P:99.5','Voc [V]']:.2f} V")
        c2.metric("Voc massima", f"{summary.loc['P:100','Voc [V]']:.2f} V")
        c3.metric("Moduli/stringa a P99.5", f"{summary.loc['P:99.5','n_module']:.2f}")
        st.subheader("Risultati sintetici")
        st.dataframe(summary, use_container_width=True)
        st.subheader("Voc corretta ora per ora")
        chart = result.reset_index(names="timestamp")
        fig = px.line(chart, x="timestamp", y="Voc_corrected_V", labels={"timestamp":"Timestamp", "Voc_corrected_V":"Voc [V]"})
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Dati orari completi"):
            st.dataframe(result, use_container_width=True)
        csv_out = result.to_csv(index=True).encode("utf-8")
        st.download_button("Scarica risultati CSV", csv_out, "risultati_voc_orari.csv", "text/csv")
    except Exception as exc:
        st.error(f"Errore di elaborazione: {exc}")
