import io
import re
from datetime import timezone as dt_timezone, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pvlib
from pvlib import solarposition

st.set_page_config(page_title="POA, temperatura cella e Voc Todd-Karin", layout="wide")
st.title("Calcolo POA, temperatura di cella e Voc Todd-Karin")
st.caption("Implementazione limitata alla sezione 'Corrected Voc: tutte le temperature'.")

REQUIRED_METEO = ["GHI", "DNI", "DIF", "TEMP", "WS"]


def _detect_separator_and_header(text):
    """Individua separatore e riga di intestazione senza assumere un numero fisso di metadati."""
    accepted_time_columns = {"Date", "Day"}
    for line_no, line in enumerate(text.splitlines()):
        for sep in (";", ",", "\t"):
            columns = [x.strip().strip('"') for x in line.split(sep)]
            names = set(columns)
            if "Time" in names and accepted_time_columns.intersection(names) and set(REQUIRED_METEO).issubset(names):
                return sep, line_no, columns
    raise ValueError(
        "Intestazione non riconosciuta. Il file deve contenere Time, GHI, DNI, DIF, TEMP, WS "
        "e una colonna temporale Date oppure Day."
    )


def _read_utc_offset(text):
    """Legge, se presente, una dicitura Solargis come 'time reference UTC+1'."""
    match = re.search(r"time\s+reference\s+UTC\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", text, re.IGNORECASE)
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3) or 0)
    return sign * (hours + minutes / 60)


def _read_coordinate(text, label):
    match = re.search(rf"#{label}:\s*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def read_input(file_bytes, filename="file.csv"):
    """Normalizza file Solargis TShourly (Date) e TMY P50 (Day)."""
    text = file_bytes.decode("utf-8-sig", errors="replace")
    sep, header_line, columns = _detect_separator_and_header(text)
    df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=header_line, comment="#", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "Date" in df.columns:
        data_format = "Solargis TShourly / serie storica (Date + Time)"
    elif "Day" in df.columns:
        data_format = "Solargis TMY P50 (Day-of-year + Time)"
    else:
        raise ValueError("Manca la colonna Date o Day.")

    missing = [c for c in REQUIRED_METEO + ["Time"] if c not in df.columns]
    if missing:
        raise ValueError("Colonne obbligatorie mancanti: " + ", ".join(missing))

    metadata = {
        "filename": filename,
        "format": data_format,
        "separator": "TAB" if sep == "\t" else sep,
        "header_line": header_line + 1,
        "utc_offset": _read_utc_offset(text),
        "latitude": _read_coordinate(text, "Latitude"),
        "longitude": _read_coordinate(text, "Longitude"),
        "elevation": _read_coordinate(text, "Elevation"),
    }
    return df, metadata


def build_timestamp(df, metadata, manual_timezone, timestamp_offset, tmy_reference_year):
    """Crea l'indice temporale per entrambi gli schemi supportati."""
    time_text = df["Time"].astype(str).str.strip()

    if "Date" in df.columns:
        timestamp = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + time_text,
            format="%d.%m.%Y %H:%M",
            errors="coerce",
        )
    else:
        day = pd.to_numeric(df["Day"], errors="coerce")
        hour_parts = time_text.str.extract(r"^(\d{1,2}):(\d{2})$")
        hour = pd.to_numeric(hour_parts[0], errors="coerce")
        minute = pd.to_numeric(hour_parts[1], errors="coerce")
        base = pd.Timestamp(year=int(tmy_reference_year), month=1, day=1)
        timestamp = base + pd.to_timedelta(day - 1, unit="D") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")

    invalid = int(pd.isna(timestamp).sum())
    if invalid:
        raise ValueError(f"{invalid} timestamp non validi. Controllare Date/Day e Time.")

    timestamp = timestamp + pd.Timedelta(minutes=int(timestamp_offset))

    # Auto usa l'offset dichiarato nei metadati Solargis; in assenza usa la timezone inserita.
    if manual_timezone.strip().lower() == "auto":
        utc_offset = metadata.get("utc_offset")
        tz = dt_timezone(timedelta(hours=utc_offset)) if utc_offset is not None else "UTC"
    else:
        tz = manual_timezone.strip()

    try:
        index = pd.DatetimeIndex(timestamp).tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception as exc:
        raise ValueError(f"Timezone non valida ({tz}): {exc}") from exc
    return index, str(tz)


with st.expander("Formati CSV supportati", expanded=True):
    st.markdown("""
L'app riconosce automaticamente l'intestazione anche dopo righe di metadati `#` e accetta separatore `;`, `,` o tabulazione.

**Formato 1, serie storica Solargis TShourly**
```text
Date;Time;GHI;DNI;DIF;flagR;SE;SA;TEMP;AP;RH;WS;WG;WD;PREC;PWAT
01.01.1994;00:30;0;0;0;0;-66.67;-123.34;3.9;974.5;96;1.5;4.3;342;0;7.8
```

**Formato 2, Solargis TMY P50**
```text
Day;Time;GHI;DNI;DIF;SE;SA;TEMP;AP;RH;WS;WG;WD;PREC;PWAT
1;00:30;0;0;0;-74.19;-159.1;5.9;971.7;68.3;4.3;11.3;338;0.1;8.9
```

Colonne minime obbligatorie: `Time`, `GHI`, `DNI`, `DIF`, `TEMP`, `WS` e una tra `Date` e `Day`.
Per il TMY, `Day` viene convertito in una data usando un anno di riferimento non bisestile configurabile. Questo conserva correttamente il giorno dell'anno usato nei calcoli solari.
    """)
    example = (
        "Date;Time;GHI;DNI;DIF;TEMP;WS\n"
        "01.01.2025;00:00;0;0;0;7.2;2.1\n"
        "01.01.2025;12:00;520;710;95;14.8;3.4\n"
    )
    st.download_button("Scarica CSV minimo di esempio", example.encode("utf-8"), "formato_input.csv", "text/csv")

uploaded = st.file_uploader("Carica il file meteo CSV", type=["csv", "txt"])

with st.sidebar:
    st.header("Posizione e tempo")
    latitude = st.number_input("Latitudine [deg]", -90.0, 90.0, 37.178550, format="%.6f")
    longitude = st.number_input("Longitudine [deg]", -180.0, 180.0, 14.650785, format="%.6f")
    altitude = st.number_input("Altitudine [m]", value=308.0)
    timezone_input = st.text_input("Timezone", "auto", help="Usa 'auto' per leggere UTC+0/UTC+1 dai metadati Solargis; altrimenti inserisci una timezone IANA, es. Europe/Rome.")
    timestamp_offset = st.number_input("Offset aggiuntivo timestamp [min]", value=0, step=1, help="I file di esempio sono già centrati all'intervallo (:30), quindi il valore consigliato è 0.")
    tmy_reference_year = st.number_input("Anno di riferimento TMY", min_value=1901, max_value=2099, value=2001, step=1)

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
        max_angle, backtrack, cross_axis_tilt = None, False, 0.0
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


def calculate(df, metadata):
    index, tz_used = build_timestamp(df, metadata, timezone_input, timestamp_offset, tmy_reference_year)
    out = df.copy()
    out.index = index

    for column in REQUIRED_METEO:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace(-9, np.nan)

    invalid_rows = out[REQUIRED_METEO].isna().any(axis=1)
    dropped_rows = int(invalid_rows.sum())
    if dropped_rows:
        out = out.loc[~invalid_rows].copy()
    if out.empty:
        raise ValueError("Nessuna riga valida dopo la gestione dei valori mancanti (-9 o non numerici).")

    ghi, dni, dhi = out["GHI"].clip(lower=0), out["DNI"].clip(lower=0), out["DIF"].clip(lower=0)
    temp_air, wind = out["TEMP"], out["WS"].clip(lower=0)
    solpos = solarposition.get_solarposition(out.index, latitude, longitude, altitude=altitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(out.index)
    actual_gcr = float(gcr if gcr_manual else collector_width / pitch)
    if not 0 < actual_gcr <= 1:
        raise ValueError(f"GCR calcolato pari a {actual_gcr:.3f}; deve essere compreso tra 0 e 1.")

    if structure == "Tracker monoassiale":
        orient = pvlib.tracking.singleaxis(
            apparent_zenith=solpos["apparent_zenith"], apparent_azimuth=solpos["azimuth"],
            axis_tilt=0.0, axis_azimuth=axis_azimuth, max_angle=max_angle,
            backtrack=backtrack, gcr=actual_gcr, cross_axis_tilt=cross_axis_tilt,
        )
        surface_tilt = orient["surface_tilt"].fillna(0)
        surface_azimuth = orient["surface_azimuth"].fillna(axis_azimuth)
        aoi = orient["aoi"]
    else:
        surface_tilt = pd.Series(float(tilt), index=out.index)
        surface_azimuth = pd.Series(float(axis_azimuth), index=out.index)
        aoi = pd.Series(
            pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solpos["apparent_zenith"], solpos["azimuth"]),
            index=out.index,
        )

    iam_front = pd.Series(pvlib.iam.ashrae(aoi, b=iam_b), index=out.index).fillna(0)
    poa = pvlib.bifacial.infinite_sheds.get_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        gcr=actual_gcr, height=height, pitch=pitch, dni_extra=dni_extra,
        albedo=albedo, dni=dni, ghi=ghi, dhi=dhi, bifaciality=bifaciality,
        iam_front=iam_front, iam_back=iam_back, shade_factor=shade_factor,
        transmission_factor=transmission_factor, npoints=200, vectorize=False,
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"], model="haydavies",
    )
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
    daylight = poa_global > 0
    voc.loc[daylight] = v1 + v1 * beta * (t_cell.loc[daylight] - t1) + v1 * a_todd.loc[daylight] * np.log(poa_global.loc[daylight] / g1)

    result = pd.DataFrame({
        "GHI_Wm2": ghi, "DNI_Wm2": dni, "DIF_Wm2": dhi,
        "T_amb_C": temp_air, "WS_ms": wind,
        "POA_front_Wm2": poa_front, "POA_back_Wm2": poa_back,
        "POA_global_Wm2": poa_global,
        "T_cell_Faiman_C": t_faiman, "T_cell_PVsyst_C": t_pvsyst,
        "T_cell_King_C": t_king, "T_cell_media_C": t_cell,
        "Voc_corrected_V": voc,
    })
    valid = result.dropna(subset=["Voc_corrected_V"])
    if valid.empty:
        raise ValueError("Nessuna ora con POA > 0: impossibile calcolare la Voc.")

    p995 = valid["Voc_corrected_V"].quantile(0.995)
    vmax = valid["Voc_corrected_V"].max()
    row995 = valid.iloc[np.abs(valid["Voc_corrected_V"].to_numpy() - p995).argmin()]
    rowmax = valid.loc[valid["Voc_corrected_V"].idxmax()]
    filtered = valid[valid["POA_global_Wm2"] > 150]
    t_min_pvsyst = filtered["T_cell_media_C"].min() if not filtered.empty else np.nan
    a_pvsyst = n_diode * kb * (t_min_pvsyst + 273.15) / q if np.isfinite(t_min_pvsyst) else np.nan
    voc_pvsyst = v1 + v1 * beta * (t_min_pvsyst - t1) + v1 * a_pvsyst * np.log(1000 / g1) if np.isfinite(t_min_pvsyst) else np.nan

    summary = pd.DataFrame({
        "Voc [V]": [p995, vmax, voc_pvsyst],
        "Temp [degC]": [row995["T_cell_media_C"], rowmax["T_cell_media_C"], t_min_pvsyst],
        "POA [W/m2]": [row995["POA_global_Wm2"], rowmax["POA_global_Wm2"], 1000],
        "n_module": [vdc_max / p995, vdc_max / vmax, vdc_max / voc_pvsyst],
    }, index=["P:99.5", "P:100", "PVsyst"]).round(2)
    return result, summary, actual_gcr, dropped_rows, tz_used


if uploaded is None:
    st.info("Carica un CSV per avviare il calcolo.")
else:
    try:
        data, metadata = read_input(uploaded.getvalue(), uploaded.name)
        st.success(f"Formato riconosciuto: {metadata['format']}")
        info_cols = st.columns(4)
        info_cols[0].metric("Righe lette", f"{len(data):,}".replace(",", "."))
        info_cols[1].metric("Separatore", metadata["separator"])
        info_cols[2].metric("Intestazione", f"riga {metadata['header_line']}")
        utc_label = f"UTC{metadata['utc_offset']:+g}" if metadata["utc_offset"] is not None else "non rilevato"
        info_cols[3].metric("Riferimento tempo", utc_label)
        if metadata["latitude"] is not None and metadata["longitude"] is not None:
            st.caption(f"Metadati file: latitudine {metadata['latitude']}, longitudine {metadata['longitude']}, elevazione {metadata['elevation']} m. I valori di calcolo restano quelli impostati nella barra laterale.")
        with st.expander("Anteprima dati normalizzati"):
            st.dataframe(data.head(24), use_container_width=True)

        result, summary, actual_gcr, dropped_rows, tz_used = calculate(data, metadata)
        message = f"Calcolo completato su {len(result):,} righe. GCR: {actual_gcr:.3f}. Timezone usata: {tz_used}.".replace(",", ".")
        st.success(message)
        if dropped_rows:
            st.warning(f"Escluse {dropped_rows} righe con valori mancanti, non numerici o codificati come -9 nelle colonne obbligatorie.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Voc P99.5", f"{summary.loc['P:99.5', 'Voc [V]']:.2f} V")
        c2.metric("Voc massima", f"{summary.loc['P:100', 'Voc [V]']:.2f} V")
        c3.metric("Moduli/stringa a P99.5", f"{summary.loc['P:99.5', 'n_module']:.2f}")
        st.subheader("Risultati sintetici")
        st.dataframe(summary, use_container_width=True)
        st.subheader("Voc corretta ora per ora")
        chart = result.reset_index(names="timestamp")
        fig = px.line(chart, x="timestamp", y="Voc_corrected_V", labels={"timestamp": "Timestamp", "Voc_corrected_V": "Voc [V]"})
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Dati orari completi"):
            st.dataframe(result, use_container_width=True)
        st.download_button("Scarica risultati CSV", result.to_csv(index=True).encode("utf-8"), "risultati_voc_orari.csv", "text/csv")
    except Exception as exc:
        st.error(f"Errore di elaborazione: {exc}")
