# PV Voc Tool Streamlit

Applicazione Streamlit per caricare dati meteo Solargis, calcolare POA bifacciale, temperatura di cella e Voc Todd-Karin.

## Formati supportati

### Solargis TShourly

```text
Date;Time;GHI;DNI;DIF;...;TEMP;...;WS;...
01.01.1994;00:30;0;0;0;...;3.9;...;1.5;...
```

### Solargis TMY P50

```text
Day;Time;GHI;DNI;DIF;...;TEMP;...;WS;...
1;00:30;0;0;0;...;5.9;...;4.3;...
```

Il parser rileva automaticamente:

- separatore `;`, `,` o tabulazione;
- riga di intestazione dopo i metadati;
- schema `Date + Time` oppure `Day + Time`;
- riferimento temporale Solargis `UTC+0`, `UTC+1`, ecc.;
- valori mancanti Solargis codificati come `-9`.

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Render

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

Il file `render.yaml` permette anche il deploy come Blueprint.
