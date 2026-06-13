import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(
    page_title="SPALLO",
    page_icon="🎣",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Stile mobile-first
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        max-width: 760px;
    }
    div.stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 14px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.25rem;
    }
    .small-note {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎣 SPALLO")
st.caption("Disegna la spallinata: peso, pallini e distanze sulla lenza.")

# -----------------------------
# Dati standard pallini
# Nota: pesi indicativi, modificabili nella sezione avanzata.
# -----------------------------
DEFAULT_SHOTS = {
    "N.13": 0.010,
    "N.12": 0.015,
    "N.11": 0.020,
    "N.10": 0.030,
    "N.9": 0.050,
    "N.8": 0.070,
    "N.7": 0.090,
    "N.6": 0.120,
    "N.5": 0.160,
    "N.4": 0.200,
    "N.3": 0.250,
    "N.2": 0.300,
    "N.1": 0.400,
    "BB": 0.500,
    "AAA": 0.800,
}

PROFILE_PARAMS = {
    "Acqua ferma": {
        "n_factor": 18,
        "weight_power": 1.10,
        "spacing_power": 1.65,
        "description": "Morbida e aperta: pallini leggeri verso l'amo e distanze più ampie nella parte bassa.",
    },
    "Corrente lenta": {
        "n_factor": 15,
        "weight_power": 1.20,
        "spacing_power": 1.45,
        "description": "Progressiva e naturale, con buona morbidezza vicino all'amo.",
    },
    "Corrente media": {
        "n_factor": 12,
        "weight_power": 1.35,
        "spacing_power": 1.25,
        "description": "Equilibrata: più controllo senza irrigidire troppo la lenza.",
    },
    "Corrente veloce": {
        "n_factor": 9,
        "weight_power": 1.55,
        "spacing_power": 1.05,
        "description": "Più raccolta e portante: peso più concentrato verso il galleggiante.",
    },
}


def choose_number_of_shots(total_weight: float, available_weights: list[float], water_type: str, min_n: int = 3) -> int:
    """Stima un numero di pallini pratico, garantendo almeno una unità per ogni taglia selezionata."""
    if not available_weights:
        return 0
    median_weight = float(np.median(available_weights))
    base_n = int(round(total_weight / median_weight)) if median_weight > 0 else 8
    profile_n = PROFILE_PARAMS[water_type]["n_factor"]
    n = int(round((base_n * 0.55) + (profile_n * 0.45)))
    return max(min_n, min(50, n))


def allocate_counts(shots: dict[str, float], n: int, water_type: str) -> dict[str, int]:
    """
    Distribuisce il numero totale di pallini tra tutte le taglie selezionate.
    Ogni taglia selezionata viene sempre usata almeno una volta.

    Ordine logico: pallini più pesanti verso il galleggiante, più leggeri verso l'amo.
    La tipologia di acqua modifica la quantità relativa:
    - acqua ferma: più pallini piccoli e distribuzione morbida;
    - corrente lenta: quasi equa, leggermente più piccoli;
    - corrente media: equilibrata;
    - corrente veloce: più pallini grandi e portanti.
    """
    ordered = sorted(shots.items(), key=lambda kv: kv[1], reverse=True)
    k = len(ordered)
    n = max(n, k)

    counts = {tag: 1 for tag, _ in ordered}
    extra = n - k
    if extra == 0:
        return counts

    # Da 1 per i più pesanti a 0 per i più leggeri.
    rank_heavy_to_light = np.linspace(1, 0, k)

    if water_type == "Acqua ferma":
        # Favorisce i piccoli vicino all'amo.
        weights = 0.55 + (1 - rank_heavy_to_light) * 1.45
    elif water_type == "Corrente lenta":
        weights = 0.80 + (1 - rank_heavy_to_light) * 0.70
    elif water_type == "Corrente media":
        weights = np.ones(k)
    else:  # Corrente veloce
        # Favorisce i grandi verso il galleggiante.
        weights = 0.55 + rank_heavy_to_light * 1.45

    raw = weights / weights.sum() * extra
    floors = np.floor(raw).astype(int)
    remainders = raw - floors

    for (tag, _), add in zip(ordered, floors):
        counts[tag] += int(add)

    missing = extra - int(floors.sum())
    for idx in np.argsort(-remainders)[:missing]:
        tag = ordered[int(idx)][0]
        counts[tag] += 1

    return counts


def improve_counts_for_weight(counts: dict[str, int], shots: dict[str, float], target_weight: float, water_type: str, fixed_total: bool) -> dict[str, int]:
    """
    Avvicina il peso totale al target senza mai eliminare una taglia selezionata.
    Se fixed_total=True mantiene invariato il numero totale di pallini, spostando quantità tra taglie.
    Se fixed_total=False può aggiungere/togliere pallini, ma lascia almeno 1 per taglia.
    """
    ordered = sorted(shots.items(), key=lambda kv: kv[1], reverse=True)
    tags = [t for t, _ in ordered]
    weights = dict(ordered)
    current = counts.copy()

    def total(c):
        return sum(c[t] * weights[t] for t in tags)

    for _ in range(300):
        current_error = abs(total(current) - target_weight)
        best = None

        if fixed_total:
            # Prova a spostare 1 pallino da una taglia a un'altra, senza scendere sotto 1.
            for src in tags:
                if current[src] <= 1:
                    continue
                for dst in tags:
                    if src == dst:
                        continue
                    trial = current.copy()
                    trial[src] -= 1
                    trial[dst] += 1
                    err = abs(total(trial) - target_weight)
                    if err + 1e-12 < current_error:
                        best = trial
                        current_error = err
        else:
            # Prova aggiunte e rimozioni singole.
            for tag in tags:
                trial = current.copy()
                trial[tag] += 1
                err = abs(total(trial) - target_weight)
                if err + 1e-12 < current_error:
                    best = trial
                    current_error = err

                if current[tag] > 1:
                    trial = current.copy()
                    trial[tag] -= 1
                    err = abs(total(trial) - target_weight)
                    if err + 1e-12 < current_error:
                        best = trial
                        current_error = err

        if best is None:
            break
        current = best

    return current


def expand_counts_to_sequence(counts: dict[str, int], shots: dict[str, float]) -> list[tuple[str, float]]:
    """Espande i conteggi in sequenza dal galleggiante all'amo: peso decrescente."""
    ordered = sorted(shots.items(), key=lambda kv: kv[1], reverse=True)
    selected = []
    for tag, weight in ordered:
        selected.extend([(tag, weight)] * counts[tag])
    return selected


def build_weight_targets(total_weight: float, n: int, water_type: str) -> np.ndarray:
    """
    Funzione lasciata per eventuali versioni future.
    Crea target di peso dal galleggiante all'amo.
    """
    power = PROFILE_PARAMS[water_type]["weight_power"]
    x = np.linspace(1.0, 0.25, n) ** power
    weights = x / x.sum() * total_weight
    return weights


def nearest_available_shots(targets: np.ndarray, shots: dict[str, float]) -> list[tuple[str, float]]:
    """Funzione lasciata per eventuali versioni future."""
    items = sorted(shots.items(), key=lambda kv: kv[1], reverse=True)
    result = []
    for t in targets:
        tag, w = min(items, key=lambda kv: abs(kv[1] - t))
        result.append((tag, w))
    return result


def improve_total_weight(selected: list[tuple[str, float]], shots: dict[str, float], target_weight: float) -> list[tuple[str, float]]:
    """Funzione lasciata per eventuali versioni future."""
    if not selected:
        return selected
    return sorted(selected, key=lambda kv: kv[1], reverse=True)


def calculate_positions(length_cm: float, n: int, water_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Posizioni dal galleggiante verso l'amo.
    Distanze strette vicino al galleggiante, più larghe avvicinandosi all'amo.
    """
    if n <= 1:
        return np.array([0.0]), np.array([])

    power = PROFILE_PARAMS[water_type]["spacing_power"]
    raw = np.linspace(0.55, 1.85, n - 1) ** power
    distances = raw / raw.sum() * length_cm
    positions = np.concatenate([[0.0], np.cumsum(distances)])
    return positions, distances


def make_plan(length_cm: float, total_weight: float, water_type: str, shots: dict[str, float], manual_n: int | None):
    available_weights = list(shots.values())
    min_n = len(shots)
    fixed_total = manual_n is not None

    if fixed_total:
        n = max(manual_n, min_n)
    else:
        n = choose_number_of_shots(total_weight, available_weights, water_type, min_n=min_n)

    counts = allocate_counts(shots, n, water_type)
    counts = improve_counts_for_weight(counts, shots, total_weight, water_type, fixed_total=fixed_total)
    selected = expand_counts_to_sequence(counts, shots)

    positions, distances = calculate_positions(length_cm, len(selected), water_type)

    rows = []
    for i, ((tag, weight), pos) in enumerate(zip(selected, positions), start=1):
        distance_from_previous = 0.0 if i == 1 else pos - positions[i - 2]
        rows.append({
            "#": i,
            "Pallino": tag,
            "Peso g": round(weight, 3),
            "Da galleggiante cm": round(pos, 1),
            "Distanza cm": round(distance_from_previous, 1),
        })
    return pd.DataFrame(rows)


def plot_rig(df: pd.DataFrame, length_cm: float, total_weight: float, water_type: str):
    fig_height = max(7, min(13, 0.32 * len(df) + 5))
    fig, ax = plt.subplots(figsize=(4.8, fig_height))

    ax.vlines(0, 0, length_cm, linewidth=2)
    ax.text(0, -length_cm * 0.035, "Galleggiante", ha="center", va="top", fontsize=10)
    ax.text(0, length_cm * 1.035, "Amo", ha="center", va="bottom", fontsize=10)

    max_w = max(df["Peso g"].max(), 0.01)
    for _, row in df.iterrows():
        y = row["Da galleggiante cm"]
        size = 90 + (row["Peso g"] / max_w) * 420
        ax.scatter(0, y, s=size, zorder=3)
        ax.text(0.22, y, f"{row['Pallino']}  {row['Distanza cm']} cm", va="center", fontsize=9)

    ax.set_ylim(length_cm * 1.08, -length_cm * 0.08)
    ax.set_xlim(-0.6, 2.1)
    ax.set_xticks([])
    ax.set_ylabel("cm dal galleggiante")
    ax.set_title(f"{water_type} · target {total_weight:.2f} g", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    return fig


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8")


# -----------------------------
# UI
# -----------------------------
with st.container(border=True):
    st.subheader("1. Parametri")
    col1, col2 = st.columns(2)
    with col1:
        length_cm = st.number_input("Lunghezza cm", min_value=20, max_value=500, value=150, step=5)
    with col2:
        total_weight = st.number_input("Peso totale g", min_value=0.05, max_value=20.0, value=1.00, step=0.05, format="%.2f")

    water_type = st.selectbox(
        "Tipo di acqua",
        ["Acqua ferma", "Corrente lenta", "Corrente media", "Corrente veloce"],
        index=2,
    )
    st.markdown(f"<div class='small-note'>{PROFILE_PARAMS[water_type]['description']}</div>", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("2. Pallini disponibili")
    st.caption("Seleziona le misure che vuoi usare: SPALLO userà sempre almeno un pallino per ogni misura selezionata.")

    default_selected = {"N.11", "N.10", "N.9", "N.8", "N.7", "N.6"}
    selected_tags = []
    cols = st.columns(3)
    for idx, (tag, weight) in enumerate(DEFAULT_SHOTS.items()):
        with cols[idx % 3]:
            if st.checkbox(f"{tag} · {weight:.3f}g", value=tag in default_selected):
                selected_tags.append(tag)

    custom_mode = st.toggle("Modifica pesi pallini", value=False)
    shots = {}
    if custom_mode:
        st.caption("Puoi correggere i pesi se usi una tabella diversa del produttore.")
        for tag in selected_tags:
            shots[tag] = st.number_input(f"Peso {tag}", min_value=0.001, max_value=2.0, value=float(DEFAULT_SHOTS[tag]), step=0.001, format="%.3f")
    else:
        shots = {tag: DEFAULT_SHOTS[tag] for tag in selected_tags}

with st.expander("Opzioni avanzate"):
    use_manual_n = st.toggle("Imposta manualmente il numero di pallini", value=False)
    manual_n = None
    if use_manual_n:
        min_manual = max(3, len(selected_tags)) if selected_tags else 3
        manual_n = st.slider("Numero pallini", min_value=min_manual, max_value=50, value=max(14, min_manual))

calculate = st.button("Genera spallinata")

if calculate:
    if not shots:
        st.error("Seleziona almeno una misura di pallini.")
        st.stop()

    df = make_plan(length_cm, total_weight, water_type, shots, manual_n)
    actual_weight = df["Peso g"].sum()
    diff = actual_weight - total_weight

    st.success("Spallinata generata")

    c1, c2, c3 = st.columns(3)
    c1.metric("Target", f"{total_weight:.2f} g")
    c2.metric("Ottenuto", f"{actual_weight:.2f} g")
    c3.metric("Scarto", f"{diff:+.2f} g")

    st.subheader("Schema visuale")
    fig = plot_rig(df, length_cm, total_weight, water_type)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Distanze")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Riepilogo pallini")
    summary = (
        df.groupby("Pallino", as_index=False)
        .agg(Numero=("Pallino", "count"), Peso_totale_g=("Peso g", "sum"))
        .sort_values("Peso_totale_g", ascending=False)
    )
    summary["Peso_totale_g"] = summary["Peso_totale_g"].round(3)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.download_button(
        "Scarica CSV",
        data=dataframe_to_csv(df),
        file_name="spallo_spallinata.csv",
        mime="text/csv",
    )
else:
    st.info("Inserisci i parametri e genera la prima spallinata.")

st.markdown("---")
st.caption("SPALLO · versione prova v2. Usa sempre tutte le taglie selezionate; peso decrescente e distanze crescenti dal galleggiante verso l'amo.")
