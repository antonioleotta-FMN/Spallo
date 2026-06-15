import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from io import BytesIO

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
        padding-top: 0.8rem;
        padding-left: 0.7rem;
        padding-right: 0.7rem;
        max-width: 860px;
    }
    h1 {
        text-align: center;
        color: #06345f;
        letter-spacing: 0.04em;
    }
    div.stButton > button, div.stDownloadButton > button {
        width: 100%;
        min-height: 3rem;
        font-size: 1.02rem;
        font-weight: 700;
        border-radius: 14px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.18rem;
    }
    .small-note {
        font-size: 0.9rem;
        color: #5f6670;
        line-height: 1.35;
    }
    .spallo-card {
        border: 1px solid #d7e0ea;
        border-radius: 16px;
        padding: 0.75rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SPALLO 🎣")
st.caption("Disegna una spallinata verticale con pallini, distanze, galleggiante e amo.")

# -----------------------------
# Dati standard pallini
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

# Diametri indicativi solo per tabella/export grafico


def shot_sort_key(tag: str) -> float:
    """Ordina i pallini dal più pesante al più leggero: AAA, BB, N.1 ... N.13."""
    if tag == "AAA":
        return -2
    if tag == "BB":
        return -1
    if tag.startswith("N."):
        try:
            return int(tag.replace("N.", ""))
        except ValueError:
            return 999
    return 999


def shot_label(tag: str) -> str:
    """Etichetta più leggibile nello schema."""
    return tag.replace("N.", "N°")

SHOT_DIAMETERS = {
    "N.13": 1.00,
    "N.12": 1.10,
    "N.11": 1.30,
    "N.10": 1.60,
    "N.9": 1.90,
    "N.8": 2.20,
    "N.7": 2.50,
    "N.6": 2.80,
    "N.5": 3.10,
    "N.4": 3.40,
    "N.3": 3.70,
    "N.2": 4.00,
    "N.1": 4.40,
    "BB": 4.80,
    "AAA": 5.50,
}

PROFILE_PARAMS = {
    "Acqua ferma": {
        "n_factor": 18,
        "spacing_power": 1.75,
        "description": "Spallinata morbida: più pallini piccoli e distanze ampie verso l'amo.",
    },
    "Corrente lenta": {
        "n_factor": 15,
        "spacing_power": 1.50,
        "description": "Progressiva e naturale, con lenza morbida nella parte bassa.",
    },
    "Corrente media": {
        "n_factor": 12,
        "spacing_power": 1.28,
        "description": "Equilibrata: buon controllo e buona naturalezza dell'esca.",
    },
    "Corrente veloce": {
        "n_factor": 9,
        "spacing_power": 1.08,
        "description": "Più raccolta: pallini più portanti e distanze meno aperte.",
    },
}


# -----------------------------
# Algoritmo
# -----------------------------
def choose_number_of_shots(total_weight: float, available_weights: list[float], water_type: str, min_n: int = 3) -> int:
    if not available_weights:
        return 0
    median_weight = float(np.median(available_weights))
    base_n = int(round(total_weight / median_weight)) if median_weight > 0 else 8
    profile_n = PROFILE_PARAMS[water_type]["n_factor"]
    n = int(round((base_n * 0.55) + (profile_n * 0.45)))
    return max(min_n, min(50, n))


def allocate_counts(shots: dict[str, float], n: int, water_type: str) -> dict[str, int]:
    """Usa sempre tutte le taglie selezionate almeno una volta."""
    ordered = sorted(shots.items(), key=lambda kv: kv[1], reverse=True)
    k = len(ordered)
    n = max(n, k)
    counts = {tag: 1 for tag, _ in ordered}
    extra = n - k
    if extra == 0:
        return counts

    rank_heavy_to_light = np.linspace(1, 0, k)
    if water_type == "Acqua ferma":
        weights = 0.55 + (1 - rank_heavy_to_light) * 1.65
    elif water_type == "Corrente lenta":
        weights = 0.75 + (1 - rank_heavy_to_light) * 0.85
    elif water_type == "Corrente media":
        weights = np.ones(k)
    else:
        weights = 0.55 + rank_heavy_to_light * 1.65

    raw = weights / weights.sum() * extra
    floors = np.floor(raw).astype(int)
    remainders = raw - floors

    for (tag, _), add in zip(ordered, floors):
        counts[tag] += int(add)

    missing = extra - int(floors.sum())
    for idx in np.argsort(-remainders)[:missing]:
        counts[ordered[int(idx)][0]] += 1
    return counts


def improve_counts_for_weight(counts: dict[str, int], shots: dict[str, float], target_weight: float, fixed_total: bool) -> dict[str, int]:
    ordered = sorted(shots.items(), key=lambda kv: kv[1], reverse=True)
    tags = [t for t, _ in ordered]
    weights = dict(ordered)
    current = counts.copy()

    def total(c):
        return sum(c[t] * weights[t] for t in tags)

    for _ in range(350):
        current_error = abs(total(current) - target_weight)
        best = None
        if fixed_total:
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
    """Sequenza dal galleggiante all'amo: peso decrescente."""
    ordered = sorted(shots.items(), key=lambda kv: kv[1], reverse=True)
    selected = []
    for tag, weight in ordered:
        selected.extend([(tag, weight)] * counts[tag])
    return selected


def calculate_positions(length_cm: float, n: int, water_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Distanze strette vicino al galleggiante e larghe verso l'amo."""
    if n <= 1:
        return np.array([0.0]), np.array([])
    power = PROFILE_PARAMS[water_type]["spacing_power"]
    raw = np.linspace(0.55, 1.95, n - 1) ** power
    distances = raw / raw.sum() * length_cm
    positions = np.concatenate([[0.0], np.cumsum(distances)])
    return positions, distances


def make_plan(length_cm: float, total_weight: float, water_type: str, shots: dict[str, float], manual_n: int | None):
    min_n = len(shots)
    fixed_total = manual_n is not None
    if fixed_total:
        n = max(manual_n, min_n)
    else:
        n = choose_number_of_shots(total_weight, list(shots.values()), water_type, min_n=min_n)

    counts = allocate_counts(shots, n, water_type)
    counts = improve_counts_for_weight(counts, shots, total_weight, fixed_total=fixed_total)
    selected = expand_counts_to_sequence(counts, shots)
    positions, _ = calculate_positions(length_cm, len(selected), water_type)

    rows = []
    for i, ((tag, weight), pos) in enumerate(zip(selected, positions), start=1):
        distance_from_previous = np.nan if i == 1 else pos - positions[i - 2]
        rows.append({
            "#": i,
            "Pallino": tag,
            "Peso g": round(weight, 3),
            "Diametro mm": SHOT_DIAMETERS.get(tag, np.nan),
            "Da galleggiante cm": round(pos, 1),
            "Distanza dal precedente cm": None if i == 1 else round(distance_from_previous, 1),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Disegno vettoriale Matplotlib
# -----------------------------
def draw_float(ax, x: float, y: float, scale: float = 1.0):
    """Galleggiante stilizzato vettoriale."""
    ax.plot([x, x], [y - 8*scale, y + 7*scale], color="#111111", lw=1.3, zorder=5)
    ax.add_patch(Ellipse((x, y), 0.42*scale, 6.0*scale, facecolor="#0d65a6", edgecolor="#111111", lw=1.0, zorder=6))
    ax.add_patch(Ellipse((x, y - 1.5*scale), 0.36*scale, 2.8*scale, facecolor="#0b3d63", edgecolor="none", alpha=0.45, zorder=7))
    ax.add_patch(Polygon([[x - 0.22*scale, y - 3.0*scale], [x + 0.22*scale, y - 3.0*scale], [x, y - 4.2*scale]], facecolor="#ffd400", edgecolor="#111111", lw=0.8, zorder=8))
    ax.add_patch(Ellipse((x, y - 6.8*scale), 0.16*scale, 1.8*scale, facecolor="#ff1d1d", edgecolor="#cc0000", lw=0.7, zorder=8))


def draw_hook(ax, x: float, y: float, scale: float = 1.0):
    """Amo stilizzato vettoriale."""
    verts = [
        (x, y - 5.2*scale),
        (x, y - 2.3*scale),
        (x, y + 1.6*scale),
        (x - 0.55*scale, y + 3.2*scale),
        (x - 1.0*scale, y + 1.4*scale),
        (x - 0.72*scale, y + 0.2*scale),
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.LINETO]
    ax.add_patch(PathPatch(Path(verts, codes), lw=2.0, edgecolor="#202020", facecolor="none", capstyle="round", joinstyle="round", zorder=6))
    ax.add_patch(Ellipse((x, y - 5.6*scale), 0.22*scale, 0.7*scale, facecolor="white", edgecolor="#202020", lw=1.3, zorder=7))


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("Pallino", as_index=False)
        .agg(
            Diametro_mm=("Diametro mm", "first"),
            Peso_unitario_g=("Peso g", "first"),
            Quantita=("Pallino", "count"),
        )
    )
    # ordine per peso decrescente
    out = out.sort_values("Peso_unitario_g", ascending=False)
    return out


def draw_panel(ax, x, y, w, h, title, lines=None):
    rect = plt.Rectangle((x, y), w, h, facecolor="#ffffff", edgecolor="#0b3766", lw=1.1, zorder=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 3.2, title, ha="center", va="top", fontsize=10, fontweight="bold", color="#0b3766", zorder=2)
    if lines:
        yy = y + h - 8
        for label, value in lines:
            ax.text(x + 3, yy, label, ha="left", va="center", fontsize=8.6, color="#111111", zorder=2)
            ax.text(x + w - 3, yy, value, ha="right", va="center", fontsize=8.6, color="#111111", zorder=2)
            yy -= 5.2


def create_schema_figure(
    df: pd.DataFrame,
    length_cm: float,
    final_to_hook_cm: float,
    total_weight: float,
    water_type: str,
    float_capacity: float,
    float_loading: str,
    include_tables: bool = True,
):
    """Crea figura stile infografica, adatta a schermo mobile e download PNG."""
    actual_weight = float(df["Peso g"].sum())
    total_rig_cm = length_cm + final_to_hook_cm
    top_margin = 24
    bottom_margin = 18
    usable_h = 160
    y_top = top_margin
    y_last = top_margin + usable_h * (length_cm / total_rig_cm)
    y_hook = top_margin + usable_h

    if include_tables:
        fig_w = 8.5
        x_line = 2.1
        x_right = 4.15
        fig_h = 13.5
    else:
        fig_w = 4.2
        x_line = 2.15
        x_right = None
        fig_h = 13.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")
    ax.set_ylim(y_hook + bottom_margin, 0)
    ax.set_xlim(0, 8.5 if include_tables else 4.2)

    # Titolo
    ax.text(0.25, 7.0, "SPALLO", fontsize=19, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0b3766", edgecolor="#0b3766"))
    ax.text(0.25, 16, "SCHEMA SPALLINATA", fontsize=11.5, fontweight="bold", color="#111111")

    # Galleggiante e amo
    draw_float(ax, x_line, y_top - 9, scale=1.8)
    ax.text(x_line + 0.55, y_top - 9, "GALLEGGIANTE", ha="left", va="center", fontsize=9.2, fontweight="bold")
    ax.text(x_line + 0.55, y_top - 4.5, f"Portata: {float_capacity:g} g", ha="left", va="center", fontsize=8.2)
    ax.text(x_line + 0.55, y_top - 0.4, f"Piombatura: {float_loading}", ha="left", va="center", fontsize=8.2)

    # Lenza principale e finale
    ax.plot([x_line, x_line], [y_top, y_last], color="#111111", lw=1.2, zorder=1)
    ax.plot([x_line, x_line], [y_last, y_hook - 7], color="#111111", lw=0.9, ls=(0, (2, 2)), zorder=1)
    draw_hook(ax, x_line, y_hook - 3, scale=1.45)
    ax.text(x_line - 0.62, y_hook - 4, "AMO", ha="right", va="center", fontsize=10, fontweight="bold")

    # Posizioni proporzionali
    max_weight = max(df["Peso g"].max(), 0.001)
    scaled_positions = y_top + usable_h * (df["Da galleggiante cm"].to_numpy() / total_rig_cm)

    # Etichette: le teniamo in ordine e con distanza minima, così su cellulare non si sovrappongono.
    label_positions = scaled_positions.copy()
    min_label_gap = 4.2 if len(df) <= 22 else 3.2
    for i in range(1, len(label_positions)):
        if label_positions[i] - label_positions[i - 1] < min_label_gap:
            label_positions[i] = label_positions[i - 1] + min_label_gap
    overflow = label_positions[-1] - (y_last + 2) if len(label_positions) else 0
    if overflow > 0:
        label_positions = label_positions - overflow
        for i in range(len(label_positions) - 2, -1, -1):
            if label_positions[i + 1] - label_positions[i] < min_label_gap:
                label_positions[i] = label_positions[i + 1] - min_label_gap

    for idx, (_, row) in enumerate(df.iterrows()):
        y = scaled_positions[idx]
        y_label = label_positions[idx]
        radius = 0.065 + 0.115 * (row["Peso g"] / max_weight)
        ax.add_patch(Ellipse((x_line, y), radius * 2.0, radius * 2.0, facecolor="#333333", edgecolor="#111111", lw=0.7, zorder=4))
        ax.add_patch(Ellipse((x_line - radius*0.25, y - radius*0.25), radius*0.65, radius*0.65, facecolor="#9ca3aa", edgecolor="none", alpha=0.55, zorder=5))
        ax.text(x_line - 0.55, y_label, shot_label(row["Pallino"]), ha="right", va="center", fontsize=8.8)
        if abs(y_label - y) > 0.3:
            ax.plot([x_line - 0.48, x_line - 0.13], [y_label, y], color="#b5c1cc", lw=0.45, zorder=1)
        ax.plot([x_line + 0.12, x_line + 0.62], [y, y], color="#9eafbf", lw=0.7, ls=(0, (3, 3)), zorder=1)

        if idx > 0:
            y_prev = scaled_positions[idx - 1]
            d = row["Distanza dal precedente cm"]
            x_arrow = x_line + 0.72
            ax.add_patch(FancyArrowPatch((x_arrow, y_prev), (x_arrow, y), arrowstyle="<|-|>", mutation_scale=8,
                                         lw=0.85, color="#0b5da8", zorder=3))
            ax.text(x_arrow + 0.16, (y_prev + y) / 2, f"{d:.0f} cm", ha="left", va="center", fontsize=8.8, color="#0b3766")

    # Distanza ultimo pallino - amo
    if final_to_hook_cm > 0:
        x_arrow = x_line + 0.72
        ax.plot([x_line + 0.12, x_line + 0.62], [y_hook - 7, y_hook - 7], color="#c4ced8", lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.add_patch(FancyArrowPatch((x_arrow, y_last), (x_arrow, y_hook - 7), arrowstyle="<|-|>", mutation_scale=8,
                                     lw=0.85, color="#0b5da8", zorder=3))
        ax.text(x_arrow + 0.16, (y_last + y_hook - 7)/2, f"{final_to_hook_cm:.0f} cm", ha="left", va="center", fontsize=8.8, color="#0b3766")

    # Pannelli laterali
    if include_tables:
        summary = summary_table(df)
        panel_x = x_right
        panel_w = 3.95
        y0 = 24
        row_h = 5.6
        h = 11 + row_h * (len(summary) + 1) + 8
        ax.add_patch(plt.Rectangle((panel_x, y0), panel_w, h, facecolor="#ffffff", edgecolor="#0b3766", lw=1.1))
        ax.text(panel_x + panel_w/2, y0 + 5, "RIEPILOGO PALLINI", ha="center", va="center", fontsize=10, fontweight="bold", color="#0b3766")
        cols = ["N°", "Ø mm", "Peso g", "Q.tà"]
        col_x = [panel_x + 0.45, panel_x + 1.45, panel_x + 2.55, panel_x + 3.5]
        yy = y0 + 11
        for cx, lab in zip(col_x, cols):
            ax.text(cx, yy, lab, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.plot([panel_x, panel_x + panel_w], [yy + 3, yy + 3], color="#bac7d4", lw=0.6)
        yy += row_h
        for _, r in summary.iterrows():
            num = r["Pallino"].replace("N.", "")
            values = [num, f"{r['Diametro_mm']:.2f}", f"{r['Peso_unitario_g']:.3f}", f"{int(r['Quantita'])}"]
            for cx, val in zip(col_x, values):
                ax.text(cx, yy, val, ha="center", va="center", fontsize=8.2)
            ax.plot([panel_x, panel_x + panel_w], [yy + 3, yy + 3], color="#e2e8ef", lw=0.5)
            yy += row_h
        ax.add_patch(plt.Rectangle((panel_x, y0 + h - 8), panel_w, 8, facecolor="#eef6ff", edgecolor="#0b3766", lw=0.6))
        ax.text(panel_x + 0.65, y0 + h - 4, "PESO TOTALE", ha="left", va="center", fontsize=9, fontweight="bold", color="#0b3766")
        ax.text(panel_x + panel_w - 0.35, y0 + h - 4, f"{actual_weight:.3f} g", ha="right", va="center", fontsize=10, fontweight="bold", color="#0b3766")

        draw_panel(ax, panel_x, y0 + h + 7, panel_w, 35, "PARAMETRI", [
            ("Lunghezza spallinata", f"{length_cm:.0f} cm"),
            ("Tratto finale amo", f"{final_to_hook_cm:.0f} cm"),
            ("Peso da distribuire", f"{total_weight:.2f} g"),
            ("Tipo di acqua", water_type),
            ("Totale pallini", f"{len(df)}"),
        ])
        draw_panel(ax, panel_x, y0 + h + 48, panel_w, 29, "LEGENDA", [
            ("Linea continua", "lenza madre"),
            ("Linea tratteggiata", "tratto finale"),
            ("Frecce blu", "distanze"),
        ])
        ax.add_patch(plt.Rectangle((panel_x, y0 + h + 84), panel_w, 24, facecolor="#eef6ff", edgecolor="#0b3766", lw=1.0))
        ax.text(panel_x + 0.42, y0 + h + 91, "CONSIGLIO", ha="left", va="center", fontsize=9, fontweight="bold", color="#0b3766")
        ax.text(panel_x + 0.42, y0 + h + 99, "Le distanze aumentano verso l'amo\nper rendere la lenza più morbida.", ha="left", va="center", fontsize=8.2, color="#111111")

    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8")


# -----------------------------
# UI
# -----------------------------
with st.container(border=True):
    st.subheader("1. Parametri")
    col1, col2 = st.columns(2)
    with col1:
        length_cm = st.number_input("Lunghezza spallinata cm", min_value=20, max_value=500, value=150, step=5)
    with col2:
        total_weight = st.number_input("Peso da distribuire g", min_value=0.05, max_value=20.0, value=1.00, step=0.05, format="%.2f")

    col3, col4 = st.columns(2)
    with col3:
        final_to_hook_cm = st.number_input("Distanza ultimo pallino-amo cm", min_value=0, max_value=200, value=30, step=5)
    with col4:
        float_capacity = st.number_input("Portata galleggiante g", min_value=0.1, max_value=30.0, value=4.0, step=0.5, format="%.1f")

    float_loading = st.text_input("Piombatura galleggiante", value="3+1")

    water_type = st.selectbox(
        "Tipo di acqua",
        ["Acqua ferma", "Corrente lenta", "Corrente media", "Corrente veloce"],
        index=2,
    )
    st.markdown(f"<div class='small-note'>{PROFILE_PARAMS[water_type]['description']}</div>", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("2. Pallini disponibili")
    st.caption("SPALLO usa sempre almeno un pallino per ogni misura selezionata.")

    default_selected = {"N.11", "N.10", "N.9", "N.8", "N.7", "N.6"}

    # Su cellulare le colonne di checkbox diventano confuse: uso un multiselect ordinato.
    # Ordine visuale: pallini più pesanti in alto, più leggeri in basso.
    ordered_tags = sorted(DEFAULT_SHOTS.keys(), key=shot_sort_key)
    selected_tags = st.multiselect(
        "Misure da usare",
        options=ordered_tags,
        default=[tag for tag in ordered_tags if tag in default_selected],
        format_func=lambda tag: f"{shot_label(tag)} · {DEFAULT_SHOTS[tag]:.3f} g",
        help="SPALLO userà sempre almeno un pallino per ogni misura selezionata.",
    )
    selected_tags = sorted(selected_tags, key=shot_sort_key)

    custom_mode = st.toggle("Modifica pesi pallini", value=False)
    shots = {}
    if custom_mode:
        st.caption("Correggi i pesi se usi una tabella diversa del produttore.")
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
    include_tables = st.toggle("Includi pannelli riepilogo nel disegno", value=True)

calculate = st.button("Genera schema")

if calculate:
    if not shots:
        st.error("Seleziona almeno una misura di pallini.")
        st.stop()

    df = make_plan(length_cm, total_weight, water_type, shots, manual_n)
    actual_weight = df["Peso g"].sum()
    diff = actual_weight - total_weight

    st.success("Schema generato")
    c1, c2, c3 = st.columns(3)
    c1.metric("Target", f"{total_weight:.2f} g")
    c2.metric("Ottenuto", f"{actual_weight:.2f} g")
    c3.metric("Scarto", f"{diff:+.2f} g")

    tab_schema, tab_table = st.tabs(["Schema", "Tabelle"])

    with tab_schema:
        fig = create_schema_figure(
            df=df,
            length_cm=length_cm,
            final_to_hook_cm=final_to_hook_cm,
            total_weight=total_weight,
            water_type=water_type,
            float_capacity=float_capacity,
            float_loading=float_loading,
            include_tables=include_tables,
        )
        st.pyplot(fig, use_container_width=True)
        png_bytes = fig_to_png_bytes(fig)
        st.download_button(
            "Scarica schema PNG",
            data=png_bytes,
            file_name="spallo_schema_spallinata.png",
            mime="image/png",
        )

    with tab_table:
        st.subheader("Distanze")
        display_df = df.copy()
        display_df["Distanza dal precedente cm"] = display_df["Distanza dal precedente cm"].fillna("-")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Riepilogo pallini")
        summary = summary_table(df)
        summary = summary.rename(columns={
            "Diametro_mm": "Diametro mm",
            "Peso_unitario_g": "Peso unitario g",
            "Quantita": "Quantità",
        })
        summary["Diametro mm"] = summary["Diametro mm"].round(2)
        summary["Peso unitario g"] = summary["Peso unitario g"].round(3)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.download_button(
            "Scarica CSV distanze",
            data=dataframe_to_csv(df),
            file_name="spallo_spallinata.csv",
            mime="text/csv",
        )
else:
    st.info("Inserisci i parametri e genera lo schema.")

st.markdown("---")
st.caption("SPALLO · v4 prova. Selezione pallini ordinata per cellulare e schema con etichette anti-sovrapposizione.")
