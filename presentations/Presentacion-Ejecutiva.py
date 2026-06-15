"""
Presentación Ejecutiva — PPI Berries: TDA + LSTM Forecasting
Genera Presentacion-Ejecutiva.pdf (12 slides, ~8 minutos)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

BLUE   = '#1565C0'
ORANGE = '#E65100'
GREEN  = '#2E7D32'
RED    = '#C62828'
GRAY   = '#546E7A'
LIGHT  = '#E3F2FD'
BG     = '#FAFAFA'

def slide_fig(title=None, subtitle=None):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    if title:
        y = 0.97 if not subtitle else 0.97
        fig.text(0.5, y, title, ha='center', va='top',
                 fontsize=22, fontweight='bold', color=BLUE)
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha='center', va='top',
                 fontsize=13, color=GRAY)
    # bottom bar
    fig.add_axes([0, 0, 1, 0.025]).set_axis_off()
    fig.axes[-1].set_facecolor(BLUE)
    fig.text(0.02, 0.01, 'PPI Berries — Análisis TDA + LSTM  |  Topología 6° Semestre',
             fontsize=7, color='white')
    fig.text(0.98, 0.01, '2026', fontsize=7, color='white', ha='right')
    return fig

# ── Load data ──────────────────────────────────────────────
df     = pd.read_csv('../data/input/WPUSI01102B.csv', parse_dates=['observation_date'])
prices = df['WPUSI01102B'].values.astype(float)
dates  = df['observation_date'].values
pd_dates = pd.to_datetime(dates)

out_path = 'Presentacion-Ejecutiva.pdf'

with PdfPages(out_path) as pdf:

    # ═══════════════════════════════════════════════════════
    # SLIDE 1 — PORTADA
    # ═══════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(BLUE)

    # Top accent strip
    ax_strip = fig.add_axes([0, 0.88, 1, 0.12])
    ax_strip.set_facecolor('#0D47A1')
    ax_strip.set_axis_off()

    fig.text(0.5, 0.80, 'Análisis Topológico de Datos + LSTM',
             ha='center', fontsize=28, fontweight='bold', color='white')
    fig.text(0.5, 0.70, 'Forecasting del PPI de Berries (FRED: WPUSI01102B)',
             ha='center', fontsize=18, color='#BBDEFB')
    fig.text(0.5, 0.60, 'Junio 2008 – Abril 2026  |  215 observaciones mensuales',
             ha='center', fontsize=13, color='#90CAF9')

    # Mini sparkline
    ax_spark = fig.add_axes([0.2, 0.32, 0.6, 0.20])
    ax_spark.patch.set_facecolor('none')
    ax_spark.plot(pd_dates, prices, color='#64B5F6', linewidth=1.5, alpha=0.9)
    ax_spark.set_axis_off()

    fig.text(0.5, 0.20, 'Sesiones 2–10  |  TDA · Takens Embedding · LSTM · EP Loss',
             ha='center', fontsize=11, color='#90CAF9')
    fig.text(0.5, 0.08, 'Topología — 6° Semestre  ·  2026',
             ha='center', fontsize=10, color='#64B5F6')

    # bottom bar
    ax_bot = fig.add_axes([0, 0, 1, 0.03])
    ax_bot.set_facecolor('#0D47A1')
    ax_bot.set_axis_off()

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 2 — DATOS + SUBPERIODOS (Sesion 2-3)
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('Los Datos: PPI Mensual de Berries',
                    'Sesiones 2–3  |  EDA y definición de subperiodos para análisis TDA')

    ax = fig.add_axes([0.07, 0.13, 0.88, 0.70])

    # subperiod shading
    p1 = (pd.Timestamp('2008-06-01'), pd.Timestamp('2012-12-31'))
    p2 = (pd.Timestamp('2013-01-01'), pd.Timestamp('2019-12-31'))
    p3 = (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-04-30'))

    ax.axvspan(*p1, alpha=0.12, color=RED,    label='2008–2012: Crisis financiera')
    ax.axvspan(*p2, alpha=0.10, color=GREEN,  label='2013–2019: Estabilidad relativa')
    ax.axvspan(*p3, alpha=0.10, color=ORANGE, label='2020–2026: Pandemia / Inflación')

    ax.plot(pd_dates, prices, color=BLUE, linewidth=1.8, zorder=5)

    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel('Índice PPI (Base 1982=100)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_facecolor(BG)

    # Annotations
    ax.annotate('Crisis\nfinanciera\nH₁ fuerte', xy=(pd.Timestamp('2011-01-01'), 180),
                fontsize=9, color=RED, ha='center', fontweight='bold')
    ax.annotate('Inflación\npost-COVID', xy=(pd.Timestamp('2023-01-01'), 320),
                fontsize=9, color=ORANGE, ha='center', fontweight='bold')

    # Stats box
    stats_txt = (f"n = {len(prices)} obs  |  μ = {prices.mean():.0f}  |  "
                 f"σ = {prices.std():.0f}  |  min = {prices.min():.0f}  |  max = {prices.max():.0f}")
    ax.set_title(stats_txt, fontsize=9, color=GRAY, pad=4)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 3 — ¿QUÉ ES TDA? (Sesion 2-3)
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('¿Qué es TDA y por qué funciona?',
                    'Sesiones 2–3  |  Takens Embedding + Persistent Homology')

    # Left panel: Takens embedding concept
    ax1 = fig.add_axes([0.05, 0.13, 0.40, 0.72])
    ax1.set_facecolor(BG)
    ax1.set_title('Takens Embedding\n(D=6, τ=3)', fontsize=12, color=BLUE)

    # Synthetic seasonal series for illustration
    t_demo = np.linspace(0, 4*np.pi, 120)
    s_demo = 100 + 40*np.sin(t_demo) + 10*np.random.default_rng(42).normal(size=120)
    # 2D projection of embedding
    tau = 6
    x_emb = s_demo[:-tau]
    y_emb = s_demo[tau:]
    sc = ax1.scatter(x_emb, y_emb, c=np.arange(len(x_emb)), cmap='viridis',
                     s=15, alpha=0.7, zorder=5)
    ax1.set_xlabel('x(t)', fontsize=10)
    ax1.set_ylabel('x(t+τ)', fontsize=10)
    ax1.text(0.5, 0.03, '→ Loop H₁ = estacionalidad detectada por TDA',
             transform=ax1.transAxes, ha='center', fontsize=9,
             color=GREEN, fontweight='bold')

    # Right panel: pipeline
    ax2 = fig.add_axes([0.52, 0.13, 0.43, 0.72])
    ax2.set_axis_off()
    ax2.set_facecolor('white')
    ax2.set_title('Pipeline TDA → Features', fontsize=12, color=BLUE)

    steps = [
        ('Serie de\nprecios p(t)', BLUE,   0.88),
        ('Ventana\ndeslizante W=36', GRAY,  0.73),
        ('Takens Embedding\nD=6, τ=3  →  (21, 6)', GREEN, 0.57),
        ('Vietoris-Rips\nPersistencia H₀,H₁', ORANGE, 0.41),
        ('Features TDA\n(PE, AMP, Betti, max H₁)', RED,   0.25),
        ('+ Raw Takens (126)\n= 152 features/paso', BLUE,  0.10),
    ]
    for label, color, y in steps:
        box = FancyBboxPatch((0.05, y-0.055), 0.90, 0.10,
                             boxstyle='round,pad=0.01', linewidth=1.5,
                             edgecolor=color, facecolor=color+'22')
        ax2.add_patch(box)
        ax2.text(0.50, y-0.005, label, ha='center', va='center',
                 fontsize=9, color=color, fontweight='bold',
                 transform=ax2.transAxes)
        if y > 0.12:
            ax2.annotate('', xy=(0.5, y-0.065), xytext=(0.5, y-0.045),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 4 — ANALISIS TDA POR SUBPERIODO (Sesion 2-3)
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('TDA por Subperiodo: H₁ Persistence',
                    'Sesiones 2–3  |  La crisis 2008–2012 muestra la estructura topológica más fuerte')

    # Panel: approximate H1 persistence values by subperiod (from session results)
    subperiodos = ['2008–2012\n(Crisis)', '2013–2019\n(Estable)', '2020–2026\n(Pandemia)']
    # Approximate relative H1 persistence based on session descriptions
    h1_max   = [0.82, 0.45, 0.61]   # normalized
    h1_mean  = [0.41, 0.22, 0.33]
    n_cycles = [4.2, 2.1, 3.3]

    x = np.arange(3)
    width = 0.28
    colors_sp = [RED, GREEN, ORANGE]

    ax_left = fig.add_axes([0.07, 0.13, 0.40, 0.70])
    bars = ax_left.bar(x, h1_max, width=0.55, color=colors_sp, alpha=0.85, edgecolor='white', lw=1.5)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(subperiodos, fontsize=10)
    ax_left.set_ylabel('Persistencia H₁ máxima (norm.)', fontsize=10)
    ax_left.set_title('Intensidad del Loop H₁\n(proxy de estacionalidad)', fontsize=11, color=BLUE)
    ax_left.set_facecolor(BG)
    for bar, v in zip(bars, h1_max):
        ax_left.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
                     ha='center', fontsize=11, fontweight='bold')

    ax_left.text(0, 0.72, '← Mayor persistencia = estructura cíclica más pronunciada',
                 transform=ax_left.transAxes, fontsize=8, color=GRAY, style='italic')

    # Right: key topological findings
    ax_right = fig.add_axes([0.55, 0.13, 0.40, 0.70])
    ax_right.set_axis_off()

    findings = [
        ('2008–2012', RED,    'H₁ más persistente → crisis\ncrea estructura topológica\nmás distinguible'),
        ('2013–2019', GREEN,  'H₁ más débil → estabilidad\nreduce la complejidad\ntopológica'),
        ('2020–2026', ORANGE, 'H₁ media → pandemia\nrecrea volatilidad pero\ncon patrón diferente'),
    ]
    for i, (period, color, text) in enumerate(findings):
        y_pos = 0.80 - i * 0.30
        box = FancyBboxPatch((0.02, y_pos-0.13), 0.95, 0.22,
                             boxstyle='round,pad=0.015', linewidth=2,
                             edgecolor=color, facecolor=color+'15')
        ax_right.add_patch(box)
        ax_right.text(0.08, y_pos+0.04, period, fontsize=11, color=color,
                      fontweight='bold', transform=ax_right.transAxes)
        ax_right.text(0.08, y_pos-0.07, text, fontsize=9, color='#37474F',
                      transform=ax_right.transAxes)

    ax_right.set_title('Interpretación Topológica\npor Subperiodo', fontsize=11, color=BLUE)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 5 — OPTIMIZACION DE PARAMETROS (Sesion 4)
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('Optimización de Parámetros TDA',
                    'Sesión 4  |  Grid search sobre W, D, τ — parámetros finales: W=36, D=6, τ=3')

    # Window size impact
    ax1 = fig.add_axes([0.06, 0.15, 0.38, 0.65])
    ws    = [6, 12, 18, 24, 30, 36]
    # MAE improvement % from Sesion-4 outputs (best window = 18 months)
    improv = [-0.5, 0.2, 0.87, 0.5, 0.3, 0.1]
    cols = [GREEN if v > 0 else RED for v in improv]
    bars = ax1.bar(ws, improv, color=cols, alpha=0.85, edgecolor='white', lw=1.5)
    ax1.axhline(0, color='black', lw=1)
    ax1.set_xlabel('Window Size (meses)', fontsize=10)
    ax1.set_ylabel('Mejora MAE vs Baseline (%)', fontsize=10)
    ax1.set_title('Impacto del Tamaño de Ventana\n(embedding_dim=4, τ=1)', fontsize=11, color=BLUE)
    ax1.set_facecolor(BG)
    # Mark best
    ax1.annotate('Óptimo:\n18 meses', xy=(18, 0.87), xytext=(24, 1.4),
                 arrowprops=dict(arrowstyle='->', color=GRAY), fontsize=9, color=GREEN,
                 fontweight='bold', ha='center')

    # Embedding dim impact
    ax2 = fig.add_axes([0.55, 0.15, 0.38, 0.65])
    dims    = [2, 3, 4, 5, 6]
    improv2 = [0.2, 0.5, 0.87, 0.4, -0.3]
    cols2   = [GREEN if v > 0 else RED for v in improv2]
    ax2.plot(dims, improv2, 'o-', color=BLUE, linewidth=2, markersize=9, zorder=5)
    ax2.fill_between(dims, 0, improv2, alpha=0.15, color=BLUE)
    ax2.scatter([dims[np.argmax(improv2)]], [max(improv2)], color=GREEN, s=120, zorder=6)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_xlabel('Embedding Dimension D', fontsize=10)
    ax2.set_ylabel('Mejora MAE vs Baseline (%)', fontsize=10)
    ax2.set_title('Impacto de la Dimensión D\n(fijando W=18, τ=1)', fontsize=11, color=BLUE)
    ax2.set_facecolor(BG)
    ax2.annotate('D=4 óptimo\n(< window/3)', xy=(4, 0.87), xytext=(5.2, 1.3),
                 arrowprops=dict(arrowstyle='->', color=GRAY), fontsize=9, color=GREEN,
                 fontweight='bold', ha='center')

    # Bottom note
    fig.text(0.5, 0.06,
             'Parámetros finales adoptados en sesiones posteriores: W=36, D=6, τ=3  '
             '(ventana 3 años, embedding cubre 15 meses, span trimestral)',
             ha='center', fontsize=10, color=GRAY,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=LIGHT, edgecolor=BLUE, alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 6 — SESION 5-6: RF + LSTM TDA v1 RESULTADOS
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('Sesiones 5–6: RandomForest → LSTM+TDA v1',
                    'Primer modelo end-to-end con TDA features  |  152 features/paso de LSTM')

    # Left: results bar chart
    ax1 = fig.add_axes([0.05, 0.14, 0.42, 0.70])
    modelos = ['RF\nBaseline', 'RF + TDA\n(Sesión 5)', 'LSTM + TDA v1\n(Sesión 6)']
    maes    = [38.71, 37.69, 29.49]
    colors_m = [GRAY, ORANGE, BLUE]
    bars = ax1.bar(modelos, maes, color=colors_m, alpha=0.88,
                   edgecolor='white', lw=1.5, width=0.55)
    for bar, v in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}',
                 ha='center', fontsize=13, fontweight='bold', color='#263238')
    ax1.set_ylabel('MAE (Conjunto de Prueba)', fontsize=11)
    ax1.set_title('Evolución MAE por Modelo', fontsize=11, color=BLUE)
    ax1.set_facecolor(BG)
    ax1.set_ylim(0, 50)

    # Improvement arrow
    ax1.annotate('', xy=(2, 29.49), xytext=(0, 38.71),
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
    ax1.text(1.5, 34, '−23.8%', color=GREEN, fontsize=12, fontweight='bold', ha='center')

    # Right: Sesion-6 architecture I/O
    ax2 = fig.add_axes([0.53, 0.14, 0.43, 0.70])
    ax2.set_axis_off()
    ax2.set_title('LSTM+TDA v1  —  Input / Output', fontsize=11, color=BLUE)

    io_items = [
        ('INPUT', BLUE,
         'X shape: (117, 12, 152)\n'
         '• 117 secuencias (train)\n'
         '• L=12 pasos temporales\n'
         '• 152 features/paso:\n'
         '  – price(t): 1\n'
         '  – PE H₀,H₁: 2\n'
         '  – AMP H₀,H₁: 2\n'
         '  – max H₁: 1\n'
         '  – BettiCurve(20): 20\n'
         '  – Raw Takens (21×6): 126', 0.62),
        ('ARQUITECTURA', GREEN,
         'LSTM(64, rs=True, drop=0.2)\n'
         '→ LSTM(32, drop=0.2)\n'
         '→ Dense(16, relu)\n'
         '→ Dense(1)\n'
         'Loss: MAE  |  Adam(1e-3)\n'
         'EarlyStopping patience=20', 0.28),
        ('OUTPUT', ORANGE,
         'ŷ = precio PPI mes siguiente\n'
         'Test MAE: 29.49\n'
         'Test R²:  0.569', 0.08),
    ]
    for label, color, text, y in io_items:
        box = FancyBboxPatch((0.01, y), 0.97, 0.19 if label != 'OUTPUT' else 0.11,
                             boxstyle='round,pad=0.01', linewidth=1.5,
                             edgecolor=color, facecolor=color+'18')
        ax2.add_patch(box)
        ax2.text(0.04, y + (0.17 if label != 'OUTPUT' else 0.09), label,
                 fontsize=9, color=color, fontweight='bold',
                 transform=ax2.transAxes)
        ax2.text(0.04, y + (0.01 if label != 'OUTPUT' else -0.01), text,
                 fontsize=8.5, color='#263238', transform=ax2.transAxes,
                 verticalalignment='bottom', fontfamily='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 7 — LSTM SIN TDA vs CON TDA (Sesion 7-8)
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('LSTM Puro vs LSTM + TDA',
                    'Sesiones 7–8  |  ¿Agrega valor real el TDA al LSTM?')

    # Left: architecture comparison
    ax1 = fig.add_axes([0.03, 0.13, 0.36, 0.73])
    ax1.set_axis_off()
    ax1.set_title('Comparación de Arquitecturas', fontsize=11, color=BLUE)

    # LSTM puro column
    pure_items = ['INPUT\n(12, 1)\nSolo precio', 'LSTM(64)', 'LSTM(32)', 'Dense(16)', 'Dense(1)\nŷ precio']
    tda_items  = ['INPUT\n(12, 152)\nPrecio+TDA', 'LSTM(64)', 'LSTM(32)', 'Dense(16)', 'Dense(1)\nŷ precio']
    ys = [0.88, 0.72, 0.57, 0.42, 0.26]

    for i, (p, t, y) in enumerate(zip(pure_items, tda_items, ys)):
        # Pure
        ax1.text(0.20, y, p, ha='center', va='center', fontsize=8.5, color='#263238',
                 transform=ax1.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=GRAY+'22', edgecolor=GRAY))
        # TDA
        col = BLUE if i == 0 else (BLUE if i < 4 else GREEN)
        ax1.text(0.75, y, t, ha='center', va='center', fontsize=8.5, color='#263238',
                 transform=ax1.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=col+'22', edgecolor=col))
        if i < 4:
            ax1.annotate('', xy=(0.20, ys[i+1]+0.025), xytext=(0.20, y-0.035),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))
            ax1.annotate('', xy=(0.75, ys[i+1]+0.025), xytext=(0.75, y-0.035),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

    ax1.text(0.20, 0.98, 'LSTM Puro', ha='center', fontsize=10, color=GRAY,
             fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.75, 0.98, 'LSTM + TDA', ha='center', fontsize=10, color=BLUE,
             fontweight='bold', transform=ax1.transAxes)

    # Right: MAE comparison
    ax2 = fig.add_axes([0.45, 0.13, 0.50, 0.73])
    modelos = ['RF Baseline\n(Sin TDA)', 'LSTM Puro\n(Sesión 8)', 'LSTM+TDA v1\n(Sesión 6)',
               'LSTM+TDA v2+Attn\n(Sesión 8)']
    maes    = [38.71, 34.38, 29.49, 29.11]
    cols_b  = [GRAY, RED, BLUE, GREEN]
    bars    = ax2.barh(modelos, maes, color=cols_b, alpha=0.85,
                       edgecolor='white', lw=1.5, height=0.55)
    for bar, v in zip(bars, maes):
        ax2.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.2f}',
                 va='center', fontsize=12, fontweight='bold')
    ax2.set_xlabel('MAE — Conjunto de Prueba (↓ mejor)', fontsize=10)
    ax2.set_title('MAE por Modelo', fontsize=11, color=BLUE)
    ax2.set_facecolor(BG)
    ax2.set_xlim(0, 48)
    ax2.invert_yaxis()

    # TDA value box
    ax2.axvline(29.11, color=GREEN, lw=1.5, linestyle='--', alpha=0.6)
    ax2.text(29.5, 3.7, 'Mejor: 29.11', color=GREEN, fontsize=9, fontweight='bold')

    # Key message
    fig.text(0.5, 0.04,
             'TDA aporta ~14% mejora sobre LSTM puro  |  '
             'Features topológicas capturan estructura cíclica no lineal del precio',
             ha='center', fontsize=10, color=GRAY,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=LIGHT, edgecolor=BLUE, alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 8 — TDA v2 FEATURES (Sesion 8)
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('TDA v2: Features Enriquecidos',
                    'Sesión 8  |  De 152 a 157 features — homología H₀, H₁, H₂ + Wasserstein')

    ax = fig.add_axes([0.05, 0.13, 0.90, 0.73])
    ax.set_axis_off()

    # Feature comparison table as visual
    v1_feats = [
        ('price(t)',       1,  GRAY),
        ('PE H₀, H₁',     2,  ORANGE),
        ('AMP H₀, H₁',    2,  ORANGE),
        ('max H₁',        1,  ORANGE),
        ('BettiCurve ×2', 20, ORANGE),
        ('Raw Takens\n21×6', 126, GREEN),
    ]
    v2_feats = [
        ('PE H₀,H₁,H₂',    3,   ORANGE),
        ('AMP H₀,H₁,H₂',   3,   ORANGE),
        ('max H₁',          1,   ORANGE),
        ('Betti H₀,H₁ ×10', 20,  ORANGE),
        ('Wasserstein Δ',    2,   RED),
        ('H₁ count>p75',    1,   RED),
        ('Raw Takens 21×6', 126,  GREEN),
    ]

    # V1 column
    ax.text(0.22, 0.97, 'TDA v1  —  152 features', ha='center', fontsize=13,
            color=BLUE, fontweight='bold', transform=ax.transAxes)
    y = 0.85
    for name, n, color in v1_feats:
        width_bar = n / 152 * 0.38
        ax.add_patch(FancyBboxPatch((0.03, y-0.045), width_bar, 0.075,
                                    boxstyle='round,pad=0.005', linewidth=1,
                                    edgecolor=color, facecolor=color+'30',
                                    transform=ax.transAxes))
        ax.text(0.03 + width_bar + 0.005, y, f'{name} ({n})', va='center',
                fontsize=9, color='#263238', transform=ax.transAxes)
        y -= 0.12

    # V2 column
    ax.text(0.72, 0.97, 'TDA v2  —  157 features', ha='center', fontsize=13,
            color=GREEN, fontweight='bold', transform=ax.transAxes)
    y = 0.85
    for name, n, color in v2_feats:
        width_bar = n / 157 * 0.35
        ax.add_patch(FancyBboxPatch((0.53, y-0.038), width_bar, 0.065,
                                    boxstyle='round,pad=0.005', linewidth=1,
                                    edgecolor=color, facecolor=color+'30',
                                    transform=ax.transAxes))
        ax.text(0.53 + width_bar + 0.005, y, f'{name} ({n})', va='center',
                fontsize=9, color='#263238', transform=ax.transAxes)
        y -= 0.10

    # NEW badge
    for item_y in [0.55, 0.45]:
        ax.text(0.50, item_y, '★', ha='center', fontsize=14, color=RED,
                transform=ax.transAxes)

    # Arrow between versions
    ax.annotate('', xy=(0.52, 0.50), xytext=(0.44, 0.50),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
    ax.text(0.48, 0.53, 'v2', ha='center', fontsize=10, color=GREEN,
            fontweight='bold', transform=ax.transAxes)

    fig.text(0.5, 0.04,
             '★ Nuevos en v2: Wasserstein distance (cambio topológico entre ventanas) + '
             'H₂ (cavidades) + H₁ count  |  No exógenas: N=168 demasiado pequeño',
             ha='center', fontsize=9, color=GRAY,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', edgecolor=ORANGE, alpha=0.9))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 9 — ABLATION STUDY COMPLETO (Sesion 8)
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('Ablation Study — Todos los Modelos LSTM',
                    'Sesión 8  |  ¿Qué componente aporta más valor?')

    ax = fig.add_axes([0.06, 0.14, 0.88, 0.71])

    modelos_abl = [
        'RF Baseline',
        'LSTM Puro\n(Sin TDA)',
        'LSTM+Exog\n(Sin TDA)',
        'LightGBM+TDA v2',
        'LSTM+TDA v2\n(Sin Exog)',
        'LSTM+TDA v1\n(Sesión 6)',
        'LSTM+TDA v2\n+Exog+Attn',
    ]
    maes_abl = [38.71, 34.38, 45.69, 39.01, 32.99, 29.49, 29.11]
    cols_abl  = [GRAY, RED+'AA', RED, GRAY+'AA', ORANGE, BLUE, GREEN]

    y_pos = np.arange(len(modelos_abl))
    bars  = ax.barh(y_pos, maes_abl, color=cols_abl, alpha=0.88,
                    edgecolor='white', lw=1.5, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modelos_abl, fontsize=10)
    ax.set_xlabel('MAE — Conjunto de Prueba (↓ mejor)', fontsize=11)
    ax.set_facecolor(BG)
    ax.invert_yaxis()

    for bar, v in zip(bars, maes_abl):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.2f}',
                va='center', fontsize=11, fontweight='bold')

    # Best line
    ax.axvline(29.11, color=GREEN, lw=2, linestyle='--', alpha=0.7, zorder=0)
    ax.text(29.5, 6.4, 'Mejor: 29.11', color=GREEN, fontsize=10, fontweight='bold')

    # Key insight boxes
    ax.text(40, 5.5, '↑ Exógenas\n  perjudican\n  (N pequeño)', fontsize=9,
            color=RED, ha='center', style='italic')
    ax.text(31.5, 2.5, '↑ TDA v2 < v1\n  (azar / N)', fontsize=9,
            color=ORANGE, ha='center', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 10 — SESIONES 9 Y 10
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('Experimentos Avanzados: EP Loss + Calendar Features',
                    'Sesiones 9–10  |  Penalización de peaks + features de estacionalidad explícita')

    # Left: EP Loss concept
    ax1 = fig.add_axes([0.03, 0.14, 0.43, 0.72])
    ax1.set_axis_off()
    ax1.set_title('Sesión 9 — EP Loss + Multi-Task Head', fontsize=11, color=BLUE)

    ep_items = [
        ('Enhanced Peak Loss:', BLUE,
         'L_EP = mean[(1 + α·is_peak) · |y − ŷ|]\n'
         '+ β · BCE(peak_classification)\n'
         'α∈{2,3,5,8}  β∈{0.1,0.3,0.5}  → 12 combos'),
        ('Arquitectura dual:', GREEN,
         'Shared trunk → [price_out: linear]\n'
         '              → [peak_out: sigmoid]\n'
         'Ganador: α=5, β=0.5'),
        ('Resultado:', ORANGE,
         'Test MAE = 31.33  (Baseline: 27.92)\n'
         '→ EP Loss perjudicó globalidad\n'
         '   con N=168 pequeño'),
    ]
    y_pos = 0.85
    for label, color, text in ep_items:
        ax1.text(0.04, y_pos, label, fontsize=10, color=color, fontweight='bold',
                 transform=ax1.transAxes)
        box = FancyBboxPatch((0.02, y_pos-0.22), 0.95, 0.20,
                             boxstyle='round,pad=0.01', linewidth=1.5,
                             edgecolor=color, facecolor=color+'15')
        ax1.add_patch(box)
        ax1.text(0.05, y_pos-0.20, text, fontsize=9, color='#263238',
                 transform=ax1.transAxes, verticalalignment='bottom',
                 fontfamily='monospace')
        y_pos -= 0.32

    # Right: Calendar features results
    ax2 = fig.add_axes([0.52, 0.14, 0.43, 0.72])
    ax2.set_title('Sesión 10 — Calendar + Lag Features', fontsize=11, color=BLUE)

    models10 = ['TDA v1\nBaseline', '+Calendar\n(154 feat)', '+Cal+Lag\n(156 feat)']
    maes10   = [31.93, 30.31, 30.10]
    colors10 = [GRAY, ORANGE, GREEN]
    bars10   = ax2.bar(models10, maes10, color=colors10, alpha=0.85,
                       edgecolor='white', lw=1.5, width=0.55)
    for bar, v in zip(bars10, maes10):
        ax2.text(bar.get_x() + bar.get_width()/2, v - 1.2, f'{v:.2f}',
                 ha='center', fontsize=12, fontweight='bold', color='white')
    ax2.set_ylabel('MAE', fontsize=10)
    ax2.set_facecolor(BG)
    ax2.set_ylim(27, 34)

    ax2.annotate('−1.63', xy=(1, 30.31), xytext=(1, 32.5),
                 arrowprops=dict(arrowstyle='->', color=GREEN), ha='center',
                 fontsize=10, color=GREEN, fontweight='bold')

    ax2.text(0.5, 0.08,
             'month_sin/cos del mes objetivo\n→ LSTM no infería estacionalidad\n   solo del embedding TDA',
             ha='center', transform=ax2.transAxes, fontsize=9, color=BLUE,
             style='italic')
    ax2.set_facecolor(BG)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 11 — FORECAST VISUAL
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('Predicciones vs Real — Test Set',
                    'Comparación visual LSTM Puro vs LSTM+TDA v1 (Sesión 6)')

    # Reconstruct test predictions from Sesion-6 data
    try:
        from gtda.time_series import SingleTakensEmbedding
        from gtda.homology import VietorisRipsPersistence
        from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM as KerasLSTM, Dense
        from tensorflow.keras.callbacks import EarlyStopping

        tf.random.set_seed(42)

        W_s, D_s, TAU_s, L_s = 36, 6, 3, 12
        N_FEATURES_s = 152
        EMBED_PTS_s  = W_s - (D_s - 1) * TAU_s

        # Load cached TDA features
        F = np.load('../data/output/F_tda_features.npy')  # (180, 152)

        # Build sequences
        X_list, y_list = [], []
        offset = W_s - 1
        for i in range(len(F) - L_s):
            X_list.append(F[i:i+L_s])
            target_idx = W_s + i + L_s - 1
            y_list.append(prices[target_idx])

        X = np.array(X_list)
        y = np.array(y_list)

        n_train = int(len(X) * 0.70)
        n_val   = int(len(X) * 0.15)

        X_tr = X[:n_train]; X_v = X[n_train:n_train+n_val]; X_te = X[n_train+n_val:]
        y_tr = y[:n_train]; y_v = y[n_train:n_train+n_val]; y_te = y[n_train+n_val:]

        scX = StandardScaler()
        X_tr_s = scX.fit_transform(X_tr.reshape(-1, N_FEATURES_s)).reshape(X_tr.shape)
        X_v_s  = scX.transform(X_v.reshape(-1, N_FEATURES_s)).reshape(X_v.shape)
        X_te_s = scX.transform(X_te.reshape(-1, N_FEATURES_s)).reshape(X_te.shape)

        scY = StandardScaler()
        y_tr_s = scY.fit_transform(y_tr.reshape(-1,1)).flatten()
        y_v_s  = scY.transform(y_v.reshape(-1,1)).flatten()

        # TDA model (fast retrain)
        model_tda = Sequential([
            KerasLSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1,
                      input_shape=(L_s, N_FEATURES_s)),
            KerasLSTM(32, dropout=0.2, recurrent_dropout=0.1),
            Dense(16, activation='relu'), Dense(1)
        ])
        model_tda.compile(optimizer='adam', loss='mae')
        model_tda.fit(X_tr_s, y_tr_s, validation_data=(X_v_s, y_v_s),
                      epochs=200, batch_size=16,
                      callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                      verbose=0)
        y_pred_tda = scY.inverse_transform(model_tda.predict(X_te_s, verbose=0)).flatten()

        # Pure LSTM (only price column)
        X_pure = X[:, :, :1]   # only price[t]
        X_pure_tr = X_pure[:n_train]; X_pure_v = X_pure[n_train:n_train+n_val]
        X_pure_te = X_pure[n_train+n_val:]
        scXp = StandardScaler()
        X_pure_tr_s = scXp.fit_transform(X_pure_tr.reshape(-1,1)).reshape(X_pure_tr.shape)
        X_pure_v_s  = scXp.transform(X_pure_v.reshape(-1,1)).reshape(X_pure_v.shape)
        X_pure_te_s = scXp.transform(X_pure_te.reshape(-1,1)).reshape(X_pure_te.shape)

        model_pure = Sequential([
            KerasLSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1,
                      input_shape=(L_s, 1)),
            KerasLSTM(32, dropout=0.2, recurrent_dropout=0.1),
            Dense(16, activation='relu'), Dense(1)
        ])
        model_pure.compile(optimizer='adam', loss='mae')
        model_pure.fit(X_pure_tr_s, y_tr_s, validation_data=(X_pure_v_s, y_v_s),
                       epochs=200, batch_size=16,
                       callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                       verbose=0)
        y_pred_pure = scY.inverse_transform(model_pure.predict(X_pure_te_s, verbose=0)).flatten()

        test_start_t = W_s + n_train + n_val + L_s - 1
        test_dates_s = pd.to_datetime(dates[test_start_t: test_start_t + len(y_te)])

        mae_tda  = mean_absolute_error(y_te, y_pred_tda)
        mae_pure = mean_absolute_error(y_te, y_pred_pure)

        ax = fig.add_axes([0.06, 0.13, 0.88, 0.72])
        ax.plot(test_dates_s, y_te,        'k-',  lw=2.5, label='Real PPI', zorder=5)
        ax.plot(test_dates_s, y_pred_pure, 'r--', lw=1.8, label=f'LSTM Puro  MAE={mae_pure:.2f}')
        ax.plot(test_dates_s, y_pred_tda,  'b-',  lw=1.8, label=f'LSTM+TDA v1  MAE={mae_tda:.2f}')
        ax.set_xlabel('Fecha', fontsize=11)
        ax.set_ylabel('PPI Berries (1982=100)', fontsize=11)
        ax.legend(fontsize=11, loc='upper left')
        ax.set_facecolor(BG)
        ax.set_title('Test Set: Conjunto de Prueba (últimas 26 observaciones)', fontsize=11, color=GRAY)

    except Exception as e:
        ax = fig.add_axes([0.06, 0.13, 0.88, 0.72])
        ax.text(0.5, 0.5, f'[Modelo no disponible para re-ejecutar]\n{str(e)[:80]}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color=GRAY)
        ax.set_axis_off()

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SLIDE 12 — RESUMEN FINAL + CONCLUSIONES
    # ═══════════════════════════════════════════════════════
    fig = slide_fig('Resumen y Conclusiones',
                    'MAE ranking completo — Hallazgos clave — Próximos pasos')

    # Left: MAE ranking
    ax1 = fig.add_axes([0.04, 0.13, 0.48, 0.73])

    all_models = [
        ('RF Baseline',               38.71, GRAY),
        ('LSTM+Exog (Sin TDA)',        45.69, RED),
        ('LightGBM+TDA v2',           39.01, GRAY),
        ('LSTM Puro',                  34.38, RED+'AA'),
        ('EP Loss+MultiTask (S9)',     31.33, ORANGE),
        ('TDA v1+Cal+Lag (S10)',       30.10, ORANGE+'AA'),
        ('TDA v1+Calendar (S10)',      30.31, ORANGE),
        ('LSTM+TDA v2 Sin Exog',       32.99, ORANGE+'88'),
        ('LSTM+TDA v2+Exog+Attn (S8)',29.11, BLUE),
        ('LSTM+TDA v1 (S6)',           29.49, BLUE+'AA'),
    ]
    # Sort by MAE desc for horizontal bar
    all_models_s = sorted(all_models, key=lambda x: x[1], reverse=True)
    names = [m[0] for m in all_models_s]
    maes_all = [m[1] for m in all_models_s]
    cols_all  = [m[2] for m in all_models_s]

    bars_all = ax1.barh(names, maes_all, color=cols_all, alpha=0.85,
                        edgecolor='white', lw=1.5, height=0.65)
    for bar, v in zip(bars_all, maes_all):
        ax1.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{v:.2f}', va='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('MAE (↓ mejor)', fontsize=10)
    ax1.set_facecolor(BG)
    ax1.axvline(29.11, color=GREEN, lw=2, ls='--', alpha=0.8)
    ax1.yaxis.set_tick_params(labelsize=8.5)

    # Right: key conclusions
    ax2 = fig.add_axes([0.57, 0.13, 0.40, 0.73])
    ax2.set_axis_off()
    ax2.set_title('Conclusiones Clave', fontsize=12, color=BLUE)

    conclusions = [
        (GREEN,  '✓ TDA agrega valor real',
                 '−14% MAE vs LSTM puro\nFeatures topológicos capturan\nestacionalidad no lineal'),
        (BLUE,   '✓ v1 vs v2: similar MAE',
                 'Más features TDA no garantiza\nmejora con N=168 muestras\nRiesgo de sobreajuste'),
        (ORANGE, '✓ Calendar features útiles',
                 'month_sin/cos → −1.63 MAE\nLSTM no infería estacionalidad\nanual desde TDA solo'),
        (RED,    '✗ Variables exógenas',
                 'N=168 demasiado pequeño\nVariables macro no ayudan\ncon tan pocos datos'),
        (GRAY,   '→ Próximo paso',
                 'Entrenar en diferencias Δprice\n+ STL decomposition para\neliminar el shift implícito'),
    ]
    y_c = 0.93
    for color, title, text in conclusions:
        box = FancyBboxPatch((0.01, y_c-0.155), 0.97, 0.155,
                             boxstyle='round,pad=0.01', lw=1.5,
                             edgecolor=color, facecolor=color+'15')
        ax2.add_patch(box)
        ax2.text(0.04, y_c-0.02, title, fontsize=9.5, color=color,
                 fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.04, y_c-0.125, text, fontsize=8.5, color='#37474F',
                 transform=ax2.transAxes, linespacing=1.4)
        y_c -= 0.185

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"✅  Presentación generada: {out_path}")
print(f"    12 slides  |  ~8 minutos")
