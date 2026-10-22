"""Generador del Reporte Técnico de Validación Fase 2A en formato PDF.

Proyecto ChileFuego: Sistema de Predicción y Alerta Temprana de Incendios Forestales para Ñuble.
Produce un informe técnico formal, de alta calidad gráfica y tipográfica con ReportLab (4 páginas exactas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Canvas de dos pasadas para numeración dinámica 'Página X de Y' y encabezados elegantes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count: int) -> None:
        self.saveState()

        # Encabezado en páginas posteriores a la portada
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1E3A8A"))
            self.drawString(54, 755, "PROYECTO CHILEFUEGO")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#4B5563"))
            self.drawString(165, 755, "|  Reporte Técnico de Validación Fase 2A — Ñuble, Chile")

            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(54, 747, 558, 747)

        # Pie de página en todas las páginas
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 34, "Fase 2A: Rejilla 1 km · ERA5-Land · FWI Canadiense · Benchmark LOSO sin ML")

        page_str = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(558, 34, page_str)

        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def build_pdf_report(output_path: Path, figures_dir: Path) -> Path:
    """Construye el documento PDF completo en exactamente 4 páginas de alta densidad técnica."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Paleta de colores técnicos
    c_primary = colors.HexColor("#1E3A8A")     # Navy
    c_secondary = colors.HexColor("#B91C1C")   # Fire Red
    c_accent = colors.HexColor("#0D9488")      # Deep Teal
    c_dark = colors.HexColor("#0F172A")        # Slate 900
    c_body = colors.HexColor("#334155")        # Slate 700
    c_light_bg = colors.HexColor("#F8FAFC")    # Slate 50
    c_card_border = colors.HexColor("#E2E8F0") # Slate 200

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=c_primary,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=c_accent,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=c_body,
        spaceAfter=5,
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=c_dark,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell,
        alignment=1,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
    )

    table_cell_center_bold = ParagraphStyle(
        "TableCellCenterBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
        alignment=1,
    )

    story: list[Any] = []

    # =========================================================================
    # PÁGINA 1: PORTADA, RESUMEN EJECUTIVO Y DOMINIO ESPACIOTEMPORAL (1 KM)
    # =========================================================================
    badge_table = Table(
        [[
            Paragraph("<b>PROYECTO CHILEFUEGO</b> · MODELO OPERACIONAL REGIONAL", meta_style),
            Paragraph("<b>FASE 2A</b> · OCTUBRE 2026 · V1.0 FINAL", ParagraphStyle("BadgeR", parent=meta_style, alignment=2)),
        ]],
        colWidths=[330, 174],
    )
    badge_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("Reporte Técnico de Validación — Fase 2A", title_style))
    story.append(Paragraph(
        "Rejilla Operacional a 1 km, Ingesta Criptográfica ERA5-Land, Sistema Canadiense FWI "
        "y Benchmark Leave-One-Season-Out (7 Pliegues) sin Machine Learning en la Región de Ñuble",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=8))

    exec_summary_html = """
    <b>SÍNTESIS EJECUTIVA Y ALCANCE:</b> La Fase 2A consolida la columna vertebral espaciotemporal del sistema predictor
    para la Región de Ñuble, adoptando formalmente como target primario las igniciones registradas por CONAF a resolución
    de celda regular de 1 km y ventana diaria de 24 horas.<br/><br/>
    <b>• Ingesta ERA5-Land y Auditoría:</b> 99 NetCDF mensuales (2017-04 a 2025-06) auditados contra el manifiesto SHA-256
    (100% íntegros, 0 faltantes). Desacumulación de precipitación verificada con balance idéntico a las 00 UTC siguientes.<br/>
    <b>• FWI Canadiense Riguroso:</b> Cálculo vectorial continuo al mediodía local estándar (12:00 CLT / 16:00 UTC). El test de
    insensibilidad al arranque con datos reales ERA5-Land sobre 5 zonas geográficas (Costa, Valle, Precordillera, Cordillera y Sur)
    certificó convergencia absoluta (&Delta;DC = 0.0, &Delta;DMC = 0.0, &Delta;FWI = 0.0) antes del inicio del verano.<br/>
    <b>• Panel Diario y Universo Efectivo:</b> Se generaron <b>33.345.071 celdas-día</b> efectivas en 7 temporadas fechadas,
    conteniendo <b>3.237 positivos CONAF colapsados</b> (tasa base efectiva = 9.708 &times; 10<sup>-5</sup>).<br/>
    <b>• Benchmark LOSO sin ML:</b> El baseline multiplicativo <b>FWI &times; Prior Espacial Creciente</b> alcanza un
    <b>PR-AUC medio de 1.171 &times; 10<sup>-3</sup> (Lift 12.66x)</b>, capturando el <b>17.27% de las igniciones en el 1% de celdas-día
    de mayor riesgo</b> y el <b>42.86% en el top 5%</b>, superando con creces a cualquiera de los predictores aislados.
    """
    summary_table = Table([[Paragraph(exec_summary_html, callout_style)]], colWidths=[504])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c_light_bg),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#CBD5E1")),
        ("LINELEFT", (0, 0), (0, 0), 4.0, c_secondary),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Definición del Dominio Espaciotemporal y Rejilla Operacional (F2.1 - F2.2)", h1_style))
    story.append(Paragraph(
        "Se adoptó una cuadrícula regular de <b>1.000 m &times; 1.000 m (1 km)</b> proyectada en EPSG:32719 "
        "(WGS 84 / UTM Huso 19S). Sobre el límite administrativo oficial SUBDERE de la Región de Ñuble, se indexaron "
        "<b>13.108 celdas terrestres</b>, totalizando un área modelada de <b>13.108,00 km²</b> (discrepancia relativa de "
        "<b>-0.53%</b> frente a los 13.178 km² oficiales, cumpliendo estrictamente el criterio de tolerancia de &plusmn;5%).<br/>"
        "El calendario comprende las 7 temporadas con registro intra-estacional de fecha: 2017_18 y 2019_20 a 2024_25 "
        "(2.557 días). Se excluyó la temporada 2018_19 (sin fechas) y se aplicaron máscaras sanitarias para evitar artefactos:",
        body_style,
    ))

    domain_data = [
        [Paragraph("Categoría / Motivo", table_header), Paragraph("Celdas-Día", table_header), Paragraph("% Universo", table_header), Paragraph("Justificación y Criterio Metodológico", table_header)],
        [Paragraph("Universo Bruto Fechado", table_cell_bold), Paragraph("33.517.156", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("13.108 celdas × 2.557 días en 7 temporadas fechadas", table_cell)],
        [Paragraph("undated_season (2018_19)", table_cell), Paragraph("4.784.420", table_cell_center), Paragraph("—", table_cell_center), Paragraph("Temporada sin fechas; excluida de entrenamiento y evaluación", table_cell)],
        [Paragraph("conaf_active_footprint", table_cell), Paragraph("132.139", table_cell_center), Paragraph("0.39%", table_cell_center), Paragraph("Máscara temporal activa entre inicio y extinción reportada CONAF", table_cell)],
        [Paragraph("firms_unmatched_summer_rule_b", table_cell), Paragraph("39.946", table_cell_center), Paragraph("0.12%", table_cell_center), Paragraph("Focos satelitales FIRMS de alta confianza en verano sin reporte CONAF", table_cell)],
        [Paragraph("small_or_agri", table_cell), Paragraph("0", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("No se excluyen: fuegos pequeños y agrícolas permanecen como target", table_cell)],
        [Paragraph("<b>Universo Efectivo Entrenamiento</b>", table_cell_bold), Paragraph("<b>33.345.071</b>", table_cell_center_bold), Paragraph("<b>99.49%</b>", table_cell_center_bold), Paragraph("<b>Celdas-día limpias disponibles para validación honesta</b>", table_cell_bold)],
        [Paragraph("<b>Positivos CONAF Colapsados</b>", table_cell_bold), Paragraph("<b>3.237</b>", table_cell_center_bold), Paragraph("<b>9.708 &times; 10<sup>-5</sup></b>", table_cell_center_bold), Paragraph("<b>1 ignición positiva por cada 10.301 celdas-día efectivas</b>", table_cell_bold)],
    ]
    t_domain = Table(domain_data, colWidths=[140, 75, 55, 234])
    t_domain.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("GRID", (0, 0), (-1, -1), 0.5, c_card_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -3), [colors.white, c_light_bg]),
        ("BACKGROUND", (0, -2), (-1, -1), colors.HexColor("#FEF3C7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_domain)

    # Fin Página 1
    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 2: METEOROLOGÍA ERA5-LAND, TEST DE ARRANQUE Y FWI CANADIENSE (B1-B3)
    # =========================================================================
    story.append(Paragraph("2. Ingesta Meteorológica ERA5-Land y Test de Insensibilidad al Arranque (B1 - B2)", h1_style))
    story.append(Paragraph(
        "<b>Auditoría Criptográfica (B1):</b> Se descargaron <b>99 archivos NetCDF mensuales</b> de ERA5-Land "
        "(0.1° de resolución, ~9 km) abarcando ininterrumpidamente desde <b>2017-04-01 al 2025-06-30</b> (3.013 días). "
        "La verificación contra <code>data/raw/era5_land/manifest.json</code> certificó que el <b>100% de los archivos "
        "(99/99) se encuentran presentes y validados con su hash SHA-256 oficial</b>, garantizando trazabilidad y reproducibilidad.<br/>"
        "<b>Desacumulación Horaria de Precipitación:</b> ERA5-Land acumula la precipitación total (<i>tp</i>) hora a hora "
        "desde las 01:00 UTC hasta las 24:00 UTC (00:00 UTC del día siguiente), momento en que el acumulador se reinicia a cero. "
        "Se aplicó el algoritmo de desacumulación horaria diferencial. La prueba de regresión unitaria confirmó que la suma de "
        "las 24 horas desacumuladas es numéricamente idéntica al acumulador de las 00:00 UTC siguientes (&Delta; &lt; 10<sup>-5</sup> mm).",
        body_style,
    ))

    story.append(Paragraph(
        "<b>Test de Insensibilidad al Arranque con Datos Reales (B2):</b> Para asegurar que el cálculo de FWI en Chile "
        "no dependa de condiciones iniciales arbitrarias, se ejecutó un test de convergencia con ERA5-Land <b>real</b> "
        "desde abril 2017 hasta febrero 2018 en 5 nodos geográficos contrastantes de Ñuble, comparando un arranque húmedo estándar "
        "(<i>DC<sub>0</sub>=15, DMC<sub>0</sub>=6</i>) contra un arranque en sequía extrema (<i>DC<sub>0</sub>=500, DMC<sub>0</sub>=100</i>):",
        body_style,
    ))

    b2_data = [
        [Paragraph("Zona Geográfica", table_header), Paragraph("Nodo ERA5-Land", table_header), Paragraph("Referencia Geográfica", table_header), Paragraph("&Delta; Máx. DC<br/>(Dic-Feb)", table_header), Paragraph("&Delta; Máx. DMC<br/>(Dic-Feb)", table_header), Paragraph("&Delta; Máx. FWI<br/>(Dic-Feb)", table_header)],
        [Paragraph("Costa", table_cell_bold), Paragraph("-36.1° / -72.8°", table_cell_center), Paragraph("Cobquecura / Litoral Norte", table_cell), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center_bold)],
        [Paragraph("Valle Central", table_cell_bold), Paragraph("-36.6° / -72.1°", table_cell_center), Paragraph("Chillán / Valle Agrícola", table_cell), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center_bold)],
        [Paragraph("Precordillera", table_cell_bold), Paragraph("-36.6° / -71.6°", table_cell_center), Paragraph("San Fabián de Alico", table_cell), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center_bold)],
        [Paragraph("Cordillera Alta", table_cell_bold), Paragraph("-36.9° / -71.4°", table_cell_center), Paragraph("Pinto / Termas de Chillán", table_cell), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center_bold)],
        [Paragraph("Sur Regional", table_cell_bold), Paragraph("-36.8° / -72.5°", table_cell_center), Paragraph("Quillón / Valle del Sol", table_cell), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center), Paragraph("0.0000", table_cell_center_bold)],
    ]
    t_b2 = Table(b2_data, colWidths=[80, 75, 125, 75, 75, 74])
    t_b2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_accent),
        ("GRID", (0, 0), (-1, -1), 0.5, c_card_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_b2)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Serie Continua de FWI Canadiense y Mapeo Espacial a 1 km (B3)", h1_style))
    story.append(Paragraph(
        "Siguiendo el estándar de Van Wagner (1987) y Wang et al. (2015), los seis componentes del FWI "
        "(FFMC, DMC, DC, ISI, BUI y FWI) se calcularon diariamente al <b>mediodía solar estándar (12:00 CLT / 16:00 UTC)</b> "
        "con entradas de temperatura a 2 m, humedad relativa (ecuación de Tetens sobre <i>t2m</i> y <i>d2m</i>), rapidez de viento a 10 m "
        "y lluvia acumulada en 24 horas. Para respetar la memoria hidrológica del suelo, el cálculo se ejecutó de forma continua "
        "sobre los 3.013 días (incluyendo 2018_19).<br/>"
        "• <b>Eficiencia Computacional:</b> 399 nodos terrestres &times; 3.013 días (1.202.187 registros diarios) calculados en "
        "<b>64.42 s</b> y persistidos en <code>era5_land_daily_fwi_nodes.parquet</code> (25.18 MB, 0 nulos).<br/>"
        "• <b>Mapeo a 1 km:</b> Cada celda fue asociada a su nodo más próximo mediante cKDTree. <b>0.00% de celdas-día sin dato meteorológico.</b>",
        body_style,
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("4. Benchmark Leave-One-Season-Out (LOSO): Comparación Síntesis", h1_style))
    story.append(Paragraph(
        "Se evaluaron tres modelos heurísticos honestos sin machine learning mediante validación cruzada Leave-One-Season-Out (7 pliegues). "
        "El Prior Espacial utiliza ventana creciente temporal (temporadas 2010–2016, 2018_19 si <i>s &ge; 2019_20</i>, y temporadas fechadas "
        "anteriores a <i>s</i>), garantizando cero fuga del futuro:",
        body_style,
    ))

    comp_data = [
        [
            Paragraph("Modelo Heurístico Evaluado", table_header),
            Paragraph("PR-AUC Promedio", table_header),
            Paragraph("Lift Medio", table_header),
            Paragraph("Recall @ 1%", table_header),
            Paragraph("Recall @ 5%", table_header),
            Paragraph("Rec@1% (&ge;1 ha)", table_header),
            Paragraph("Rec@1% (&ge;10 ha)", table_header),
        ],
        [
            Paragraph("<b>(a) FWI Solo (Meteorología Instantánea)</b>", table_cell),
            Paragraph("3.668 &times; 10<sup>-4</sup>", table_cell_center),
            Paragraph("3.85x", table_cell_center),
            Paragraph("6.61%", table_cell_center),
            Paragraph("21.90%", table_cell_center),
            Paragraph("8.16%", table_cell_center),
            Paragraph("<b>11.88%</b>", table_cell_center_bold),
        ],
        [
            Paragraph("<b>(b) Prior Espacial Creciente Solo</b>", table_cell),
            Paragraph("4.786 &times; 10<sup>-4</sup>", table_cell_center),
            Paragraph("5.08x", table_cell_center),
            Paragraph("7.66%", table_cell_center),
            Paragraph("28.45%", table_cell_center),
            Paragraph("6.14%", table_cell_center),
            Paragraph("4.84%", table_cell_center),
        ],
        [
            Paragraph("<b>(c) FWI &times; Prior Espacial Creciente</b>", table_cell_bold),
            Paragraph("<b>1.171 &times; 10<sup>-3</sup></b>", table_cell_center_bold),
            Paragraph("<b>12.66x</b>", table_cell_center_bold),
            Paragraph("<b>17.27%</b>", table_cell_center_bold),
            Paragraph("<b>42.86%</b>", table_cell_center_bold),
            Paragraph("<b>15.24%</b>", table_cell_center_bold),
            Paragraph("<b>12.08%</b>", table_cell_center_bold),
        ],
    ]
    t_comp = Table(comp_data, colWidths=[144, 60, 50, 60, 60, 65, 65])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("GRID", (0, 0), (-1, -1), 0.5, c_card_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, c_light_bg]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#FEF08A")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)

    # Fin Página 2
    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 3: RESULTADOS DETALLADOS LOSO, FIGURAS CIENTÍFICAS Y TEMPORALIDAD
    # =========================================================================
    story.append(Paragraph("Desempeño Detallado por Pliegue (Temporada): Modelo C (FWI &times; Prior Creciente)", h2_style))
    story.append(Paragraph(
        "La tabla refleja los 7 pliegues Leave-One-Season-Out del modelo producto. La superficie del incendio "
        "(<i>superficie_ha</i>) se utiliza exclusivamente como estrato evaluativo post-hoc, sin intervenir en el ranking predictivo:",
        body_style,
    ))

    loso_detail_data = [
        [
            Paragraph("Temporada (Fold)", table_header),
            Paragraph("Positivos (N)", table_header),
            Paragraph("PR-AUC", table_header),
            Paragraph("Lift Base Rate", table_header),
            Paragraph("Rec@1% (Total)", table_header),
            Paragraph("Rec@5% (Total)", table_header),
            Paragraph("Rec@1% (&ge;1 ha)", table_header),
            Paragraph("Rec@1% (&ge;10 ha)", table_header),
            Paragraph("Top10% Días FWI", table_header),
        ],
        [Paragraph("2017_18", table_cell_bold), Paragraph("444", table_cell_center), Paragraph("2.818 &times; 10<sup>-3</sup>", table_cell_center), Paragraph("32.12x", table_cell_center_bold), Paragraph("27.88%", table_cell_center), Paragraph("56.01%", table_cell_center), Paragraph("24.84% (N=153)", table_cell), Paragraph("23.68% (N=38)", table_cell), Paragraph("41.59%", table_cell_center)],
        [Paragraph("2019_20", table_cell_bold), Paragraph("471", table_cell_center), Paragraph("9.046 &times; 10<sup>-4</sup>", table_cell_center), Paragraph("9.18x", table_cell_center_bold), Paragraph("15.92%", table_cell_center), Paragraph("42.68%", table_cell_center), Paragraph("16.05% (N=162)", table_cell), Paragraph("13.04% (N=46)", table_cell), Paragraph("33.76%", table_cell_center)],
        [Paragraph("2020_21", table_cell_bold), Paragraph("469", table_cell_center), Paragraph("7.500 &times; 10<sup>-4</sup>", table_cell_center), Paragraph("7.61x", table_cell_center_bold), Paragraph("13.22%", table_cell_center), Paragraph("40.72%", table_cell_center), Paragraph("9.92% (N=121)", table_cell), Paragraph("8.33% (N=24)", table_cell), Paragraph("26.01%", table_cell_center)],
        [Paragraph("2021_22", table_cell_bold), Paragraph("516", table_cell_center), Paragraph("8.268 &times; 10<sup>-4</sup>", table_cell_center), Paragraph("7.64x", table_cell_center_bold), Paragraph("13.95%", table_cell_center), Paragraph("39.73%", table_cell_center), Paragraph("14.20% (N=176)", table_cell), Paragraph("10.26% (N=39)", table_cell), Paragraph("28.29%", table_cell_center)],
        [Paragraph("2022_23", table_cell_bold), Paragraph("584", table_cell_center), Paragraph("8.629 &times; 10<sup>-4</sup>", table_cell_center), Paragraph("7.01x", table_cell_center_bold), Paragraph("12.33%", table_cell_center), Paragraph("39.04%", table_cell_center), Paragraph("12.62% (N=206)", table_cell), Paragraph("9.26% (N=54)", table_cell), Paragraph("22.95%", table_cell_center)],
        [Paragraph("2023_24", table_cell_bold), Paragraph("344", table_cell_center), Paragraph("9.501 &times; 10<sup>-4</sup>", table_cell_center), Paragraph("13.21x", table_cell_center_bold), Paragraph("17.44%", table_cell_center), Paragraph("40.41%", table_cell_center), Paragraph("8.25% (N=97)", table_cell), Paragraph("0.00% (N=19)", table_cell), Paragraph("33.43%", table_cell_center)],
        [Paragraph("2024_25", table_cell_bold), Paragraph("409", table_cell_center), Paragraph("1.082 &times; 10<sup>-3</sup>", table_cell_center), Paragraph("11.81x", table_cell_center_bold), Paragraph("20.14%", table_cell_center), Paragraph("41.42%", table_cell_center), Paragraph("20.80% (N=125)", table_cell), Paragraph("20.00% (N=35)", table_cell), Paragraph("30.43%", table_cell_center)],
        [
            Paragraph("<b>PROMEDIO 7P</b>", table_cell_bold),
            Paragraph("<b>3.237 total</b>", table_cell_center_bold),
            Paragraph("<b>1.171 &times; 10<sup>-3</sup></b>", table_cell_center_bold),
            Paragraph("<b>12.66x</b>", table_cell_center_bold),
            Paragraph("<b>17.27%</b>", table_cell_center_bold),
            Paragraph("<b>42.86%</b>", table_cell_center_bold),
            Paragraph("<b>15.24%</b>", table_cell_center_bold),
            Paragraph("<b>12.08%</b>", table_cell_center_bold),
            Paragraph("<b>30.92%</b>", table_cell_center_bold),
        ],
    ]
    t_loso = Table(loso_detail_data, colWidths=[60, 48, 58, 48, 52, 52, 64, 64, 58])
    t_loso.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("GRID", (0, 0), (-1, -1), 0.5, c_card_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, c_light_bg]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#DCFCE7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_loso)
    story.append(Spacer(1, 8))

    # Inserción de Figuras Científicas
    fig1_path = figures_dir / "fig1_loso_performance.png"
    fig2_path = figures_dir / "fig2_recall_tradeoff.png"

    if fig1_path.exists():
        story.append(Image(str(fig1_path), width=504, height=180))
        story.append(Spacer(1, 4))
    if fig2_path.exists():
        story.append(Image(str(fig2_path), width=504, height=180))
        story.append(Spacer(1, 6))

    # Fin Página 3
    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 4: DISCUSIÓN TÉCNICA, HOJA DE RUTA FASE 2B Y AUDITORÍA DE CALIDAD
    # =========================================================================
    story.append(Paragraph("5. Discusión Técnica y Mecanismos de Predicción (B5)", h1_style))
    story.append(Paragraph(
        "<b>1. Complementariedad Estructural entre Clima y Geografía Humana:</b><br/>"
        "• El <b>FWI meteorológico aislado</b> presenta un PR-AUC modesto (Lift 3.85x) debido a que el peligro atmosférico cubre "
        "amplias extensiones sincrónicamente sin discriminar la presencia de fuentes de ignición antrópica. Sin embargo, en la "
        "estratificación de incendios mayores (<b>&ge; 10 ha</b>), el FWI solo alcanza un Recall@1% de <b>11.88%</b>, "
        "superando ampliamente al prior espacial (4.84%). Esto confirma que la atmósfera extrema es el principal habilitador del escape.<br/>"
        "• El <b>Prior Espacial Creciente aislado</b> captura los hotspots recurrentes de actividad humana (Lift 5.08x, "
        "Recall@5% de 28.45%), pero resulta insensible al ciclo estacional y diario de sequía y olas de calor.<br/>"
        "• La interacción multiplicativa <b>FWI &times; Prior</b> fusiona las ventajas de ambos mundos: eleva el lift a <b>12.66x</b> "
        "(+144% sobre el prior y +219% sobre el FWI), permitiendo que priorizar únicamente el <b>5% del territorio regional</b> "
        "capture el <b>42.86% de todas las igniciones</b> efectivas ocurridas en 7 temporadas.<br/>"
        "<b>2. Dinámica Temporal Regional:</b> El <b>30.92% de las igniciones CONAF ocurren en el 10% de días con mayor FWI regional</b> "
        "(frente al 10.0% esperado por azar), validando que la alerta meteorológica concentra de forma efectiva la actividad de fuego.<br/>"
        "<b>3. Comportamiento de la Ventana Creciente:</b> La temporada 2017_18 (la más temprana y con menor historial base) "
        "exhibe un lift sobresaliente (32.12x), mientras que en temporadas más recientes el lift se estabiliza entre 7.0x y 13.2x "
        "a medida que el prior espacial se suaviza espacialmente.",
        body_style,
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("6. Hoja de Ruta Operacional hacia la Fase 2B", h1_style))
    story.append(Paragraph(
        "Habiendo establecido un benchmark sin machine learning riguroso y honesto, la Fase 2B integrará:<br/>"
        "• <b>Variables Fisiográficas y Topográficas Estáticas:</b> Modelo Digital de Elevación (DEM), pendiente, orientación "
        "(aspect), índice de radiación y Topographic Wetness Index (TWI) remuestreados a la rejilla de 1 km.<br/>"
        "• <b>Combustibles y Vulnerabilidad Territorial:</b> Clasificación de combustible forestal (<i>fuel_model_chile</i>), "
        "interfaz urbano-forestal (WUI) y proximidad euclidiana a la red vial y centros poblados.<br/>"
        "• <b>Modelación Predictiva Supervisada:</b> Algoritmos de Gradient Boosting (LightGBM/XGBoost) con funciones de costo "
        "focalizadas en desbalance extremo, respetando rigurosamente la validación cruzada temporal Leave-One-Season-Out (7 pliegues).",
        body_style,
    ))
    story.append(Spacer(1, 8))

    qa_box = [
        [
            Paragraph("<b>AUDITORÍA DE CALIDAD Y CERTIFICACIÓN DEL REPOSITORIO</b>", ParagraphStyle("QATitle", parent=table_cell_bold, textColor=c_primary)),
            Paragraph("<b>ESTADO FINAL FASE 2A</b>", ParagraphStyle("QATitleR", parent=table_cell_bold, textColor=c_accent, alignment=2)),
        ],
        [
            Paragraph("• Ingesta ERA5-Land: 99/99 archivos NetCDF verificados contra <code>manifest.json</code> (100% SHA-256).<br/>"
                      "• Suite de Pruebas: <code>pytest</code> 26 passed, 0 failed en <code>tests/test_phase2a.py</code>.<br/>"
                      "• Linter & Tipado Estricto: <code>ruff</code> 100% limpio | <code>mypy --strict</code> 20 archivos validados.<br/>"
                      "• Persistencia de Series: Zarr y Parquet utilizados exclusivamente (cero CSV intermediarios).", table_cell),
            Paragraph("<b>FASE 2A COMPLETADA Y CERRADA</b><br/>"
                      "Sin datos sintéticos en tablas finales.<br/>"
                      "Sin fuga de información temporal.<br/>"
                      "Target primario CONAF validado.", table_cell),
        ],
    ]
    t_qa = Table(qa_box, colWidths=[330, 174])
    t_qa.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c_light_bg),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#CBD5E1")),
        ("LINELEFT", (0, 0), (0, -1), 3.0, c_primary),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_qa)

    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path


if __name__ == "__main__":
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    pdf_path = out_dir / "Reporte_Tecnico_ChileFuego_Fase2A.pdf"
    build_pdf_report(pdf_path, fig_dir)
    print(f"PDF generado exitosamente en: {pdf_path.resolve()}")
    print(f"Tamaño del archivo: {pdf_path.stat().st_size:,} bytes")
