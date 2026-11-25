#!/usr/bin/env python
"""
list_oatrack_names.py
=====================
Writes the full list of the files and variables of OA-Track from oatrack_names.py:
    OA-Track_variables.pdf    every file and every variable with dimensions and units; long name and description for
                              the files of Omega_arag; the uncompressed size of every file
    OA-Track_variables.xlsx   the same as a table, with every long name and description
Usage:  python list_oatrack_names.py --out <folder>
"""
import argparse
import datetime
import html
from pathlib import Path

import oatrack_names as N


def gb(b):
    return f'{b / 1e9:.1f} GB' if b >= 1e8 else f'{b / 1e6:.2f} MB'


def xlsx(path, files):
    from openpyxl import Workbook
    wb = Workbook(write_only=True)
    ws = wb.create_sheet('files')
    ws.append(['file', 'content', 'variables', 'size uncompressed (GB)'])
    for F in files:
        ws.append([F.name, F.title, len(F.rows), round(F.size() / 1e9, 3)])
    ws = wb.create_sheet('variables')
    ws.append(['file', 'variable', 'dimensions', 'units', 'long_name', 'description'])
    for F in files:
        for r in F.rows:
            ws.append([F.name, r['name'], r['dims'], r['units'], r['long_name'], r['description']])
    ws = wb.create_sheet('csv')
    ws.append(['file', 'columns'])
    for F in files:
        if '_global_mean' in F.name:
            ws.append([F.name.replace('.nc', '.csv'), ', '.join(N.csv_columns(F))])
    wb.save(path)


CSS = """
@page { size: A4 landscape; margin: 11mm 10mm 12mm 10mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 8pt; color: #111; }
h1 { font-size: 15pt; margin: 0 0 3mm 0; } h2 { font-size: 12pt; margin: 5mm 0 2mm 0; page-break-after: avoid; }
h3 { font-size: 9.5pt; margin: 4mm 0 1.5mm 0; page-break-after: avoid; }
p, li { font-size: 8.5pt; line-height: 1.35; } ul { margin: 1mm 0 2mm 5mm; padding-left: 3mm; }
table { border-collapse: collapse; width: 100%; margin: 1mm 0 3mm 0; }
th { background: #e8ecf1; text-align: left; font-size: 7.5pt; padding: 1.2mm 1.5mm; border-bottom: 0.4mm solid #889; }
td { font-size: 7.2pt; padding: 0.7mm 1.5mm; border-bottom: 0.15mm solid #ccd; vertical-align: top; }
tr { page-break-inside: avoid; } thead { display: table-header-group; }
td.n { font-family: 'DejaVu Sans Mono', monospace; font-size: 7pt; white-space: nowrap; }
td.d { white-space: nowrap; } .small { color: #555; } .csv { font-family: 'DejaVu Sans Mono', monospace; font-size: 7pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.6pt; }
.break { page-break-before: always; }
"""


def table(rows, full):
    """full: long name and description of every variable (as in the file); otherwise name, dimensions and units"""
    cols = (('Variable', '33%'), ('Dimensionen', '22%'), ('Einheit', '6%')) + (
        (('long_name und Beschreibung in der Datei', '39%'),) if full else ())
    head = ''.join(f'<th style="width:{w}">{c}</th>' for c, w in cols)
    body = ''.join(f'<tr><td class="n">{html.escape(r["name"])}</td><td class="d">{html.escape(r["dims"])}</td>'
                   f'<td class="d">{html.escape(r["units"])}</td>' + (
                       f'<td><b>{html.escape(r["long_name"])}</b>. {html.escape(r["description"])}</td>' if full else '')
                   + '</tr>' for r in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def compact(rows):
    """name, dimensions and units in two columns of pairs"""
    half = (len(rows) + 1) // 2
    cell = lambda r: (f'<td class="n">{html.escape(r["name"])}</td><td class="d">{html.escape(r["dims"])}</td>'
                      f'<td class="d">{html.escape(r["units"])}</td>') if r else '<td></td><td></td><td></td>'
    head = ''.join(f'<th style="width:{w}">{c}</th>' for c, w in (('Variable', '27%'), ('Dimensionen', '18%'),
                                                                   ('Einheit', '5%')) * 2)
    body = ''.join(f'<tr>{cell(rows[i])}{cell(rows[i + half] if i + half < len(rows) else None)}</tr>' for i in range(half))
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def pdf(path, files):
    n = sum(len(F.rows) for F in files)
    today = datetime.date.today().strftime('%d. %m. %Y')
    tot = lambda fs: gb(sum(F.size() for F in fs))
    H = [f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>']
    H.append(f"""<h1>OA-Track: alle Dateien und Variablen</h1>
<p class="small">Entwurf zur Durchsicht, {today}, aus oatrack_names.py (dieselbe Definition schreibt die Dateien)</p>
<h2>Aufbau</h2>
<p>Pro Größe 3 netCDF-Dateien und eine CSV, für 15 Größen {len(files)} Dateien mit {n:,} Variablen (Koordinaten
mitgezählt). Unkomprimiert zusammen {tot(files)}; komprimiert deutlich weniger (Landzellen, unbelegte Jahre, Rundung auf
5 Stellen), die genaue Größe nach dem ersten Lauf.</p>
<ul>
<li><code>OA-Track_&lt;Größe&gt;_global_mean.nc</code> und <code>.csv</code>: Globalmittel 1750–2100, vorindustrielle
Zustände, beobachtungsbasierte Produkte und ESMs, je Produkt und Mittel, auf drei Flächen, mit Änderung seit 1750 und 1850.</li>
<li><code>OA-Track_&lt;Größe&gt;_maps_obs_based_products.nc</code>: Karten der vorindustriellen Zustände und der beobachtungsbasierten Produkte
1980–2025, je Produkt und Mittel.</li>
<li><code>OA-Track_&lt;Größe&gt;_maps_esms.nc</code>: Karten der ESMs 1750–2100 (historical und vier SSPs), Mittel der
Produkte.</li>
</ul>
<p>Ein Name ist aufgebaut als <code>&lt;Größe&gt;_&lt;Block&gt;[_gm][_change][_&lt;Term&gt;]</code>; derselbe Name bedeutet
in jeder Datei dasselbe. Ohne <code>_gm</code> ist es eine Karte.</p>
<ul>
<li><b>Dimensionen:</b> <code>area</code> bei den Globalmitteln (common_area, product_area, whole_ocean; whole_ocean nur
Ωarag; product_area leer für das Mittel, whole_ocean leer für die Jahreswerte), <code>product</code> (mean, OceanSODA,
UExP, JMA, CMEMS), <code>scenario</code> (historical: alle 18 ESMs 1750–2014; ssp126 bis ssp585: 1750–2100),
<code>state</code> (1750, 1850), <code>year</code>. In Python etwa
<code>.sel(area='whole_ocean', product='mean', scenario='ssp245')</code>.</li>
<li><b>Blöcke:</b> <code>preindustrial</code>, <code>obs_annual</code> (1980–2025), <code>obs_annual_forced</code>
(1980–2024), <code>esms_annual_forced</code>.</li>
<li><b><code>_change</code>:</b> Änderung ab dem vorindustriellen Zustand desselben Produkts und derselben Fläche, relativ
in %, absolut für pH, Temperatur und Salinität; beobachtungsbasierte Produkte mit <code>_sigma</code>, ESMs mit
<code>_sigma_upper_limit</code>.</li>
<li><b><code>_sigma</code> enthält immer alles.</b></li>
<li><b><code>_alkalinity_adjustment</code></b> (10 Größen, nicht pCO2, fCO2, CO2*, T, S): ohne Korrektur Wert minus a;
1 sigma ohne Korrektur global √(σ² − a²), in den Karten √(σ² − <code>_sigma_alkalinity</code>²).</li>
<li><b>Masken:</b> <code>mask_common_area</code>, <code>mask_product_area</code> (je Produkt), <code>filled_flag</code>
(nur das Mittel ist aufgefüllt), <code>constrained_flag</code>, bei 11 Größen <code>sigma_flag</code>.</li>
<li><b>CSV:</b> langes Format, eine Zeile pro Wert.</li>
</ul>
<p>Ωarag steht mit long_name und Beschreibung jeder Variable; die anderen Größen mit Name, Dimensionen und Einheit (alles
in der Excel-Datei).</p>""")
    H.append('<h2 class="break">Alle Dateien</h2><table><thead><tr><th style="width:40%">Datei</th><th style="width:40%">'
             'Inhalt</th><th style="width:8%">Variablen</th><th style="width:12%">unkomprimiert</th></tr></thead><tbody>'
             + ''.join(f'<tr><td class="n">{F.name}</td><td>{html.escape(F.title)}</td><td>{len(F.rows)}</td>'
                       f'<td>{gb(F.size())}</td></tr>' for F in files) + '</tbody></table>')
    q0 = None
    for F in files:
        first = F.q == N.QUANTITIES[0]
        if F.q != q0:
            H.append(f'<h2 class="break">{html.escape(N.LONG[F.q])} ({F.q})</h2>')
            q0 = F.q
        H.append(f'<h3>{F.name} ({len(F.rows)} Variablen, {gb(F.size())} unkomprimiert)</h3>')
        H.append(table(F.rows, True) if first else compact(F.rows))
        if '_global_mean' in F.name:
            H.append(f'<p><b>{F.name.replace(".nc", ".csv")}</b>, Spalten: <span class="csv">'
                     f'{", ".join(N.csv_columns(F))}</span></p>')
    H.append('</body></html>')
    tmp = path.with_suffix('.html')
    tmp.write_text(''.join(H))
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.resolve().as_uri())
        pg.pdf(path=str(path), format='A4', landscape=True, print_background=True,
               margin=dict(top='11mm', bottom='12mm', left='10mm', right='10mm'), display_header_footer=True,
               header_template='<span></span>',
               footer_template='<div style="font-size:7pt;width:100%;text-align:center;color:#666">'
                               '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        b.close()
    tmp.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = N.files()
    xlsx(out / 'OA-Track_variables.xlsx', files)
    pdf(out / 'OA-Track_variables.pdf', files)
    print(f'{len(files)} files, {sum(len(F.rows) for F in files)} variables, uncompressed '
          f'{sum(F.size() for F in files) / 1e9:.0f} GB -> {out}')


if __name__ == '__main__':
    main()
