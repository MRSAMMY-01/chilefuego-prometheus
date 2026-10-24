"""Inspección de fecha de corte de valores nulos en variables previous_day1 (Adenda H4).

Lee los 236 archivos JSON originales en data/raw/open_meteo/ y determina la fecha
exacta donde desaparecen los valores nulos, junto con el desglose mensual de nulos.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    raw_dir = Path("data/raw/open_meteo")
    files = sorted(raw_dir.glob("batch_2023_24_*.json"))
    print(f"Total archivos inspeccionados: {len(files)}")

    vars_to_check = [
        "temperature_2m_previous_day1",
        "relative_humidity_2m_previous_day1",
        "wind_speed_10m_previous_day1",
        "precipitation_previous_day1",
    ]

    for var in vars_to_check:
        month_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "nulls": 0})
        first_null: str | None = None
        last_null: str | None = None
        first_valid: str | None = None

        for f in files:
            try:
                with open(f, encoding="utf-8") as jf:
                    c = json.load(jf)
                data = c.get("data", [])
                for node in data:
                    hourly = node.get("hourly", {})
                    times = hourly.get("time", [])
                    vals = hourly.get(var, [])
                    if not vals:
                        continue
                    for t_str, val in zip(times, vals, strict=True):
                        m_str = t_str[:7]
                        month_counts[m_str]["total"] += 1
                        if val is None:
                            month_counts[m_str]["nulls"] += 1
                            if first_null is None or t_str < first_null:
                                first_null = t_str
                            if last_null is None or t_str > last_null:
                                last_null = t_str
                        else:
                            if first_valid is None or t_str < first_valid:
                                first_valid = t_str
            except Exception as exc:
                print(f"Error procesando {f.name}: {exc}")

        print("\n=======================================================")
        print(f"VARIABLE: {var}")
        print("=======================================================")
        print(f"Primer nulo registrado:      {first_null}")
        print(f"ÚLTIMO NULO REGISTRADO:      {last_null}")
        print(f"Primer registro válido:      {first_valid}")
        print("\nDesglose mensual de nulos:")
        for m in sorted(month_counts.keys()):
            tot = month_counts[m]["total"]
            nul = month_counts[m]["nulls"]
            pct = (nul / tot * 100.0) if tot else 0.0
            print(f"  * {m}: {nul:7d} / {tot:7d} ({pct:6.2f}% nulos)")


if __name__ == "__main__":
    main()
