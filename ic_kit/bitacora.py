"""Bitacora de la jornada: lo que despues se califica en el probatorio.

El reglamento da 30 minutos para el probatorio y lo que se evalua ahi no son
las metricas —esas las genera el kit solo— sino **las hipotesis, los hallazgos
con su evidencia, las decisiones y lo que se descarto y por que**.

Eso no se reconstruye a las 09:30 del viernes. Se anota mientras pasa, en una
linea. `probatorio.generar_notebook` lee este archivo y arma las secciones.

    from ic_kit.bitacora import Bitacora
    b = Bitacora()
    b.hipotesis("El lactato y la saturacion concentran la senal de gravedad")
    b.hallazgo("Fuga en codigo_interno", "AUC de la variable aislada = 1.00",
               "descartada: su distribucion en test no coincide")
    b.decision("Se uso decision de minimo costo esperado en vez de argmax",
               "costo OOF 0.747 -> 0.588")
    b.descarte("Pseudo-etiquetado sobre train_unlabeled", "no bajo el costo OOF")
    b.mostrar()
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

TIPOS = ("hipotesis", "hallazgo", "decision", "descarte", "nota")


class Bitacora:
    """Registro append-only, tolerante a que la sesion se reinicie."""

    def __init__(self, ruta="work/bitacora.json", verbose: bool = True):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.entradas = self._leer()

    def _leer(self) -> list:
        if not self.ruta.exists():
            return []
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return json.loads(self.ruta.read_text(encoding=enc))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        return []

    def _agregar(self, tipo: str, texto: str, evidencia="", decision=""):
        e = {"tipo": tipo, "hora": datetime.now().strftime("%H:%M"),
             "texto": texto, "evidencia": evidencia, "decision": decision}
        self.entradas.append(e)
        self.ruta.write_text(
            json.dumps(self.entradas, indent=2, ensure_ascii=False),
            encoding="utf-8")
        if self.verbose:
            extra = (" | %s" % evidencia) if evidencia else ""
            print("[%s] %-9s %s%s" % (e["hora"], tipo, texto, extra))
        return e

    # ------------------------------------------------------------ atajos
    def hipotesis(self, texto: str):
        """Algo que creemos antes de comprobarlo. Anotarla ANTES vale doble:
        si despues se cae, eso mismo es un hallazgo para el informe."""
        return self._agregar("hipotesis", texto)

    def hallazgo(self, texto: str, evidencia: str = "", decision: str = ""):
        """Algo que encontramos en los datos, con el numero que lo respalda."""
        return self._agregar("hallazgo", texto, evidencia, decision)

    def decision(self, texto: str, evidencia: str = ""):
        """Algo que elegimos hacer, y por que."""
        return self._agregar("decision", texto, evidencia)

    def descarte(self, texto: str, motivo: str = ""):
        """Algo que probamos y NO usamos. Es la seccion que mas suma y la que
        todos olvidan: muestra que el recorrido fue exploratorio."""
        return self._agregar("descarte", texto, motivo)

    def nota(self, texto: str):
        return self._agregar("nota", texto)

    # ------------------------------------------------------------ lectura
    @staticmethod
    def _con_motivo(e: dict) -> str:
        motivo = e.get("decision") or e.get("evidencia") or ""
        return e["texto"] + ((": %s" % motivo) if motivo else "")

    def por_tipo(self, tipo: str) -> list:
        return [e for e in self.entradas if e["tipo"] == tipo]

    def para_probatorio(self) -> dict:
        """Devuelve los argumentos que espera `probatorio.generar_notebook`."""
        return {
            "hipotesis": [e["texto"] for e in self.por_tipo("hipotesis")],
            "trampas": [{"hallazgo": e["texto"], "evidencia": e["evidencia"],
                         "decision": e["decision"]}
                        for e in self.por_tipo("hallazgo")],
            "decisiones": [e["texto"] + ((" (%s)" % e["evidencia"]) if e["evidencia"] else "")
                           for e in self.por_tipo("decision")],
            "descartado": [self._con_motivo(e) for e in self.por_tipo("descarte")],
        }

    def mostrar(self):
        if not self.entradas:
            print("bitacora vacia")
            return
        print("%-6s %-10s %s" % ("hora", "tipo", "texto"))
        for e in self.entradas:
            print("%-6s %-10s %s" % (e["hora"], e["tipo"], e["texto"]))
            if e["evidencia"]:
                print("%-6s %-10s   evidencia: %s" % ("", "", e["evidencia"]))
            if e["decision"]:
                print("%-6s %-10s   decision:  %s" % ("", "", e["decision"]))
        faltan = [t for t in ("hipotesis", "hallazgo", "decision", "descarte")
                  if not self.por_tipo(t)]
        if faltan:
            print("\nSin entradas de: %s" % ", ".join(faltan))
            print("Esas secciones van a quedar vacias en el probatorio.")
