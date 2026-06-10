"""
Predictor de Estadisticas para el Mundial
==========================================
App de consola que estima, para un partido entre dos selecciones,
basandose en estadisticas historicas/actuales de cada equipo:

- Favorito y probabilidades de ganar / empate / perder
- Resultado mas probable (marcador)
- Posibles goleadores
- Tarjetas (amarillas/rojas) esperadas
- Goles totales esperados (over/under)
- Tiros de esquina esperados

No es una garantia de resultados reales, es solo una referencia
estadistica para apoyar decisiones de apuestas.
"""

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Datos base de los equipos (puedes ajustarlos/ampliarlos como quieras)
# ---------------------------------------------------------------------------

@dataclass
class Equipo:
    nombre: str
    ranking_fifa: int          # menor = mejor
    ataque: float              # goles promedio anotados por partido
    defensa: float             # goles promedio recibidos por partido
    posesion: float            # % posesion promedio
    tarjetas_amarillas: float  # promedio por partido
    tarjetas_rojas: float      # probabilidad de roja en el partido (0-1)
    corners: float             # corners promedio por partido
    goleadores: list = field(default_factory=list)  # (nombre, goles_torneo, prob_gol)


EQUIPOS = {
    "Argentina": Equipo("Argentina", 1, 1.9, 0.7, 56, 1.6, 0.06, 5.8,
                         [("L. Messi", 7, 0.42), ("J. Alvarez", 4, 0.30), ("A. Di Maria", 2, 0.18)]),
    "Francia": Equipo("Francia", 2, 2.1, 0.9, 54, 1.4, 0.05, 5.2,
                       [("K. Mbappe", 8, 0.48), ("O. Giroud", 4, 0.25), ("A. Griezmann", 3, 0.20)]),
    "Brasil": Equipo("Brasil", 3, 1.8, 0.8, 58, 1.7, 0.07, 5.5,
                      [("Vinicius Jr", 5, 0.35), ("Rodrygo", 3, 0.22), ("Raphinha", 2, 0.18)]),
    "España": Equipo("España", 4, 1.7, 0.6, 62, 1.5, 0.04, 6.1,
                      [("A. Morata", 3, 0.25), ("F. Torres", 4, 0.30), ("Pedri", 2, 0.15)]),
    "Inglaterra": Equipo("Inglaterra", 5, 1.6, 0.7, 55, 1.6, 0.05, 5.0,
                          [("H. Kane", 6, 0.40), ("B. Saka", 3, 0.22), ("P. Foden", 2, 0.18)]),
    "Alemania": Equipo("Alemania", 6, 1.7, 0.9, 57, 1.4, 0.04, 5.4,
                        [("K. Havertz", 3, 0.22), ("J. Musiala", 3, 0.25), ("N. Fullkrug", 3, 0.25)]),
    "Portugal": Equipo("Portugal", 7, 1.6, 0.8, 53, 1.5, 0.05, 5.0,
                        [("C. Ronaldo", 5, 0.35), ("B. Fernandes", 3, 0.20), ("R. Leao", 2, 0.18)]),
    "Mexico": Equipo("Mexico", 14, 1.2, 1.1, 50, 2.0, 0.06, 4.5,
                      [("S. Gimenez", 3, 0.28), ("H. Lozano", 2, 0.20)]),
    "Estados Unidos": Equipo("Estados Unidos", 11, 1.3, 1.0, 52, 1.8, 0.05, 4.6,
                              [("C. Pulisic", 3, 0.28), ("F. Balogun", 2, 0.20)]),
    "Croacia": Equipo("Croacia", 8, 1.4, 0.9, 56, 1.6, 0.04, 4.8,
                       [("L. Modric", 2, 0.15), ("A. Kramaric", 3, 0.25)]),
    "Marruecos": Equipo("Marruecos", 12, 1.3, 0.7, 48, 1.9, 0.06, 4.2,
                         [("Hakimi", 1, 0.10), ("En-Nesyri", 3, 0.25)]),
    "Japon": Equipo("Japon", 17, 1.3, 1.0, 53, 1.5, 0.03, 4.4,
                     [("T. Kubo", 2, 0.20), ("D. Asano", 2, 0.20)]),
}


# ---------------------------------------------------------------------------
# Modelo de prediccion (Poisson + ajuste por ranking)
# ---------------------------------------------------------------------------

def factor_local(es_local: bool) -> float:
    """Pequeña ventaja para el equipo 'local' (sede del partido)."""
    return 1.08 if es_local else 1.0


def goles_esperados(equipo: Equipo, rival: Equipo, es_local: bool) -> float:
    """
    Combina el ataque del equipo con la debilidad defensiva del rival,
    aplica un ajuste leve por diferencia de ranking FIFA.
    """
    base = (equipo.ataque + rival.defensa) / 2
    # ajuste por ranking: equipos mejor rankeados rinden un poco mas
    diff_ranking = (rival.ranking_fifa - equipo.ranking_fifa) / 50
    ajuste = 1 + max(-0.25, min(0.25, diff_ranking * 0.05))
    return max(0.2, base * ajuste * factor_local(es_local))


def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def matriz_marcadores(lam_local: float, lam_visita: float, max_goles: int = 6):
    matriz = {}
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            matriz[(i, j)] = poisson_pmf(i, lam_local) * poisson_pmf(j, lam_visita)
    return matriz


def probabilidades_resultado(matriz):
    p_local, p_empate, p_visita = 0.0, 0.0, 0.0
    for (i, j), p in matriz.items():
        if i > j:
            p_local += p
        elif i == j:
            p_empate += p
        else:
            p_visita += p
    return p_local, p_empate, p_visita


def marcador_mas_probable(matriz):
    return max(matriz.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Predicciones complementarias
# ---------------------------------------------------------------------------

def prediccion_goleadores(equipo: Equipo, lam_goles: float):
    """Ajusta la probabilidad de cada jugador segun los goles esperados del equipo."""
    resultado = []
    for nombre, goles_torneo, prob_base in equipo.goleadores:
        prob_ajustada = min(0.85, prob_base * (lam_goles / equipo.ataque))
        resultado.append((nombre, round(prob_ajustada * 100, 1)))
    return sorted(resultado, key=lambda x: -x[1])


def prediccion_tarjetas(equipo_a: Equipo, equipo_b: Equipo):
    amarillas_total = equipo_a.tarjetas_amarillas + equipo_b.tarjetas_amarillas
    prob_roja = 1 - (1 - equipo_a.tarjetas_rojas) * (1 - equipo_b.tarjetas_rojas)
    return round(amarillas_total, 1), round(prob_roja * 100, 1)


def prediccion_corners(equipo_a: Equipo, equipo_b: Equipo):
    total = equipo_a.corners + equipo_b.corners
    return round(total, 1)


def prediccion_over_under(lam_local: float, lam_visita: float, linea: float = 2.5):
    total_esperado = lam_local + lam_visita
    # Probabilidad de over usando distribucion de Poisson para la suma
    lam_total = total_esperado
    prob_under = sum(
        poisson_pmf(k, lam_total) for k in range(int(math.floor(linea)) + 1)
    )
    prob_over = 1 - prob_under
    return total_esperado, round(prob_over * 100, 1), round(prob_under * 100, 1)


# ---------------------------------------------------------------------------
# Reporte principal
# ---------------------------------------------------------------------------

def analizar_partido(nombre_local: str, nombre_visita: str, sede_neutral: bool = True):
    if nombre_local not in EQUIPOS or nombre_visita not in EQUIPOS:
        disponibles = ", ".join(EQUIPOS.keys())
        print(f"Equipo no encontrado. Equipos disponibles:\n{disponibles}")
        return

    local = EQUIPOS[nombre_local]
    visita = EQUIPOS[nombre_visita]
    es_local = not sede_neutral

    lam_local = goles_esperados(local, visita, es_local)
    lam_visita = goles_esperados(visita, local, False)

    matriz = matriz_marcadores(lam_local, lam_visita)
    p_local, p_empate, p_visita = probabilidades_resultado(matriz)
    marcador = marcador_mas_probable(matriz)

    print("=" * 60)
    print(f"  PARTIDO: {local.nombre} vs {visita.nombre}")
    print("=" * 60)

    # --- Favorito ---
    print("\n--- FAVORITO Y RESULTADO ---")
    if p_local > p_visita and p_local > p_empate:
        favorito = local.nombre
    elif p_visita > p_local and p_visita > p_empate:
        favorito = visita.nombre
    else:
        favorito = "Empate (ningun favorito claro)"
    print(f"Favorito: {favorito}")
    print(f"Probabilidad {local.nombre} gane:  {p_local*100:5.1f}%")
    print(f"Probabilidad de Empate:        {p_empate*100:5.1f}%")
    print(f"Probabilidad {visita.nombre} gane: {p_visita*100:5.1f}%")
    print(f"Marcador mas probable: {local.nombre} {marcador[0]} - {marcador[1]} {visita.nombre}")

    # --- Goles esperados / over-under ---
    total_esp, prob_over, prob_under = prediccion_over_under(lam_local, lam_visita, 2.5)
    print("\n--- GOLES EN EL PARTIDO ---")
    print(f"Goles esperados {local.nombre}:  {lam_local:.2f}")
    print(f"Goles esperados {visita.nombre}: {lam_visita:.2f}")
    print(f"Total de goles esperados:       {total_esp:.2f}")
    print(f"Probabilidad Over 2.5 goles:    {prob_over}%")
    print(f"Probabilidad Under 2.5 goles:   {prob_under}%")

    # --- Goleadores ---
    print("\n--- POSIBLES GOLEADORES ---")
    print(f"{local.nombre}:")
    for nombre, prob in prediccion_goleadores(local, lam_local):
        print(f"  - {nombre:<15} {prob:5.1f}% prob. de marcar")
    print(f"{visita.nombre}:")
    for nombre, prob in prediccion_goleadores(visita, lam_visita):
        print(f"  - {nombre:<15} {prob:5.1f}% prob. de marcar")

    # --- Tarjetas ---
    amarillas, prob_roja = prediccion_tarjetas(local, visita)
    print("\n--- TARJETAS ---")
    print(f"Tarjetas amarillas esperadas (total): {amarillas}")
    print(f"Probabilidad de al menos una roja:    {prob_roja}%")

    # --- Corners ---
    corners = prediccion_corners(local, visita)
    print("\n--- TIROS DE ESQUINA (CORNERS) ---")
    print(f"Corners totales esperados: {corners}")
    print(f"  Linea sugerida Over/Under: {round(corners - 0.5, 1)}")

    print("\n" + "=" * 60)
    print("Nota: Estimaciones basadas en estadisticas, solo de referencia.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Menu interactivo
# ---------------------------------------------------------------------------

def menu():
    print("\nPREDICTOR DE ESTADISTICAS - MUNDIAL")
    print("Equipos disponibles:")
    for nombre in EQUIPOS:
        print(f"  - {nombre}")

    while True:
        print("\nEscribe 'salir' para terminar.")
        local = input("\nEquipo Local: ").strip()
        if local.lower() == "salir":
            break
        visita = input("Equipo Visitante: ").strip()
        if visita.lower() == "salir":
            break

        if local == visita:
            print("Los equipos deben ser diferentes.")
            continue

        sede = input("¿Es cancha neutral? (s/n, default s): ").strip().lower()
        sede_neutral = sede != "n"

        analizar_partido(local, visita, sede_neutral)


if __name__ == "__main__":
    menu()
