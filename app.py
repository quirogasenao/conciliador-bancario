import pandas as pd
import streamlit as st
from datetime import timedelta

# ==============================================
# Utilidades de normalización
# ==============================================

def cargar_extracto(file) -> pd.DataFrame:
    """
    Carga y normaliza el extracto bancario a columnas:
    Fecha, Importe, Concepto, AbsImporte, RowID
    """
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Normalizamos nombres de columnas
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    cols = set(df.columns)

    # Detectar fecha
    col_fecha = None
    for c in ["fecha_mov", "fecha", "date"]:
        if c in cols:
            col_fecha = c
            break
    if col_fecha is None:
        raise ValueError(
            f"No se encontró columna de fecha en el extracto. Columnas: {df.columns.tolist()}"
        )

    # Detectar importe
    col_importe = None
    for c in ["importe", "importe_mov", "amount", "importe_fac"]:
        if c in cols:
            col_importe = c
            break
    if col_importe is None:
        raise ValueError(
            f"No se encontró columna de importe en el extracto. Columnas: {df.columns.tolist()}"
        )

    # Detectar concepto
    col_concepto = None
    for c in ["concepto_raw", "concepto", "descripcion", "descripción"]:
        if c in cols:
            col_concepto = c
            break
    if col_concepto is None:
        df["concepto"] = ""
        col_concepto = "concepto"

    # Normalizar campos
    df["Fecha"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)
    df["Importe"] = (
        df[col_importe]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        .astype(float)
    )
    df["Concepto"] = df[col_concepto].astype(str).str.strip()
    df["AbsImporte"] = df["Importe"].abs().round(2)

    # Añadimos RowID para control interno
    df = df.reset_index(drop=True)
    df["RowID"] = df.index + 1

    return df[["RowID", "Fecha", "Concepto", "Importe", "AbsImporte"]]


def cargar_facturas(file) -> pd.DataFrame:
    """
    Carga y normaliza el listado de facturas a columnas:
    Fecha, Importe, Proveedor, NumFactura, fact_id

    Adaptado a formato típico:
    - fecha_fac / fecha
    - importe_fac / importe / total
    - proveedor
    - num_fac / num_factura / numero_factura
    """
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Normalizamos nombres de columnas
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    cols = set(df.columns)

    # Fecha
    col_fecha = None
    for c in ["fecha_fac", "fecha", "fecha_factura"]:
        if c in cols:
            col_fecha = c
            break
    if col_fecha is None:
        df["fecha"] = pd.NaT
        col_fecha = "fecha"

    # Importe
    col_importe = None
    for c in ["importe_fac", "importe", "total", "total_factura"]:
        if c in cols:
            col_importe = c
            break
    if col_importe is None:
        raise ValueError(
            f"No se encontró columna de importe en facturas. Columnas: {df.columns.tolist()}"
        )

    # Proveedor
    col_proveedor = None
    for c in ["proveedor", "cliente", "nombre"]:
        if c in cols:
            col_proveedor = c
            break
    if col_proveedor is None:
        df["proveedor"] = "(desconocido)"
        col_proveedor = "proveedor"

    # Número de factura
    col_num = None
    for c in ["num_fac", "num_factura", "numero_factura", "factura", "num", "numero"]:
        if c in cols:
            col_num = c
            break
    if col_num is None:
        df["num_fac"] = df.index.astype(str)
        col_num = "num_fac"

    # Normalizar
    df["Fecha"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)
    df["Importe"] = (
        df[col_importe]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        .astype(float)
        .abs()
        .round(2)
    )
    df["Proveedor"] = df[col_proveedor].astype(str).str.strip()
    df["NumFactura"] = df[col_num].astype(str).str.strip()

    # fact_id único
    df["fact_id"] = (
        df["Fecha"].fillna(pd.Timestamp("1900-01-01")).astype(str)
        + "|" + df["Proveedor"]
        + "|" + df["NumFactura"]
        + "|" + df["Importe"].round(2).astype(str)
    )

    df = df.reset_index(drop=True)
    return df[["Fecha", "Importe", "Proveedor", "NumFactura", "fact_id"]]


# ==============================================
# Clasificación por reglas (sin IA)
# ==============================================

def clasificar_por_reglas(concepto: str) -> str:
    """
    Clasificación muy básica por reglas sobre el texto del concepto.
    Devuelve:
      - "comision_bancaria"
      - "factura_proveedor"
      - "impuesto_o_tasa"
      - "nomina_o_seg_social"
      - "otro"
    """
    if not isinstance(concepto, str):
        concepto = str(concepto) if concepto is not None else ""
    c = concepto.upper()

    # Comisiones y gastos bancarios típicos
    if any(
        palabra in c
        for palabra in [
            "COMISION",
            "COMISIÓN",
            "GASTOS",
            "GASTO",
            "INTERES",
            "INTERÉS",
            "MANTENIMIENTO",
            "TPV",
            "LIQUIDACION",
            "LIQUIDACIÓN",
            "CUOTA",
            "SERVICIO CORRESPONDENCIA",
            "COMISIONES",
        ]
    ):
        return "comision_bancaria"

    # Impuestos, tasas, AEAT, etc.
    if any(
        palabra in c
        for palabra in [
            "AEAT",
            "AGENCIA TRIBUTARIA",
            "HACIENDA",
            "IMPUESTO",
            "IVA",
            "ITP",
            "TASA",
        ]
    ):
        return "impuesto_o_tasa"

    # Nóminas, seguros sociales
    if any(
        palabra in c
        for palabra in [
            "SEGURIDAD SOCIAL",
            "SS",
            "NOMINA",
            "NÓMINA",
            "SALARIO",
            "SUELDO",
        ]
    ):
        return "nomina_o_seg_social"

    # Proveedores típicos (ejemplo, ajusta a tu realidad)
    if any(
        palabra in c
        for palabra in [
            "ENDESA",
            "IBERDROLA",
            "VODAFONE",
            "MOVISTAR",
            "ORANGE",
            "AGUA",
            "CANON",
            "ALQUILER",
            "RESTAURANTE",
            "BAR ",
            "CAFETERIA",
            "CAFETERÍA",
            "AMAZON",
            "CORREOS",
        ]
    ):
        return "factura_proveedor"

    # Por defecto
    return "otro"


# ==============================================
# Conciliación simple 1:1
# ==============================================

def conciliar_1a1(
    bank_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    dias_ventana: int = 5,
    tolerancia_cent: float = 0.0,
):
    """
    Conciliación básica 1:1:
    - Solo considera movimientos de gasto (Importe < 0)
    - Match por importe (AbsImporte ≈ Importe factura) + ventana de fechas
    """
    bank = bank_df.copy()
    inv = inv_df.copy()

    # Estructuras
    if "Conciliada" not in bank.columns:
        bank["Conciliada"] = False
    else:
        bank["Conciliada"] = False

    bank["FacturaID"] = None
    bank["Proveedor"] = bank.get("Proveedor", None)
    bank["NumFactura"] = bank.get("NumFactura", None)

    inv["Usada"] = False

    gastos = bank[bank["Importe"] < 0].copy()
    if gastos.empty or inv.empty:
        pendientes = gastos.copy()
        facturas_sin_usar = inv[~inv["Usada"]].copy()
        return bank, inv, gastos, pendientes, facturas_sin_usar

    used_facts = set()

    for idx, mov in gastos.iterrows():
        if mov["Conciliada"]:
            continue

        fecha_mov = mov["Fecha"]
        target = round(mov["AbsImporte"], 2)

        # Ventana de fechas
        if pd.isna(fecha_mov):
            rango = inv
        else:
            fecha_min = fecha_mov - timedelta(days=dias_ventana)
            fecha_max = fecha_mov + timedelta(days=dias_ventana)
            rango = inv[(inv["Fecha"] >= fecha_min) & (inv["Fecha"] <= fecha_max)]

        # No usar facturas ya usadas
        rango = rango[~rango["fact_id"].isin(used_facts)]

        # Condición de importe (tolerancia en céntimos)
        def match_importe(x):
            return abs(x - target) <= tolerancia_cent

        candidatos = rango[rango["Importe"].apply(match_importe)]

        if len(candidatos) == 1:
            fac = candidatos.iloc[0]

            # Marcamos factura usada
            used_facts.add(fac["fact_id"])
            inv.loc[inv["fact_id"] == fac["fact_id"], "Usada"] = True

            # Marcamos movimiento conciliado
            bank.loc[bank["RowID"] == mov["RowID"], "Conciliada"] = True
            bank.loc[bank["RowID"] == mov["RowID"], "FacturaID"] = fac["fact_id"]
            bank.loc[bank["RowID"] == mov["RowID"], "Proveedor"] = fac["Proveedor"]
            bank.loc[bank["RowID"] == mov["RowID"], "NumFactura"] = fac["NumFactura"]

    # Pendientes: gastos que no se han conciliado
    pendientes = bank[(bank["Importe"] < 0) & (~bank["Conciliada"])].copy()

    # Facturas sin usar
    facturas_sin_usar = inv[~inv["Usada"]].copy()

    return bank, inv, gastos, pendientes, facturas_sin_usar


# ==============================================
# APP STREAMLIT
# ==============================================

def main():
    st.set_page_config(page_title="Conciliador bancario limpio", layout="wide")
    st.title("🧾 Conciliador bancario (versión sencilla + clasificación por reglas)")

    # ----- Sidebar -----
    st.sidebar.header("1. Subir ficheros")
    file_ext = st.sidebar.file_uploader(
        "Extracto bancario (CSV/Excel)", type=["csv", "xlsx", "xls"]
    )
    file_inv = st.sidebar.file_uploader(
        "Listado de facturas (CSV/Excel)", type=["csv", "xlsx", "xls"]
    )

    st.sidebar.header("2. Parámetros de conciliación")
    dias_ventana = st.sidebar.slider(
        "Ventana de fechas para match (± días)", min_value=0, max_value=60, value=5, step=1
    )
    tolerancia_cent = st.sidebar.select_slider(
        "Tolerancia importe (céntimos)",
        options=[0.00, 0.01, 0.02, 0.05],
        value=0.00,
    )

    if not file_ext or not file_inv:
        st.info("Sube el extracto bancario y el fichero de facturas para empezar.")
        return

    # ----- Carga de datos -----
    try:
        bank_df = cargar_extracto(file_ext)
    except Exception as e:
        st.error(f"Error cargando/normalizando EXTRACTO: {e}")
        return

    try:
        inv_df = cargar_facturas(file_inv)
    except Exception as e:
        st.error(f"Error cargando/normalizando FACTURAS: {e}")
        return

    st.subheader("Extracto bancario normalizado")
    st.dataframe(bank_df.head(50), use_container_width=True)

    st.subheader("Facturas normalizadas")
    st.dataframe(inv_df.head(50), use_container_width=True)

    # ----- Diagnóstico de rangos de fechas -----
    st.markdown("---")
    st.subheader("📅 Rangos de fechas")

    ext_min, ext_max = bank_df["Fecha"].min(), bank_df["Fecha"].max()
    fac_min, fac_max = inv_df["Fecha"].min(), inv_df["Fecha"].max()

    st.write(
        f"**Extracto bancario:** "
        f"{ext_min.date() if pd.notna(ext_min) else '–'} → "
        f"{ext_max.date() if pd.notna(ext_max) else '–'}"
    )
    st.write(
        f"**Facturas:** "
        f"{fac_min.date() if pd.notna(fac_min) else '–'} → "
        f"{fac_max.date() if pd.notna(fac_max) else '–'}"
    )

    # Cálculo de solape
    overlap_start = (
        max(ext_min, fac_min) if pd.notna(ext_min) and pd.notna(fac_min) else None
    )
    overlap_end = (
        min(ext_max, fac_max) if pd.notna(ext_max) and pd.notna(fac_max) else None
    )

    if overlap_start and overlap_end and overlap_start <= overlap_end:
        st.write(
            f"**Rango común (solape):** "
            f"{overlap_start.date()} → {overlap_end.date()}"
        )

        limitar_solape = st.checkbox(
            "Limitar conciliación al rango común de fechas",
            value=True,
            help="Si se activa, solo se concilian los movimientos y facturas dentro del solape.",
        )

        if limitar_solape:
            bank_df = bank_df[
                bank_df["Fecha"].between(overlap_start, overlap_end)
            ].copy()
            inv_df = inv_df[
                inv_df["Fecha"].between(overlap_start, overlap_end)
            ].copy()
    else:
        st.warning(
            "No hay solape de fechas entre extracto y facturas. "
            "La conciliación generará muchos pendientes/facturas sin usar."
        )

    # ----- Clasificación por reglas (sin IA) -----
    st.markdown("---")
    st.subheader("🧠 Clasificación por reglas (sin IA)")

    bank_df["tipo_regla"] = bank_df["Concepto"].apply(clasificar_por_reglas)

    st.write("Ejemplos de clasificación en el extracto:")
    st.dataframe(
        bank_df[["Fecha", "Concepto", "Importe", "tipo_regla"]].head(50),
        use_container_width=True,
    )

    st.markdown("---")
    st.header("🔗 Conciliación")

    # ----- Conciliación -----
    bank_res, inv_res, gastos, pend, fact_sin_usar = conciliar_1a1(
        bank_df, inv_df, dias_ventana=dias_ventana, tolerancia_cent=tolerancia_cent
    )

    total_gastos = len(gastos)
    conciliados = int(bank_res["Conciliada"].sum())
    pendientes = len(pend)
    facturas_total = len(inv_res)
    facturas_usadas = int(inv_res["Usada"].sum())
    facturas_no_usadas = len(fact_sin_usar)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gastos totales", total_gastos)
    c2.metric("Gastos conciliados", conciliados)
    c3.metric("Cargos pendientes", pendientes)
    c4.metric("Facturas sin usar", facturas_no_usadas)

    st.subheader("Detalle de movimientos bancarios con conciliación")
    st.dataframe(bank_res, use_container_width=True)

    st.subheader("🕐 Cargos pendientes (gastos sin factura asociada)")
    if pend.empty:
        st.success("No hay cargos pendientes. 🎉")
    else:
        # Resumen por tipo_regla
        if "tipo_regla" in pend.columns:
            n_fact = (pend["tipo_regla"] == "factura_proveedor").sum()
            n_comis = (pend["tipo_regla"] == "comision_bancaria").sum()
            st.write(
                f"Pendientes clasificados por reglas: "
                f"{n_fact} posibles **facturas de proveedor**, "
                f"{n_comis} **comisiones bancarias**, "
                f"{pendientes - n_fact - n_comis} otros."
            )

            st.dataframe(
                pend[["RowID", "Fecha", "Concepto", "Importe", "tipo_regla"]],
                use_container_width=True,
            )
        else:
            st.dataframe(pend, use_container_width=True)

    st.subheader("📄 Facturas no conciliadas en el extracto")
    if fact_sin_usar.empty:
        st.info("Todas las facturas han sido usadas en la conciliación.")
    else:
        st.dataframe(fact_sin_usar, use_container_width=True)


if __name__ == "__main__":
    main()
