import os
import json
from datetime import timedelta

import pandas as pd
import streamlit as st
from openai import OpenAI


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


def cargar_proveedores(file) -> pd.DataFrame:
    """
    Carga una agenda de proveedores con columnas:
    Proveedor, Email
    """
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "proveedor" not in df.columns or "email" not in df.columns:
        raise ValueError(
            f"La agenda de proveedores debe tener columnas 'Proveedor' y 'Email'. "
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    df["Proveedor"] = df["proveedor"].astype(str).str.strip()
    df["Email"] = df["email"].astype(str).str.strip()

    # Clave normalizada para join
    df["ProveedorClave"] = df["Proveedor"].str.upper().str.strip()

    return df[["Proveedor", "Email", "ProveedorClave"]]


# ==============================================
# Clasificación por reglas (sin IA)
# ==============================================

def clasificar_por_reglas(concepto: str) -> str:
    """
    Clasificación básica por reglas sobre el texto del concepto.
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

    # Proveedores típicos (ejemplo; ajusta a tu casuística real)
    if any(
        palabra in c
        for palabra in [
            "ENDESA",
            "IBERDROLA",
            "VODAFONE",
            "MOVISTAR",
            "ORANGE",
            "AGUA",
            "SUMINISTRO",
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

    return "otro"


# ==============================================
# Clasificación por IA (opcional)
# ==============================================

def clasificar_movimiento_ia(concepto: str, importe: float, fecha, client: OpenAI,
                             model: str = "gpt-4o-mini") -> dict:
    """
    Llama a la IA para clasificar un solo movimiento bancario.
    Devuelve:
      - tipo_ia: 'comision_bancaria' | 'factura_proveedor' | 'impuesto_o_tasa' | 'nomina_o_seg_social' | 'otro'
      - proveedor_probable: str
      - es_factura: bool
    """
    if not isinstance(concepto, str):
        concepto = str(concepto) if concepto is not None else ""

    fecha_str = ""
    if pd.notna(fecha):
        fecha_str = str(fecha.date())

    prompt = (
        "Eres un asistente experto en contabilidad que clasifica movimientos bancarios.\n\n"
        "Te doy un único movimiento con estos datos:\n"
        f"- Concepto: {concepto}\n"
        f"- Importe: {importe}\n"
        f"- Fecha: {fecha_str}\n\n"
        "Quiero que devuelvas SOLO un JSON con esta estructura:\n"
        "{\n"
        '  \"tipo\": \"comision_bancaria | factura_proveedor | impuesto_o_tasa | nomina_o_seg_social | otro\",\n'
        '  \"proveedor_probable\": \"texto con el nombre del proveedor si aplica, o \"\" si no se sabe\",\n'
        '  \"es_factura\": true or false\n'
        "}\n"
        "No añadas explicaciones, solo el JSON."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Responde siempre SOLO con JSON válido."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        tipo = data.get("tipo", "otro")
        proveedor_probable = data.get("proveedor_probable", "")
        es_factura = bool(data.get("es_factura", False))
        return {
            "tipo_ia": tipo,
            "proveedor_probable": proveedor_probable,
            "es_factura": es_factura,
        }
    except Exception:
        # En caso de error, devolvemos algo neutro
        return {
            "tipo_ia": "otro",
            "proveedor_probable": "",
            "es_factura": False,
        }


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
    bank["Conciliada"] = False
    bank["FacturaID"] = None
    if "Proveedor" not in bank.columns:
        bank["Proveedor"] = None
    if "NumFactura" not in bank.columns:
        bank["NumFactura"] = None

    inv["Usada"] = False

    gastos = bank[bank["Importe"] < 0].copy()
    if gastos.empty or inv.empty:
        pendientes = gastos.copy()
        facturas_sin_usar = inv[~inv["Usada"]].copy()
        return bank, inv, gastos, pendientes, facturas_sin_usar

    used_facts = set()

    for _, mov in gastos.iterrows():
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
    st.set_page_config(page_title="Conciliador bancario", layout="wide")
    st.title("🧾 Conciliador bancario (Cloud)")

    # ----- Sidebar -----
    st.sidebar.header("1. Subir ficheros")
    file_ext = st.sidebar.file_uploader(
        "Extracto bancario (CSV/Excel)", type=["csv", "xlsx", "xls"]
    )
    file_inv = st.sidebar.file_uploader(
        "Listado de facturas (CSV/Excel)", type=["csv", "xlsx", "xls"]
    )
    file_prov = st.sidebar.file_uploader(
        "Agenda de proveedores (opcional, CSV/Excel con columnas Proveedor y Email)",
        type=["csv", "xlsx", "xls"],
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

    st.sidebar.header("3. IA (opcional)")
    use_ai = st.sidebar.checkbox(
        "Usar IA para clasificar cargos pendientes",
        value=False,
        help="La IA se usa solo sobre los cargos pendientes tras la conciliación.",
    )

    api_key_input = st.sidebar.text_input(
        "OpenAI API Key (si no usas Secrets / variable de entorno)",
        type="password",
        help="Si ya tienes OPENAI_API_KEY en el sistema/Streamlit, puedes dejar esto vacío.",
    )

    max_filas_ia = st.sidebar.slider(
        "Máx. cargos pendientes a clasificar con IA",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
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

    prov_df = None
    if file_prov is not None:
        try:
            prov_df = cargar_proveedores(file_prov)
        except Exception as e:
            st.warning(f"No se ha podido cargar la agenda de proveedores: {e}")

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

    # ----- Clasificación por reglas -----
    st.markdown("---")
    st.subheader("🧠 Clasificación por reglas (sin IA)")
    bank_df["tipo_regla"] = bank_df["Concepto"].apply(clasificar_por_reglas)
    st.dataframe(
        bank_df[["Fecha", "Concepto", "Importe", "tipo_regla"]].head(50),
        use_container_width=True,
    )

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

    tab_conc, tab_recl = st.tabs(["🔗 Conciliación", "📬 Reclamaciones"])

    # ==========================================
    # TAB 1: Conciliación
    # ==========================================
    with tab_conc:
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
            # Resumen por reglas
            if "tipo_regla" in pend.columns:
                n_fact_regla = (pend["tipo_regla"] == "factura_proveedor").sum()
                n_comis_regla = (pend["tipo_regla"] == "comision_bancaria").sum()
                st.write(
                    f"Por reglas: {n_fact_regla} posibles **facturas de proveedor**, "
                    f"{n_comis_regla} **comisiones bancarias**."
                )
            st.dataframe(
                pend[["RowID", "Fecha", "Concepto", "Importe", "tipo_regla"]],
                use_container_width=True,
            )

        st.subheader("📄 Facturas no conciliadas en el extracto")
        if fact_sin_usar.empty:
            st.info("Todas las facturas han sido usadas en la conciliación.")
        else:
            # Elegimos las columnas clave a mostrar
            cols_fact = ["Fecha", "Importe", "Proveedor", "NumFactura"]
            cols_fact = [c for c in cols_fact if c in fact_sin_usar.columns]

        st.dataframe(
            fact_sin_usar[cols_fact],
            use_container_width=True,
        )

    # ==========================================
    # TAB 2: Reclamaciones
    # ==========================================
    with tab_recl:
        st.subheader("📬 Reclamaciones de cargos sin factura")

        if pend.empty:
            st.success("No hay cargos pendientes, nada que reclamar. 🎉")
            return

        # IA sobre pendientes (opcional)
        if use_ai:
            api_key = api_key_input.strip() or os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                st.error(
                    "Has activado IA pero no hay API Key. "
                    "Configura OPENAI_API_KEY en Secrets o rellena el campo en el sidebar."
                )
            else:
                try:
                    client = OpenAI(api_key=api_key)
                    st.info(f"Clasificando con IA hasta {max_filas_ia} cargos pendientes...")
                    resultados_ia = []

                    for idx, row in pend.head(max_filas_ia).iterrows():
                        r = clasificar_movimiento_ia(
                            concepto=row.get("Concepto", ""),
                            importe=row.get("Importe", 0.0),
                            fecha=row.get("Fecha", None),
                            client=client,
                        )
                        resultados_ia.append((idx, r))

                    for idx, r in resultados_ia:
                        for col in ["tipo_ia", "proveedor_probable", "es_factura"]:
                            pend.loc[idx, col] = r[col]
                            bank_res.loc[
                                bank_res["RowID"] == pend.loc[idx, "RowID"], col
                            ] = r[col]

                    st.success("Clasificación por IA aplicada a los cargos pendientes seleccionados.")
                except Exception as e:
                    st.error(f"Error llamando a la IA: {e}")

        # Determinar qué pendientes son reclamables (factura proveedor)
        pend_recl = pend.copy()

        if "es_factura" in pend_recl.columns:
            mask_recl = pend_recl["es_factura"] == True
        elif "tipo_ia" in pend_recl.columns:
            mask_recl = pend_recl["tipo_ia"] == "factura_proveedor"
        else:
            mask_recl = pend_recl["tipo_regla"] == "factura_proveedor"

        pend_recl = pend_recl[mask_recl].copy()

        if pend_recl.empty:
            st.info(
                "No hay cargos pendientes que la IA o las reglas identifiquen como 'factura de proveedor'. "
                "Revisa la clasificación o ajusta la ventana de fechas."
            )
            return

        # Clave de proveedor probable para join con agenda (si existe)
        if "proveedor_probable" in pend_recl.columns:
            pend_recl["ProveedorProbable"] = pend_recl["proveedor_probable"].fillna("").astype(str).str.strip()
        else:
            pend_recl["ProveedorProbable"] = ""

        pend_recl["ProveedorClave"] = pend_recl["ProveedorProbable"].str.upper().str.strip()

        # Join con agenda de proveedores (si hay)
        if prov_df is not None:
            pend_recl = pend_recl.merge(
                prov_df[["Proveedor", "Email", "ProveedorClave"]],
                on="ProveedorClave",
                how="left",
                suffixes=("", "_agenda"),
            )
        else:
            pend_recl["Proveedor"] = pend_recl.get("ProveedorProbable", "")
            pend_recl["Email"] = ""

        # Generar texto de email
        def generar_email(row):
            fecha = row.get("Fecha", None)
            fecha_txt = (
                fecha.strftime("%d/%m/%Y") if isinstance(fecha, pd.Timestamp) else ""
            )
            imp = row.get("Importe", 0.0)
            imp_abs = abs(float(imp)) if pd.notna(imp) else 0.0
            proveedor = row.get("ProveedorProbable", "") or row.get("Proveedor", "")
            concepto = row.get("Concepto", "")

            saludo_prov = f"{proveedor}," if proveedor else ""
            cuerpo = (
                f"Buenas {saludo_prov}\n\n"
                f"¿Nos pueden enviar la factura correspondiente al cargo bancario "
                f"de fecha {fecha_txt} e importe {imp_abs:.2f} €"
            )
            if concepto:
                cuerpo += f" (concepto: {concepto})"
            cuerpo += "?\n\nMuchas gracias.\n\nUn saludo."

            return cuerpo

        pend_recl["TextoEmail"] = pend_recl.apply(generar_email, axis=1)

        st.write(
            f"Se han identificado **{len(pend_recl)} cargos** pendientes como "
            f"posibles facturas de proveedor susceptibles de reclamación."
        )

        cols_vista = [
            "RowID",
            "Fecha",
            "Concepto",
            "Importe",
            "tipo_regla",
        ]
        for extra in ["tipo_ia", "ProveedorProbable", "Proveedor", "Email", "TextoEmail"]:
            if extra in pend_recl.columns:
                cols_vista.append(extra)

        st.dataframe(
            pend_recl[cols_vista],
            use_container_width=True,
        )

        # Descarga en Excel
        try:
            import io

            with io.BytesIO() as buf:
                with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                    pend_recl[cols_vista].to_excel(
                        writer, index=False, sheet_name="Reclamaciones"
                    )
                data_xlsx = buf.getvalue()
            st.download_button(
                "⬇️ Descargar listado de reclamaciones (XLSX)",
                data=data_xlsx,
                file_name="reclamaciones_conciliador.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
            )
        except Exception as e:
            st.warning(f"No se ha podido generar el Excel de reclamaciones: {e}")


if __name__ == "__main__":
    main()
