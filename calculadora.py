import streamlit as st
import numpy as np
import math
import re
import streamlit.components.v1 as components
import pandas as pd
# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Engine Calculus - GeoGebra")

# --- TUS CORRECCIONES Y DICCIONARIOS ---
correcciones = {
    'sin': 'np.sin', 'cos': 'np.cos', 'tan': 'np.tan',
    'asin': 'np.arcsin', 'acos': 'np.arccos', 'atan': 'np.arctan',
    'csc': '1/np.sin', 'sec': '1/np.cos', 'cot': '1/np.tan',
    'sinh': 'np.sinh', 'cosh': 'np.cosh', 'tanh': 'np.tanh',
    'ln': 'np.log', 'log10': 'np.log10', 'exp': 'np.exp',
    'sqrt': 'np.sqrt', 'cbrt': 'np.cbrt', 'abs': 'np.abs',
    'pi': 'np.pi', 'e': 'np.e'
}

dict_labels = {
    "Grafica basica": [[], []],
    "Secante": [["x0", "x1"], ["x0", "x1", "x2", "f(x0)", "f(x1)", "error"]],
    "Newton": [["derivada de la funcion", "punto inicial"], ["x", "f(x)", "f'(x)", "error"]],
    "Biseccion": [["limite inferior", "limite superior"], ["a", "b", "c", "f(a)", "f(b)", "f(c)", "error"]],
    "Trapecio": [["limite inferior", "limite superior", "n"], ["n (segmentos)", "h", "Resultado Integral"]],
    "Simpson13": [["limite inferior", "limite superior", "n"], ["n (par)", "h", "Resultado Integral"]],
    "Simpson38": [["limite inferior", "limite superior", "n"], ["n (múltiplo 3)", "h", "Resultado Integral"]],
    "Punto Fijo": [["x inicial"], ["iteración", "x_n", "f(x_n)", "error"]],
    "Muller": [["x0", "x1", "x2"], ["iteración", "x0", "x1", "x2", "x3", "f(x3)"]]
}


# --- LÓGICA DE LIMPIEZA DE EXPRESIÓN (Adaptada de tu config_inputs) ---
def limpiar_expresion(expr):
    expr = expr.replace(' ', '').lower()
    expr = re.sub(r'root\(([^,]+),([^,]+)\)', r'(\1)**(1/\2)', expr)
    expr = re.sub(r'log\(([^,]+),([^,]+)\)', r'(np.log(\1)/np.log(\2))', expr)
    expr = re.sub(r'([a-zA-Z]+)\^(\d+)\((.+?)\)', r'(\1(\3))**\2', expr)
    expr = re.sub(r'([\d.]+)([a-zA-Z\(])', r'\1*\2', expr)
    expr = re.sub(r'(\))(\()', r'\1*\2', expr)
    expr = re.sub(r'(\))([a-zA-Z])', r'\1*\2', expr)
    expr = expr.replace('^', '**')
    for h in sorted(correcciones.keys(), key=len, reverse=True):
        expr = re.sub(r'\b' + h + r'\b', correcciones[h], expr, flags=re.IGNORECASE)
    return expr.replace('np.np.', 'np.')

# --- TUS MÉTODOS NUMÉRICOS (Bisección simplificada para ejemplo) ---

def biseccion(f, a, b, tol=1e-2):
    list_output=[]
    if f(a) * f(b) >= 0:
        raise ValueError("El intervalo [a, b] no cumple que (f(a)*f(b) >= 0).")
    max_iter=math.ceil(math.log2(abs(b-a)/tol))    
    for i in range(max_iter):
        c = (a + b) / 2
        error=(b - a) / 2
        fa=f(a)
        fb=f(b)        
        fc=f(c)
        if abs(fc) < tol or error< tol:
            break
        valores = [a, b, c, fa, fb, fc, error]
        list_output.append(valores)
        if fa * fc < 0:
            b = c
        else:
            a = c
            
    return list_output

def newton_raphson(f, df, x0, tol=1e-5):
    list_output=[]
    xn = x0
    i=0
    while True:
        f_val = f(xn)
        df_val = df(xn)
        if abs(df_val) < 1e-12:
            raise ZeroDivisionError("La derivada es cero. No se puede continuar.")
        xn_next = xn - f_val / df_val
        error=xn_next - xn
        if abs(error) < tol:
            break
        valores=[xn,f_val,df_val, error]
        list_output.append(valores)
        xn = xn_next
        i+=1
    return list_output

def secante(f, x0, x1, tol=1e-5):
    list_output=[]
    i=0
    while True:
        f_x0 = f(x0)
        f_x1 = f(x1)
        if abs(f_x1 - f_x0) < 1e-12:
            raise ZeroDivisionError("División por cero detectada en la secante.")        
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        error=x2-x1
        if abs(error) < tol:
            break
        valores=[x0, x1,x2, f_x0, f_x1,error]
        list_output.append(valores)    
        x0, x1 = x1, x2
        i+=1
    return list_output
def trapecio(f, li, ls, n):
    n = int(n)
    xs, ys = values_table(f, li, ls, n)
    h = (ls - li) / n
    suma = sum(2 * ys[i] for i in range(1, n))
    resultado = (h / 2) * (ys[0] + ys[-1] + suma)
    # Retornamos los pasos clave
    return [[n, h, resultado]]

def simpson_tercio(f, li, ls, n):
    n = int(n)
    if n % 2 != 0:
        raise ValueError("Para Simpson 1/3, 'n' debe ser par.")
    xs, ys = values_table(f, li, ls, n)
    h = (ls - li) / n
    suma = 0
    for i in range(1, n):
        suma += (4 * ys[i] if i % 2 != 0 else 2 * ys[i])
    resultado = (h / 3) * (ys[0] + ys[-1] + suma)
    return [[n, h, resultado]]

def simpson_38(f, li, ls, n):
    n = int(n)
    if n % 3 != 0:
        raise ValueError("Para Simpson 3/8, 'n' debe ser múltiplo de 3.")
    xs, ys = values_table(f, li, ls, n)
    h = (ls - li) / n
    suma = 0
    for i in range(1, n):
        suma += (2 * ys[i] if i % 3 == 0 else 3 * ys[i])
    resultado = (3 * h / 8) * (ys[0] + ys[-1] + suma)
    return [[n, h, resultado]]

def fixed_point(f, x0, tol, max_iter=50):
    list_output = []
    x = x0
    for i in range(max_iter):
        x_next = f(x)
        error = abs(x_next - x)
        list_output.append([i+1, x, x_next, error])
        if error < tol:
            break
        x = x_next
    return list_output

def muller(f, x0, x1, x2, tol=1e-5, max_iter=50):
    list_output = []
    for i in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        h0, h1 = x1 - x0, x2 - x1
        d0, d1 = (f1 - f0) / h0, (f2 - f1) / h1
        
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = f2
        
        # Discriminante
        rad = np.sqrt(b**2 - 4*a*c + 0j) # 0j para manejar complejos si es necesario
        den = b + rad if abs(b + rad) > abs(b - rad) else b - rad
        
        dx = -2 * c / den
        x3 = x2 + dx.real # Tomamos la parte real para graficar
        
        list_output.append([i+1, x0, x1, x2, x3, f(x3)])
        
        if abs(f(x3)) < tol:
            break
        x0, x1, x2 = x1, x2, x3
    return list_output


def triplePtS(pts_x):   
    if len(pts_x) < 2:
        return 0
    h = pts_x[1] - pts_x[0]
    for i in range(len(pts_x) - 1):
        dif_actual = pts_x[i+1] - pts_x[i]
        if not math.isclose(dif_actual, h, rel_tol=1e-7):
            raise ValueError("Los puntos en el eje X deben estar a la misma distancia (equidistantes).")
    return h



def values_table(f, li, ls, n):
    n = int(n)
    xs = np.linspace(li, ls, n + 1)
    try:
        ys = f(xs)
        if np.isscalar(ys):
            ys = np.full_like(xs, ys)
    except:
        ys = np.array([f(float(x)) for x in xs])
        
    return xs, ys
# --- INTERFAZ STREAMLIT ---
st.title("Metodos Numericos")


if "resultados" not in st.session_state:
    st.session_state.resultados = []

if "puntos" not in st.session_state:
    st.session_state.puntos = []

with st.sidebar:
    st.header("Configuración")
    metodo_sel = st.selectbox("Método", list(dict_labels.keys()))
    func_input = st.text_input("Función f(x)", value="x^2 - 4")
    
    params_input = {}
    for label in dict_labels[metodo_sel][0]:
        params_input[label] = st.text_input(f"Ingrese {label}", value="0")

    tol = st.number_input("Tolerancia", value=1e-5, format="%.5f")
    btn_start = st.button("CALCULAR", type="primary")

# --- PROCESAMIENTO ---
resultados = []
puntos_geogebra = []
if "calculado" not in st.session_state:
    st.session_state.calculado = False

if "metodo_anterior" not in st.session_state:
    st.session_state.metodo_anterior = None

# Detectar cambio de método
if metodo_sel != st.session_state.metodo_anterior:
    st.session_state.calculado = False
    st.session_state.resultados = []
    st.session_state.puntos = []

# Guardar método actual
st.session_state.metodo_anterior = metodo_sel

if btn_start:
    try:
        exp_py = limpiar_expresion(func_input)
        f = lambda x: eval(exp_py, {"x": x, "np": np, "math": math})
        res=[]
        if metodo_sel == "Grafica basica":
            st.session_state.puntos = []
        elif metodo_sel == "Biseccion":
            li = float(params_input["limite inferior"])
            ls = float(params_input["limite superior"])
            res = biseccion(f, li, ls, tol)
            if isinstance(res, str):
                st.error(res)
            else:
                st.session_state.resultados = res
                st.session_state.puntos = [fila[2] for fila in res]
        
        
        elif metodo_sel == "Newton":
            df_expr = limpiar_expresion(params_input["derivada de la funcion"])
            df = lambda x: eval(df_expr, {"x": x, "np": np, "math": math})
            x0 = float(params_input["punto inicial"])
            res = newton_raphson(f, df, x0, tol)
            st.session_state.puntos = [fila[0] for fila in res]      
    
        elif metodo_sel == "Secante":
            x0 = float(params_input["x0"])
            x1 = float(params_input["x1"])
            res = secante(f, x0, x1, tol)
            st.session_state.puntos = [fila[2] for fila in res]      
    
        elif metodo_sel == "Trapecio":
            li = float(params_input["limite inferior"])
            ls = float(params_input["limite superior"])
            n = int(params_input["n"])
            res = trapecio(f, li, ls, n)
            st.session_state.puntos = []      

    
        elif metodo_sel == "Simpson13":
            li = float(params_input["limite inferior"])
            ls = float(params_input["limite superior"])
            n = int(params_input["n"])
            res = simpson_tercio(f, li, ls, n)
            st.session_state.puntos = []    

    
        elif metodo_sel == "Simpson38":
            li = float(params_input["limite inferior"])
            ls = float(params_input["limite superior"])
            n = int(params_input["n"])
            res = simpson_38(f, li, ls, n)
            st.session_state.puntos = []
 
    
        elif metodo_sel == "Punto Fijo":
            x0 = float(params_input["x inicial"])
            res = fixed_point(f, x0, tol, 50)
            st.session_state.puntos = [fila[2] for fila in res]      

    
        elif metodo_sel == "Muller":
            x0 = float(params_input["x0"])
            x1 = float(params_input["x1"])
            x2 = float(params_input["x2"])
            res = muller(f, x0, x1, x2, tol)
            st.session_state.puntos = [fila[4] for fila in res]  
        st.session_state.resultados = res    
        st.session_state.calculado = True

    except Exception as e:
        st.error(f"Error en la expresión: {e}")
        
st.subheader("Consola de Resultados")
if st.session_state.calculado and metodo_sel != "Grafica basica":
    if len(st.session_state.resultados) > 0:
        columnas_disponibles = dict_labels[metodo_sel][1]

        cols_visibles = st.multiselect(
            "Filtrar valores:",
            columnas_disponibles,
            default=columnas_disponibles,
            key="filtro"
        )

        df = pd.DataFrame(st.session_state.resultados)

        if len(df.columns) == len(columnas_disponibles):
            df.columns = columnas_disponibles
            st.dataframe(df[cols_visibles], use_container_width=True)
    else:
        st.warning("El método no generó resultados.")
            
            
st.subheader("Visualización GeoGebra")

# Limpiar la función para GeoGebra
func_ggb = func_input.replace("**", "^")

# Construcción segura de los comandos de puntos
# GeoGebra usa el punto (.) como separador decimal, asegurémonos de eso
puntos_js = ""
for i, px in enumerate(st.session_state.puntos):        # Convertimos a string con formato de punto decimal por si acaso
    px_val = f"{px:.8f}".replace(",", ".")
    puntos_js += f'ggbApplet.evalCommand("P_{i}=({px_val}, 0)");\n'

ggb_html = f"""
<script src="https://www.geogebra.org/apps/deployggb.js"></script>
<div id="ggb-element" style="border: 1px solid #ddd; border-radius: 8px;"></div>

<script>
var params = {{
    "appName": "graphing",
    "width": 1100,
    "height": 600,
    "showAlgebraInput": true,
    "showToolBar": true,
    "enableLabelDrags": false,
    "errorDialogsActive": true
}};
var applet = new GGBApplet(params, true);

// Función para enviar los comandos cuando el applet esté listo
function enviarComandos() {{
    if (window.ggbApplet && typeof ggbApplet.evalCommand === 'function') {{
        // Limpiar comandos previos
        ggbApplet.newConstruction();
        
        // Graficar función
        ggbApplet.evalCommand("f(x) = {func_ggb}");
        
        // Graficar puntos
        {puntos_js}
        
        // Ajustar vista para que se vean los puntos
        ggbApplet.evalCommand("ZoomIn(-10, -10, 10, 10)"); 
    }} else {{
        // Si no está listo, reintentar en 500ms
        setTimeout(enviarComandos, 500);
    }}
}}

window.onload = function() {{   
    applet.inject('ggb-element');
    enviarComandos();
}};
</script>
"""
if st.session_state.calculado:
    components.html(ggb_html, height=600)


