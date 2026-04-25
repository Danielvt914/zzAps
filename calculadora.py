import streamlit as st
import numpy as np
import math
import re
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Fisica")

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
ALLOWED_GLOBALS = {
    "np": np,
    "math": math
}
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



def crear_funcion(expr):
    expr = limpiar_expresion(expr)

    def f(x):
        return eval(
            expr,
            {"__builtins__": {}},
            {**ALLOWED_GLOBALS, "x": x}
        )

    return f

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
def trapecio(f=None, li=None, ls=None, n=None, xs=None, ys=None):

    if xs is not None and ys is not None:
        if len(xs)!=len(ys):
            raise ValueError("x e y deben tener misma longitud")

        h=triplePtS(xs)

        suma=0
        for i in range(1,len(ys)-1):
            suma+=2*ys[i]

        resultado=(h/2)*(ys[0]+ys[-1]+suma)

        return [[len(xs)-1,h,resultado]]

    # modo función
    n=int(n)
    xs,ys=values_table(f,li,ls,n)
    h=(ls-li)/n

    suma=0
    for i in range(1,n):
        suma+=2*ys[i]

    resultado=(h/2)*(ys[0]+ys[-1]+suma)

    return [[n,h,resultado]]

def simpson_tercio(f=None, li=None, ls=None, n=None, xs=None, ys=None):

    if xs is not None and ys is not None:

        if len(xs)%2==0:
            raise ValueError("Simpson 1/3 requiere número impar de puntos")

        h=triplePtS(xs)

        suma=0

        for i in range(1,len(ys)-1):
            if i%2!=0:
                suma+=4*ys[i]
            else:
                suma+=2*ys[i]

        resultado=(h/3)*(ys[0]+ys[-1]+suma)

        return [[len(xs)-1,h,resultado]]

    n=int(n)

    if n%2!=0:
        raise ValueError("n debe ser par")

    xs,ys=values_table(f,li,ls,n)

    h=(ls-li)/n
    suma=0

    for i in range(1,n):
        if i%2!=0:
            suma+=4*ys[i]
        else:
            suma+=2*ys[i]

    resultado=(h/3)*(ys[0]+ys[-1]+suma)

    return [[n,h,resultado]]

def simpson_38(f=None, li=None, ls=None, n=None, xs=None, ys=None):

    if xs is not None and ys is not None:

        if (len(xs)-1)%3!=0:
            raise ValueError("Segmentos deben ser múltiplo de 3")

        h=triplePtS(xs)

        suma=0

        for i in range(1,len(ys)-1):
            if i%3==0:
                suma+=2*ys[i]
            else:
                suma+=3*ys[i]

        resultado=(3*h/8)*(ys[0]+ys[-1]+suma)

        return [[len(xs)-1,h,resultado]]

    n=int(n)

    if n%3!=0:
        raise ValueError("n debe ser múltiplo de 3")

    xs,ys=values_table(f,li,ls,n)

    h=(ls-li)/n
    suma=0

    for i in range(1,n):
        if i%3==0:
            suma+=2*ys[i]
        else:
            suma+=3*ys[i]

    resultado=(3*h/8)*(ys[0]+ys[-1]+suma)

    return [[n,h,resultado]]

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

def leer_puntos(texto):
    """
    formato:
    0,1
    1,2
    2,5
    """
    xs=[]
    ys=[]

    lineas=texto.strip().split("\n")

    for linea in lineas:
        x,y=linea.split(",")
        xs.append(float(x))
        ys.append(float(y))

    return np.array(xs),np.array(ys)

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

    metodo_sel=st.selectbox(
        "Método",
        list(dict_labels.keys())
    )

    usa_puntos=False

    if metodo_sel in ["Trapecio","Simpson13","Simpson38"]:

        modo=st.radio(
            "Entrada",
            [
                "Función",
                "Ingresar puntos",
                "Generar puntos"
            ]
        )

        usa_puntos=modo!="Función"

    if not usa_puntos:

        func_input=st.text_input(
            "Función f(x)",
            value="x^2-4"
        )

    else:

        func_input=""

        if modo=="Ingresar puntos":

            puntos_texto=st.text_area(
            "Puntos x,y",
"""0,1
1,2
2,5"""
            )

        elif modo=="Generar puntos":

            func_generadora=st.text_input(
                "Función para generar puntos",
                "x^2"
            )

            g_li=st.number_input(
                "Inicio",
                value=0.0
            )

            g_ls=st.number_input(
                "Fin",
                value=4.0
            )

            cantidad=st.number_input(
                "Cantidad de puntos",
                value=5,
                min_value=2
            )


    params_input={}
    for label in dict_labels[metodo_sel][0]:
        params_input[label]=st.text_input(
            label,
            value="0"
        )

    tol=st.number_input(
        "Tolerancia",
        value=1e-5,
        format="%.5f"
    )

    btn_start=st.button(
        "CALCULAR",
        use_container_width=True
    )

# --- PROCESAMIENTO ---
resultados = []
puntos_geogebra = []
if "calculado" not in st.session_state:
    st.session_state.calculado = False

if "metodo_anterior" not in st.session_state:
    st.session_state.metodo_anterior = None

# Detectar cambio de método
if metodo_sel != st.session_state.metodo_anterior:
    st.session_state.calculado=False
    st.session_state.resultados=[]
    st.session_state.puntos=[]

st.session_state.metodo_anterior=metodo_sel


if btn_start:
    try:

        res=[]

        metodos_puntos=[
            "Trapecio",
            "Simpson13",
            "Simpson38"
        ]

        # ---------------- VALIDACIONES ----------------

        if metodo_sel in metodos_puntos:

            if modo=="Función" and func_input.strip()=="":
                st.error("Ingrese una función")
                st.stop()

            if modo=="Ingresar puntos" and puntos_texto.strip()=="":
                st.error("Ingrese puntos")
                st.stop()


        # crear función solo si realmente se usa
        f=None

        if (
            metodo_sel not in metodos_puntos
            or modo=="Función"
        ):
            exp_py=limpiar_expresion(func_input)
            f = crear_funcion(exp_py)


        # ------------ MÉTODOS ----------------

        if metodo_sel=="Grafica basica":

            st.session_state.puntos=[]


        elif metodo_sel=="Biseccion":

            li=float(params_input["limite inferior"])
            ls=float(params_input["limite superior"])

            res=biseccion(
                f,
                li,
                ls,
                tol
            )

            st.session_state.puntos=[
                fila[2] for fila in res
            ]


        elif metodo_sel=="Newton":

            df_expr=limpiar_expresion(
                params_input["derivada de la funcion"]
            )

            df=crear_funcion(df_expr)

            x0=float(params_input["punto inicial"])

            res=newton_raphson(
                f,
                df,
                x0,
                tol
            )

            st.session_state.puntos=[
                fila[0] for fila in res
            ]


        elif metodo_sel=="Secante":

            x0=float(params_input["x0"])
            x1=float(params_input["x1"])

            res=secante(
                f,
                x0,
                x1,
                tol
            )

            st.session_state.puntos=[
                fila[2] for fila in res
            ]


        # --------- TRAPECIO ---------

        elif metodo_sel=="Trapecio":

            if modo=="Función":

                li=float(
                    params_input["limite inferior"]
                )

                ls=float(
                    params_input["limite superior"]
                )

                n=int(
                    params_input["n"]
                )

                res=trapecio(
                    f=f,
                    li=li,
                    ls=ls,
                    n=n
                )

            elif modo=="Ingresar puntos":

                xs,ys=leer_puntos(
                    puntos_texto
                )

                res=trapecio(
                    xs=xs,
                    ys=ys
                )

            else:

                exp_gen=limpiar_expresion(
                    func_generadora
                )

                g=lambda x: eval(
                    exp_gen,
                    {"x":x,"np":np}
                )

                xs=np.linspace(
                    g_li,
                    g_ls,
                    int(cantidad)
                )

                ys=g(xs)

                res=trapecio(
                    xs=xs,
                    ys=ys
                )

            st.session_state.puntos=[]


        # -------- SIMPSON 1/3 --------

        elif metodo_sel=="Simpson13":

            if modo=="Función":

                li=float(
                    params_input["limite inferior"]
                )

                ls=float(
                    params_input["limite superior"]
                )

                n=int(
                    params_input["n"]
                )

                res=simpson_tercio(
                    f=f,
                    li=li,
                    ls=ls,
                    n=n
                )

            elif modo=="Ingresar puntos":

                xs,ys=leer_puntos(
                    puntos_texto
                )

                res=simpson_tercio(
                    xs=xs,
                    ys=ys
                )

            else:

                exp_gen=limpiar_expresion(
                    func_generadora
                )

                g=lambda x: eval(
                    exp_gen,
                    {"x":x,"np":np}
                )

                xs=np.linspace(
                    g_li,
                    g_ls,
                    int(cantidad)
                )

                ys=g(xs)

                res=simpson_tercio(
                    xs=xs,
                    ys=ys
                )

            st.session_state.puntos=[]


        # -------- SIMPSON 3/8 --------

        elif metodo_sel=="Simpson38":

            if modo=="Función":

                li=float(
                    params_input["limite inferior"]
                )

                ls=float(
                    params_input["limite superior"]
                )

                n=int(
                    params_input["n"]
                )

                res=simpson_38(
                    f=f,
                    li=li,
                    ls=ls,
                    n=n
                )

            elif modo=="Ingresar puntos":

                xs,ys=leer_puntos(
                    puntos_texto
                )

                res=simpson_38(
                    xs=xs,
                    ys=ys
                )

            else:

                exp_gen=limpiar_expresion(
                    func_generadora
                )

                g=lambda x: eval(
                    exp_gen,
                    {"x":x,"np":np}
                )

                xs=np.linspace(
                    g_li,
                    g_ls,
                    int(cantidad)
                )

                ys=g(xs)

                res=simpson_38(
                    xs=xs,
                    ys=ys
                )

            st.session_state.puntos=[]


        elif metodo_sel=="Punto Fijo":

            x0=float(
                params_input["x inicial"]
            )

            res=fixed_point(
                f,
                x0,
                tol,
                50
            )

            st.session_state.puntos=[
                fila[2] for fila in res
            ]


        elif metodo_sel=="Muller":

            x0=float(params_input["x0"])
            x1=float(params_input["x1"])
            x2=float(params_input["x2"])

            res=muller(
                f,
                x0,
                x1,
                x2,
                tol
            )

            st.session_state.puntos=[
                fila[4] for fila in res
            ]


        st.session_state.resultados=res
        st.session_state.calculado=True


    except Exception as e:
        st.error(
            f"Error en la expresión: {e}"
        )
        
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
            
            
st.subheader("Visualización")
def graficar():

    if func_input.strip()=="":
        return

    try:

        exp_py=limpiar_expresion(func_input)
        f=lambda x: eval(
            exp_py,
            {"x":x,"np":np,"math":math}
        )

        fig,ax=plt.subplots(
            figsize=(12,6)
        )

        if len(st.session_state.puntos):
            centro=np.mean(st.session_state.puntos)
            x=np.linspace(
                centro-5,
                centro+5,
                1000
            )
        y=f(x)

        ax.plot(
            x,
            y,
            label="f(x)"
        )

        ax.axhline(
            0,
            linewidth=1
        )

        ax.axvline(
            0,
            linewidth=1
        )


        # puntos iterativos
        if len(st.session_state.puntos)>0:

            px=np.array(
                st.session_state.puntos
            )

            py=f(px)

            ax.scatter(
                px,
                py,
                s=60
            )

            for i,(xp,yp) in enumerate(zip(px,py)):
                ax.annotate(
                    f"P{i}",
                    (xp,yp)
                )


        # -------- trapecio y simpson sombreado --------

        if (
            metodo_sel in [
                "Trapecio",
                "Simpson13",
                "Simpson38"
            ]
            and st.session_state.calculado
            and 'li' in locals()
            and 'ls' in locals()
        ):

            xx=np.linspace(
                li,
                ls,
                300
            )

            yy=f(xx)

            ax.fill_between(
                xx,
                yy,
                0,
                alpha=0.3
            )


        ax.grid(True)
        ax.legend()

        st.pyplot(
            fig,
            use_container_width=True
        )

    except:
        st.info(
            "No se pudo graficar."
        )
graficar()
