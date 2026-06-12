"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SUITE DE CÁLCULO VECTORIAL — SISTEMAS UNIFICADOS                    ║
║   Coordenadas Cartesianas (x,y,z) · Cilíndricas (r,θ,z) · Esféricas (r,θ,φ)  ║
║                                                                              ║
║   Módulo 1 · Integrales (Indefinida, Definida, Doble, Triple,                ║
║                          Línea, Flujo, Volumen) + Gráficas                   ║
║   Módulo 2 · Derivadas Mixtas y de Orden Superior                            ║
║   Módulo 3 · Productos Vectoriales (Punto, Cruz, Triples)                    ║
║   Módulo 4 · Operadores Nabla (grad, div, rot, laplaciano)                   ║
║                                                                              ║
║   100% local · sin internet · sin API · Powered by SymPy                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import threading
import traceback
from tkinter import messagebox
import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE APARIENCIA (tema oscuro)
# ----------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C_BG       = "#0d1117"
C_PANEL    = "#161b27"
C_CARD     = "#1c2333"
C_BORDER   = "#2a3450"
C_ACCENT   = "#2FA572"
C_ACCENT2  = "#3b82f6"
C_DANGER   = "#ef4444"
C_PURPLE   = "#8B008B"
C_CYAN     = "#00d4ff"
C_GOLD     = "#f59e0b"
C_TEXT     = "#e2e8f0"
C_MUTED    = "#64748b"
C_OK_BG    = "#052e16"
C_ERR_BG   = "#450a0a"

# ----------------------------------------------------------------------
# MOTOR SIMBÓLICO GLOBAL
# ----------------------------------------------------------------------
x, y, z, r, theta, phi, t, u, v = sp.symbols('x y z r theta phi t u v', real=True)

_NS = {
    'x': x, 'y': y, 'z': z, 'r': r, 'theta': theta, 'phi': phi,
    't': t, 'u': u, 'v': v,
    'pi': sp.pi, 'e': sp.E, 'E': sp.E,
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
    'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
    'sqrt': sp.sqrt, 'Abs': sp.Abs, 'abs': sp.Abs,
    'oo': sp.oo, 'inf': sp.oo,
}
_TRANS = standard_transformations + (implicit_multiplication_application,)

# Allowlist de nombres permitidos en expresiones del usuario.
# parse_expr con local_dict y transformations restringe el espacio de nombres
# a solo los símbolos y funciones matemáticas definidos en _NS, bloqueando
# acceso a builtins de Python como __import__, eval, exec, etc.
_ALLOWED_NAMES = frozenset(_NS.keys())

# Caracteres permitidos en expresiones matemáticas del usuario.
# Bloquea cualquier carácter fuera del conjunto alfanumérico y operadores
# matemáticos estándar antes de llegar al parser de SymPy.
_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 +-*/^().,%_"
)

def parse(s: str):
    """Convierte texto a expresión SymPy usando solo el espacio de nombres
    matemático permitido (_NS). Valida caracteres y rechaza entradas vacías
    antes de pasarlas al parser, bloqueando inyección de código."""
    s = s.strip()
    if not s:
        raise ValueError("La expresión no puede estar vacía.")
    # Validación de caracteres: rechaza cualquier símbolo fuera del conjunto
    # matemático permitido (e.g. comillas, corchetes, punto y coma, etc.)
    invalid = set(s) - _ALLOWED_CHARS
    if invalid:
        raise ValueError(f"Caracteres no permitidos en la expresión: {invalid}")
    return parse_expr(s, local_dict=_NS, transformations=_TRANS)

def to_numpy(expr_sym, *syms):
    return sp.lambdify(syms, expr_sym, modules=['numpy'])

def fmt(expr_sym) -> str:
    try:
        s = str(sp.simplify(expr_sym))
        s = s.replace("**", "^").replace("sqrt", "√").replace("log", "ln")
        return s
    except Exception:
        return str(expr_sym)

def safe_eval1(fn, arr):
    try:
        res = fn(arr)
        return np.where(np.isfinite(res.astype(float)), res.astype(float), np.nan)
    except Exception:
        return np.array([_try_float(fn, xi) for xi in arr])

def safe_eval2(fn, A, B):
    try:
        res = fn(A, B)
        return np.where(np.isfinite(res.astype(float)), res.astype(float), np.nan)
    except Exception:
        return np.vectorize(lambda a,b: _try_float(fn,a,b))(A,B)

def _try_float(fn, *args):
    try:
        v = float(fn(*args))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

# ----------------------------------------------------------------------
# CLASE CoordSystem (corregida: parámetros con default al final)
# ----------------------------------------------------------------------
class CoordSystem:
    def __init__(self, name, sym_vars, jac2d, jac3d, grad, div, rot, lap,
                 line_ds, var_buttons, sph_to_cart=None, color=C_ACCENT2):
        self.name = name
        self.sym_vars = sym_vars
        self.var_names = [str(v) for v in sym_vars]
        self.jacobian2d = jac2d
        self.jacobian3d = jac3d
        self.grad = grad
        self.div = div
        self.rot = rot
        self.lap = lap
        self.line_ds = line_ds
        self.var_buttons = var_buttons
        self.sph_to_cart = sph_to_cart
        self.color = color

# ---------- Cartesianas ----------
def grad_cart(f):
    return (sp.diff(f, x), sp.diff(f, y), sp.diff(f, z))

def div_cart(fx, fy, fz):
    return sp.diff(fx, x) + sp.diff(fy, y) + sp.diff(fz, z)

def rot_cart(fx, fy, fz):
    return (sp.diff(fz, y) - sp.diff(fy, z),
            sp.diff(fx, z) - sp.diff(fz, x),
            sp.diff(fy, x) - sp.diff(fx, y))

def lap_cart(f):
    return sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)

def line_ds_cart(rx, ry, t):
    return sp.sqrt(sp.diff(rx, t)**2 + sp.diff(ry, t)**2)

cart_sys = CoordSystem(
    name="Cartesianas",
    sym_vars=[x, y, z],
    jac2d=1,
    jac3d=1,
    grad=grad_cart,
    div=div_cart,
    rot=rot_cart,
    lap=lap_cart,
    line_ds=line_ds_cart,
    var_buttons=[("x","x"), ("y","y"), ("z","z"), ("t","t"), ("π","pi")],
    sph_to_cart=None,
    color=C_ACCENT2
)

# ---------- Cilíndricas ----------
def grad_cil(f):
    return (sp.diff(f, r), sp.diff(f, theta)/r, sp.diff(f, z))

def div_cil(fr, fth, fz):
    return (1/r)*sp.diff(r*fr, r) + (1/r)*sp.diff(fth, theta) + sp.diff(fz, z)

def rot_cil(fr, fth, fz):
    cr = (1/r)*sp.diff(fz, theta) - sp.diff(fth, z)
    cth = sp.diff(fr, z) - sp.diff(fz, r)
    cz = (1/r)*sp.diff(r*fth, r) - (1/r)*sp.diff(fr, theta)
    return (cr, cth, cz)

def lap_cil(f):
    return sp.diff(f, r, 2) + (1/r)*sp.diff(f, r) + (1/r**2)*sp.diff(f, theta, 2) + sp.diff(f, z, 2)

def line_ds_cil(rt, tht, t):
    dr = sp.diff(rt, t)
    dth = sp.diff(tht, t)
    return sp.sqrt(dr**2 + rt**2 * dth**2)

def sph_to_cart_cil(R, TH, Z):
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    return X, Y, Z

cil_sys = CoordSystem(
    name="Cilíndricas",
    sym_vars=[r, theta, z],
    jac2d=r,
    jac3d=r,
    grad=grad_cil,
    div=div_cil,
    rot=rot_cil,
    lap=lap_cil,
    line_ds=line_ds_cil,
    var_buttons=[("r","r"), ("θ","theta"), ("z","z"), ("t","t"), ("π","pi")],
    sph_to_cart=sph_to_cart_cil,
    color=C_CYAN
)

# ---------- Esféricas ----------
def grad_esf(f):
    return (sp.diff(f, r), sp.diff(f, theta)/r, sp.diff(f, phi)/(r*sp.sin(theta)))

def div_esf(fr, fth, fph):
    t1 = (1/r**2)*sp.diff(r**2 * fr, r)
    t2 = (1/(r*sp.sin(theta)))*sp.diff(sp.sin(theta)*fth, theta)
    t3 = (1/(r*sp.sin(theta)))*sp.diff(fph, phi)
    return t1 + t2 + t3

def rot_esf(fr, fth, fph):
    cr = (1/(r*sp.sin(theta)))*(sp.diff(sp.sin(theta)*fph, theta) - sp.diff(fth, phi))
    cth = (1/r)*((1/sp.sin(theta))*sp.diff(fr, phi) - sp.diff(r*fph, r))
    cph = (1/r)*(sp.diff(r*fth, r) - sp.diff(fr, theta))
    return (cr, cth, cph)

def lap_esf(f):
    t1 = (1/r**2)*sp.diff(r**2 * sp.diff(f, r), r)
    t2 = (1/(r**2*sp.sin(theta)))*sp.diff(sp.sin(theta)*sp.diff(f, theta), theta)
    t3 = (1/(r**2*sp.sin(theta)**2))*sp.diff(f, phi, 2)
    return t1 + t2 + t3

def line_ds_esf(rt, tht, phit, t):
    dr = sp.diff(rt, t)
    dth = sp.diff(tht, t)
    dph = sp.diff(phit, t)
    return sp.sqrt(dr**2 + rt**2 * dth**2 + rt**2 * sp.sin(tht)**2 * dph**2)

def sph_to_cart_esf(R, TH, PH):
    X = R * np.sin(TH) * np.cos(PH)
    Y = R * np.sin(TH) * np.sin(PH)
    Z = R * np.cos(TH)
    return X, Y, Z

esf_sys = CoordSystem(
    name="Esféricas",
    sym_vars=[r, theta, phi],
    jac2d=r**2 * sp.sin(theta),
    jac3d=r**2 * sp.sin(theta),
    grad=grad_esf,
    div=div_esf,
    rot=rot_esf,
    lap=lap_esf,
    line_ds=line_ds_esf,
    var_buttons=[("r","r"), ("θ","theta"), ("φ","phi"), ("t","t"), ("π","pi")],
    sph_to_cart=sph_to_cart_esf,
    color=C_GOLD
)

# Diccionario de sistemas disponibles
SYSTEMS = {
    "Cartesianas": cart_sys,
    "Cilíndricas": cil_sys,
    "Esféricas": esf_sys
}

# ----------------------------------------------------------------------
# FUNCIONES DE CÁLCULO DE INTEGRALES (usando el sistema activo)
# ----------------------------------------------------------------------
def calc_indefinida(system, expr_str, var_str):
    f = parse(expr_str)
    var = sp.Symbol(var_str)
    res = sp.integrate(f, var)
    return fmt(res) + "  +  C"

def calc_definida(system, expr_str, var_str, a_str, b_str):
    f = parse(expr_str)
    var = sp.Symbol(var_str)
    a, b = parse(a_str), parse(b_str)
    res = sp.integrate(f, (var, a, b))
    try:
        simp = sp.simplify(res)
        val = float(simp.evalf())
        s_exact = str(simp)
        s_dec = f"{val:.6g}"
        if s_exact.replace(" ", "") == s_dec.replace(" ", ""):
            return s_dec
        return f"{s_exact}  ≈  {s_dec}"
    except (ValueError, TypeError):
        return fmt(res)

def calc_doble(system, expr_str, a1, b1, a2, b2):
    f = parse(expr_str)
    integrando = f * system.jacobian2d
    vars_list = system.sym_vars[:2]
    a1_expr, b1_expr = parse(a1), parse(b1)
    a2_expr, b2_expr = parse(a2), parse(b2)
    res = sp.integrate(integrando,
                       (vars_list[0], a1_expr, b1_expr),
                       (vars_list[1], a2_expr, b2_expr))
    try:
        val = float(res.evalf())
        return f"{sp.simplify(res)}  ≈  {val:.6g}"
    except (ValueError, TypeError):
        return fmt(res)

def calc_triple(system, expr_str, a1, b1, a2, b2, a3, b3):
    f = parse(expr_str)
    integrando = f * system.jacobian3d
    vars_list = system.sym_vars[:3]
    a1_expr, b1_expr = parse(a1), parse(b1)
    a2_expr, b2_expr = parse(a2), parse(b2)
    a3_expr, b3_expr = parse(a3), parse(b3)
    res = sp.integrate(integrando,
                       (vars_list[0], a1_expr, b1_expr),
                       (vars_list[1], a2_expr, b2_expr),
                       (vars_list[2], a3_expr, b3_expr))
    try:
        val = float(res.evalf())
        return f"{sp.simplify(res)}  ≈  {val:.6g}"
    except (ValueError, TypeError):
        return fmt(res)

def calc_linea(system, expr_str, param_funcs, a_str, b_str):
    f = parse(expr_str)
    subs_dict = {}
    for var in system.sym_vars:
        var_name = str(var)
        if var_name in param_funcs:
            subs_dict[var] = parse(param_funcs[var_name])
    f_t = f.subs(subs_dict)
    if system.name == "Cartesianas":
        rx = parse(param_funcs.get('x', 't'))
        ry = parse(param_funcs.get('y', 't'))
        ds = sp.sqrt(sp.diff(rx, t)**2 + sp.diff(ry, t)**2)
    elif system.name == "Cilíndricas":
        rt = parse(param_funcs.get('r', '1'))
        tht = parse(param_funcs.get('theta', 't'))
        ds = sp.sqrt(sp.diff(rt, t)**2 + rt**2 * sp.diff(tht, t)**2)
    else:  # Esféricas
        rt = parse(param_funcs.get('r', '1'))
        tht = parse(param_funcs.get('theta', 'pi/2'))
        phit = parse(param_funcs.get('phi', 't'))
        ds = sp.sqrt(sp.diff(rt, t)**2 + rt**2 * sp.diff(tht, t)**2 + rt**2 * sp.sin(tht)**2 * sp.diff(phit, t)**2)
    integrando = sp.simplify(f_t * ds)
    a, b = parse(a_str), parse(b_str)
    res = sp.integrate(integrando, (t, a, b))
    try:
        val = float(res.evalf())
        return f"{sp.simplify(res)}  ≈  {val:.6g}"
    except (ValueError, TypeError):
        return fmt(res)

def calc_flujo(system, param_funcs, field_funcs, a_str, b_str):
    if system.name == "Cartesianas":
        rx = parse(param_funcs.get('x', 'cos(t)'))
        ry = parse(param_funcs.get('y', 'sin(t)'))
        Fx = parse(field_funcs.get('Fx', '-y'))
        Fy = parse(field_funcs.get('Fy', 'x'))
        Fx_t = Fx.subs({x: rx, y: ry})
        Fy_t = Fy.subs({x: rx, y: ry})
        drx = sp.diff(rx, t)
        dry = sp.diff(ry, t)
        integrando = sp.simplify(Fx_t * dry - Fy_t * drx)
    elif system.name == "Cilíndricas":
        rt = parse(param_funcs.get('r', '1'))
        tht = parse(param_funcs.get('theta', 't'))
        Fr = parse(field_funcs.get('Fr', '-r*sin(theta)'))
        Fth = parse(field_funcs.get('Fth', 'r*cos(theta)'))
        Fr_t = Fr.subs({r: rt, theta: tht})
        Fth_t = Fth.subs({r: rt, theta: tht})
        dr = sp.diff(rt, t)
        dth = sp.diff(tht, t)
        integrando = sp.simplify(Fr_t * rt * dth - Fth_t * dr)
    else:  # Esféricas
        rt = parse(param_funcs.get('r', '1'))
        tht = parse(param_funcs.get('theta', 'pi/2'))
        phit = parse(param_funcs.get('phi', 't'))
        Fr = parse(field_funcs.get('Fr', 'r'))
        Fth = parse(field_funcs.get('Fth', '0'))
        Fph = parse(field_funcs.get('Fphi', '0'))
        Fr_t = Fr.subs({r: rt, theta: tht, phi: phit})
        Fth_t = Fth.subs({r: rt, theta: tht, phi: phit})
        Fph_t = Fph.subs({r: rt, theta: tht, phi: phit})
        dr = sp.diff(rt, t)
        dth = sp.diff(tht, t)
        dph = sp.diff(phit, t)
        integrando = sp.simplify(Fr_t * dr + rt * Fth_t * dth + rt * sp.sin(tht) * Fph_t * dph)
    a, b = parse(a_str), parse(b_str)
    res = sp.integrate(integrando, (t, a, b))
    try:
        val = float(res.evalf())
        return f"{sp.simplify(res)}  ≈  {val:.6g}"
    except (ValueError, TypeError):
        return fmt(res)

def calc_volumen(system, fsup_str, finf_str, a1, b1, a2, b2, a3=None, b3=None):
    fsup = parse(fsup_str)
    finf = parse(finf_str)
    dif = sp.simplify(fsup - finf)
    integrando = dif * system.jacobian3d
    vars_list = system.sym_vars[:3]
    a1_e, b1_e = parse(a1), parse(b1)
    a2_e, b2_e = parse(a2), parse(b2)
    if system.name == "Esféricas" and a3 is not None:
        a3_e, b3_e = parse(a3), parse(b3)
        res = sp.integrate(integrando, (vars_list[0], a1_e, b1_e),
                                        (vars_list[1], a2_e, b2_e),
                                        (vars_list[2], a3_e, b3_e))
    else:
        res = sp.integrate(integrando, (vars_list[0], a1_e, b1_e),
                                        (vars_list[1], a2_e, b2_e))
    try:
        val = float(res.evalf())
        return f"{sp.simplify(res)}  ≈  {val:.6g}"
    except (ValueError, TypeError):
        return fmt(res)
# ======================================================================
# CLASE PRINCIPAL SuiteCalculoUnificada
# ======================================================================

class SuiteCalculoUnificada(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Suite de Cálculo Vectorial — Todos los Sistemas de Coordenadas")
        self.configure(fg_color=C_BG)

        # Configuración responsiva de la ventana
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(int(sw * 0.92), 1440)
        h = min(int(sh * 0.92), 940)
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.minsize(900, 600)

        # Sistema actual
        self.current_system_name = "Cartesianas"
        self.current_system = SYSTEMS[self.current_system_name]

        # Estado para la pestaña de integrales
        self._int_modo = "Indefinida"
        self._int_p_entries = {}
        self._int_busy = False
        self._int_fig = None
        self._int_canvas = None

        # Configurar expansión responsiva de filas/columnas de la ventana principal
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Construir UI
        self._build_header()
        self._build_coord_selector()
        self._build_tabs()

        # Inicializar pestaña de integrales por defecto
        self._int_set_modo("Indefinida")

    # ------------------------------------------------------------------
    # HEADER Y SELECTOR DE COORDENADAS
    # ------------------------------------------------------------------
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=0, height=52)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(hdr, text="  ∑  Suite de Cálculo Vectorial — Unificada",
                     font=ctk.CTkFont("Consolas", 17, "bold"), text_color=C_TEXT
                     ).grid(row=0, column=0, padx=18, pady=13, sticky="w")
        ctk.CTkLabel(hdr, text="100% local · SymPy · Sin internet · Coordenadas: Cartesianas, Cilíndricas, Esféricas",
                     font=ctk.CTkFont("Consolas", 11), text_color=C_MUTED
                     ).grid(row=0, column=1, padx=10, sticky="w")

    def _build_coord_selector(self):
        frame = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=0, height=40)
        frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))
        frame.grid_columnconfigure(0, weight=1)
        selector = ctk.CTkSegmentedButton(frame, values=list(SYSTEMS.keys()),
                                          command=self._change_system,
                                          font=ctk.CTkFont("Consolas", 12, "bold"),
                                          selected_color=self.current_system.color)
        selector.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        selector.set(self.current_system_name)
        self.coord_selector = selector

    def _change_system(self, sys_name):
        self.current_system_name = sys_name
        self.current_system = SYSTEMS[sys_name]
        self.coord_selector.configure(selected_color=self.current_system.color)
        # Reconstruir pestañas para actualizar contenido (botones, variables)
        self._rebuild_tabs()
        # Actualizar el modo actual en integrales
        self._int_set_modo(self._int_modo)

    def _rebuild_tabs(self):
        if hasattr(self, '_tabs'):
            self._tabs.destroy()
        self._build_tabs()

    # ------------------------------------------------------------------
    # CONSTRUCCIÓN DE LAS 4 PESTAÑAS PRINCIPALES
    # ------------------------------------------------------------------
    def _build_tabs(self):
        self._tabs = ctk.CTkTabview(self, fg_color=C_PANEL, corner_radius=8)
        self._tabs.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        self._tabs._segmented_button.configure(font=ctk.CTkFont("Consolas", 13, "bold"))

        tab_int = self._tabs.add("∫  Integrales")
        tab_der = self._tabs.add("∂  Derivadas Mixtas")
        tab_vec = self._tabs.add("×  Prod. Vectoriales")
        tab_nab = self._tabs.add("∇  Operadores Nabla")

        for tab in (tab_int, tab_der, tab_vec, tab_nab):
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)

        self._build_tab_integrales(tab_int)
        self._build_tab_derivadas(tab_der)
        self._build_tab_vectoriales(tab_vec)
        self._build_tab_nabla(tab_nab)

    # ==================================================================
    # PESTAÑA 1: INTEGRALES (7 MODOS CON PARÁMETROS DINÁMICOS)
    # ==================================================================
    def _build_tab_integrales(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=0)
        parent.grid_columnconfigure(1, weight=1)

        # Panel izquierdo: modos, parámetros y ejemplos
        left = ctk.CTkScrollableFrame(parent, width=310, fg_color=C_CARD, corner_radius=10,
                                      scrollbar_button_color=C_BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 3), pady=4)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="TIPO DE INTEGRAL",
                     font=ctk.CTkFont("Consolas", 10), text_color=C_MUTED
                     ).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        # Definición de los 7 modos de integral (adaptados dinámicamente al sistema actual)
        self._int_modes = self._get_int_modes_dict()
        self._int_tab_btns = {}

        row = 1
        for nombre, cfg in self._int_modes.items():
            btn = ctk.CTkButton(left, text=f"  {cfg['icono']}  {nombre}", height=36, anchor="w",
                                font=ctk.CTkFont("Consolas", 12), fg_color="transparent",
                                hover_color=C_BORDER, border_width=0, corner_radius=8,
                                command=lambda m=nombre: self._int_set_modo(m))
            btn.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
            self._int_tab_btns[nombre] = btn
            row += 1

        sep = ctk.CTkFrame(left, height=1, fg_color=C_BORDER)
        sep.grid(row=row, column=0, padx=14, pady=10, sticky="ew")
        row += 1

        self._int_desc_lbl = ctk.CTkLabel(left, text="", font=ctk.CTkFont("Consolas", 12),
                                          text_color=self.current_system.color, wraplength=280)
        self._int_desc_lbl.grid(row=row, column=0, padx=14, pady=(0, 3), sticky="w")
        row += 1

        self._int_nota_lbl = ctk.CTkLabel(left, text="", font=ctk.CTkFont("Consolas", 10),
                                          text_color=C_MUTED, wraplength=280, justify="left")
        self._int_nota_lbl.grid(row=row, column=0, padx=14, pady=(0, 6), sticky="w")
        row += 1

        self._int_params_box = ctk.CTkFrame(left, fg_color="transparent")
        self._int_params_box.grid(row=row, column=0, padx=8, sticky="ew")
        row += 1

        sep2 = ctk.CTkFrame(left, height=1, fg_color=C_BORDER)
        sep2.grid(row=row, column=0, padx=14, pady=8, sticky="ew")
        row += 1

        ctk.CTkLabel(left, text="EJEMPLOS RÁPIDOS",
                     font=ctk.CTkFont("Consolas", 10), text_color=C_MUTED
                     ).grid(row=row, column=0, padx=14, pady=(0, 5), sticky="w")
        row += 1

        self._int_ej_box = ctk.CTkFrame(left, fg_color="transparent")
        self._int_ej_box.grid(row=row, column=0, padx=8, sticky="ew")

        # Panel derecho: entrada, resultado, gráfica
        right = ctk.CTkFrame(parent, fg_color=C_BG, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 4), pady=4)
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        inp = ctk.CTkFrame(right, fg_color=C_CARD, corner_radius=10,
                           border_width=1, border_color=C_BORDER)
        inp.grid(row=0, column=0, padx=4, pady=(4, 5), sticky="ew")
        inp.grid_columnconfigure(0, weight=1)

        self._int_expr_lbl = ctk.CTkLabel(inp, text="FUNCIÓN / EXPRESIÓN",
                                          font=ctk.CTkFont("Consolas", 10), text_color=C_MUTED)
        self._int_expr_lbl.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

        self._int_expr = ctk.CTkEntry(inp, height=44, font=ctk.CTkFont("Consolas", 17),
                                      placeholder_text="Ej: r**2 + sin(theta)",
                                      fg_color="#080c14", border_color=C_BORDER, text_color=C_TEXT)
        self._int_expr.grid(row=1, column=0, padx=12, pady=(0, 4), sticky="ew")
        self._int_expr.bind("<Return>", lambda e: self._int_calcular())

        kb = ctk.CTkFrame(inp, fg_color="transparent")
        kb.grid(row=2, column=0, padx=12, pady=(4, 10), sticky="w")
        self._build_kb(kb, self._int_expr)

        acc = ctk.CTkFrame(right, fg_color="transparent")
        acc.grid(row=1, column=0, padx=4, pady=(0, 5), sticky="ew")
        for c in range(3):
            acc.grid_columnconfigure(c, weight=1)

        self._int_btn_calc = ctk.CTkButton(acc, text="▶  Calcular", height=40,
                                           font=ctk.CTkFont("Consolas", 13, "bold"),
                                           fg_color=C_ACCENT, hover_color="#1a7a52",
                                           command=self._int_calcular)
        self._int_btn_calc.grid(row=0, column=0, padx=4, sticky="ew")

        ctk.CTkButton(acc, text="Graficar", height=40,
                      font=ctk.CTkFont("Consolas", 13),
                      fg_color=C_ACCENT2, hover_color="#1d4ed8",
                      command=self._int_graficar).grid(row=0, column=1, padx=4, sticky="ew")

        ctk.CTkButton(acc, text="🗑  Limpiar", height=40,
                      font=ctk.CTkFont("Consolas", 13),
                      fg_color=C_DANGER, hover_color="#b91c1c",
                      command=self._int_limpiar).grid(row=0, column=2, padx=4, sticky="ew")

        self._int_res_frame = ctk.CTkFrame(right, fg_color=C_CARD, corner_radius=10, height=52)
        self._int_res_frame.grid(row=2, column=0, padx=4, pady=(0, 5), sticky="ew")
        self._int_res_frame.grid_propagate(False)
        self._int_res_frame.grid_columnconfigure(0, weight=1)
        self._int_res_lbl = ctk.CTkLabel(self._int_res_frame, text="El resultado aparecerá aquí",
                                         font=ctk.CTkFont("Consolas", 13), text_color=C_MUTED, wraplength=860)
        self._int_res_lbl.grid(row=0, column=0, padx=14, pady=14)

        gcard = ctk.CTkFrame(right, fg_color=C_CARD, corner_radius=10,
                             border_width=1, border_color=C_BORDER)
        gcard.grid(row=3, column=0, padx=4, pady=(0, 4), sticky="nsew")
        gcard.grid_rowconfigure(0, weight=1)
        gcard.grid_columnconfigure(0, weight=1)

        self._int_fig = plt.Figure(facecolor=C_BG)
        self._int_canvas = FigureCanvasTkAgg(self._int_fig, master=gcard)
        self._int_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        self._int_canvas.get_tk_widget().configure(bg=C_BG, highlightthickness=0)

        ax0 = self._int_fig.add_subplot(111)
        ax0.set_facecolor(C_PANEL)
        ax0.text(0.5, 0.5, "Ingresa una expresión y pulsa Graficar",
                 ha="center", va="center", color=C_MUTED, fontsize=12, transform=ax0.transAxes)
        ax0.axis("off")
        self._int_canvas.draw()

    def _get_int_modes_dict(self):
        """Construye diccionario de modos de integral adaptado al sistema actual."""
        sys = self.current_system
        var_names = sys.var_names
        # Parámetros según sistema
        if sys.name == "Cartesianas":
            params_2d = [{"id":"var1","label":"1ª variable","default":var_names[0]},
                         {"id":"from1","label":"Límite inf","default":"0"},
                         {"id":"to1","label":"Límite sup","default":"1"},
                         {"id":"var2","label":"2ª variable","default":var_names[1]},
                         {"id":"from2","label":"Límite inf","default":"0"},
                         {"id":"to2","label":"Límite sup","default":"1"}]
            params_3d = [{"id":"var1","label":"1ª variable","default":var_names[0]},
                         {"id":"from1","label":"Límite inf","default":"0"},
                         {"id":"to1","label":"Límite sup","default":"1"},
                         {"id":"var2","label":"2ª variable","default":var_names[1]},
                         {"id":"from2","label":"Límite inf","default":"0"},
                         {"id":"to2","label":"Límite sup","default":"1"},
                         {"id":"var3","label":"3ª variable","default":var_names[2]},
                         {"id":"from3","label":"Límite inf","default":"0"},
                         {"id":"to3","label":"Límite sup","default":"1"}]
            params_linea = [{"id":"from","label":"t inicial","default":"0"},
                            {"id":"to","label":"t final","default":"2*pi"},
                            {"id":"rx","label":"Curva x(t)","default":"cos(t)"},
                            {"id":"ry","label":"Curva y(t)","default":"sin(t)"}]
            params_flujo = [{"id":"from","label":"t inicial","default":"0"},
                            {"id":"to","label":"t final","default":"2*pi"},
                            {"id":"rx","label":"Curva x(t)","default":"cos(t)"},
                            {"id":"ry","label":"Curva y(t)","default":"sin(t)"},
                            {"id":"Fx","label":"Campo Fx","default":"-y"},
                            {"id":"Fy","label":"Campo Fy","default":"x"}]
            params_vol = [{"id":"gsup","label":"Superficie sup","default":"1-x**2-y**2"},
                          {"id":"ginf","label":"Superficie inf","default":"0"},
                          {"id":"from1","label":"Límite inf x","default":"-1"},
                          {"id":"to1","label":"Límite sup x","default":"1"},
                          {"id":"from2","label":"Límite inf y","default":"-1"},
                          {"id":"to2","label":"Límite sup y","default":"1"}]
        elif sys.name == "Cilíndricas":
            params_2d = [{"id":"from1","label":"r inferior","default":"0"},
                         {"id":"to1","label":"r superior","default":"1"},
                         {"id":"from2","label":"θ inferior","default":"0"},
                         {"id":"to2","label":"θ superior","default":"2*pi"}]
            params_3d = [{"id":"from1","label":"r inferior","default":"0"},
                         {"id":"to1","label":"r superior","default":"1"},
                         {"id":"from2","label":"θ inferior","default":"0"},
                         {"id":"to2","label":"θ superior","default":"2*pi"},
                         {"id":"from3","label":"z inferior","default":"0"},
                         {"id":"to3","label":"z superior","default":"1"}]
            params_linea = [{"id":"from","label":"t inicial","default":"0"},
                            {"id":"to","label":"t final","default":"2*pi"},
                            {"id":"rt","label":"Curva r(t)","default":"1"},
                            {"id":"tht","label":"Curva θ(t)","default":"t"}]
            params_flujo = [{"id":"from","label":"t inicial","default":"0"},
                            {"id":"to","label":"t final","default":"2*pi"},
                            {"id":"rt","label":"Curva r(t)","default":"1"},
                            {"id":"tht","label":"Curva θ(t)","default":"t"},
                            {"id":"Fr","label":"Campo Fr","default":"-r*sin(theta)"},
                            {"id":"Fth","label":"Campo Fθ","default":"r*cos(theta)"}]
            params_vol = [{"id":"gsup","label":"Superficie sup f(r,θ)","default":"1 - r**2"},
                          {"id":"ginf","label":"Superficie inf g(r,θ)","default":"0"},
                          {"id":"from1","label":"r inferior","default":"0"},
                          {"id":"to1","label":"r superior","default":"1"},
                          {"id":"from2","label":"θ inferior","default":"0"},
                          {"id":"to2","label":"θ superior","default":"2*pi"}]
        else:  # Esféricas
            params_2d = [{"id":"from1","label":"r inferior","default":"0"},
                         {"id":"to1","label":"r superior","default":"1"},
                         {"id":"from2","label":"θ inferior","default":"0"},
                         {"id":"to2","label":"θ superior","default":"pi"}]
            params_3d = [{"id":"from1","label":"r inferior","default":"0"},
                         {"id":"to1","label":"r superior","default":"1"},
                         {"id":"from2","label":"θ inferior","default":"0"},
                         {"id":"to2","label":"θ superior","default":"pi"},
                         {"id":"from3","label":"φ inferior","default":"0"},
                         {"id":"to3","label":"φ superior","default":"2*pi"}]
            params_linea = [{"id":"from","label":"t inicial","default":"0"},
                            {"id":"to","label":"t final","default":"2*pi"},
                            {"id":"rt","label":"Curva r(t)","default":"1"},
                            {"id":"tht","label":"Curva θ(t)","default":"pi/2"},
                            {"id":"phit","label":"Curva φ(t)","default":"t"}]
            params_flujo = [{"id":"from","label":"t inicial","default":"0"},
                            {"id":"to","label":"t final","default":"2*pi"},
                            {"id":"rt","label":"Curva r(t)","default":"1"},
                            {"id":"tht","label":"Curva θ(t)","default":"pi/2"},
                            {"id":"phit","label":"Curva φ(t)","default":"t"},
                            {"id":"Fr","label":"Campo Fr","default":"r"},
                            {"id":"Fth","label":"Campo Fθ","default":"0"},
                            {"id":"Fphi","label":"Campo Fφ","default":"0"}]
            params_vol = [{"id":"gsup","label":"Superficie sup f(r,θ,φ)","default":"1"},
                          {"id":"ginf","label":"Superficie inf g(r,θ,φ)","default":"0"},
                          {"id":"from1","label":"r inferior","default":"0"},
                          {"id":"to1","label":"r superior","default":"1"},
                          {"id":"from2","label":"θ inferior","default":"0"},
                          {"id":"to2","label":"θ superior","default":"pi"},
                          {"id":"from3","label":"φ inferior","default":"0"},
                          {"id":"to3","label":"φ superior","default":"2*pi"}]

        # Ejemplos comunes (varían según sistema)
        ejemplos_indef = ["r**2", "sin(theta)", "r*exp(-r)", "1/r"]
        ejemplos_def = ["r**2", "r*sin(theta)", "exp(-r**2)"]
        ejemplos_doble = ["r**2", "r*sin(theta)", "r**2*cos(theta)"] if sys.name!="Cartesianas" else ["x*y", "x**2+y**2"]
        ejemplos_triple = ["1", "r**2", "r*cos(theta)"] if sys.name!="Cartesianas" else ["x*y*z", "x**2+y**2+z**2"]
        ejemplos_linea = ["r**2", "r*sin(theta)", "1"]
        ejemplos_flujo = ["-r*sin(theta)", "r*cos(theta)", "r**2"]
        ejemplos_vol = ["1-r**2", "4-r**2", "cos(r)*cos(theta)"]

        return {
            "Indefinida": {"icono":"∫", "desc":"∫ f d(var)", "params":[{"id":"var","label":"Variable","default":sys.var_names[0]}],
                           "ejemplos":ejemplos_indef},
            "Definida": {"icono":"∫ₐᵇ", "desc":"∫ₐᵇ f d(var)", "params":[{"id":"var","label":"Variable","default":sys.var_names[0]},
                                                                         {"id":"from","label":"Límite inferior","default":"0"},
                                                                         {"id":"to","label":"Límite superior","default":"1"}],
                         "ejemplos":ejemplos_def},
            "Doble": {"icono":"∬", "desc":"∬ f dA (Jacobiano automático)", "params":params_2d,
                      "nota":f"Jacobiano {sys.jacobian2d} incluido automáticamente. Introduce solo f.",
                      "ejemplos":ejemplos_doble},
            "Triple": {"icono":"∭", "desc":"∭ f dV (Jacobiano automático)", "params":params_3d,
                       "nota":f"Jacobiano {sys.jacobian3d} incluido automáticamente.",
                       "ejemplos":ejemplos_triple},
            "Línea": {"icono":"∫C", "desc":"∫C f ds (curva paramétrica)", "params":params_linea,
                      "nota":"ds = fórmula del sistema. Parametriza la curva.",
                      "ejemplos":ejemplos_linea},
            "Flujo": {"icono":"∫∫F", "desc":"∫C F·n ds (flujo)", "params":params_flujo,
                      "nota":"Campo vectorial en la base del sistema.",
                      "ejemplos":ejemplos_flujo},
            "Volumen": {"icono":"Vol", "desc":"V = ∬ (f_sup - f_inf) dA", "params":params_vol,
                        "nota":"Volumen entre superficies. Jacobiano automático.",
                        "ejemplos":ejemplos_vol},
        }

    def _int_set_modo(self, modo):
        self._int_modo = modo
        cfg = self._int_modes[modo]
        # Actualizar estilo de botones
        for m, btn in self._int_tab_btns.items():
            if m == modo:
                btn.configure(fg_color=C_PANEL, text_color=self.current_system.color,
                              border_width=1, border_color=self.current_system.color)
            else:
                btn.configure(fg_color="transparent", text_color=C_TEXT, border_width=0)
        self._int_desc_lbl.configure(text=cfg["desc"])
        self._int_nota_lbl.configure(text=cfg.get("nota", ""))
        self._int_expr_lbl.configure(text="FUNCIÓN / EXPRESIÓN")

        # Limpiar y reconstruir parámetros
        for w in self._int_params_box.winfo_children():
            w.destroy()
        self._int_p_entries.clear()
        self._int_params_box.grid_columnconfigure(0, weight=1)

        if cfg["params"]:
            ctk.CTkLabel(self._int_params_box, text="PARÁMETROS",
                         font=ctk.CTkFont("Consolas", 10), text_color=C_MUTED
                         ).grid(row=0, column=0, padx=4, pady=(6, 4), sticky="w")
            for i, prm in enumerate(cfg["params"]):
                ctk.CTkLabel(self._int_params_box, text=prm["label"],
                             font=ctk.CTkFont("Consolas", 10), text_color=C_MUTED
                             ).grid(row=i*2+1, column=0, padx=4, pady=(2, 0), sticky="w")
                ent = ctk.CTkEntry(self._int_params_box, height=26,
                                   font=ctk.CTkFont("Consolas", 11),
                                   fg_color="#080c14", border_color=C_BORDER, text_color=C_TEXT)
                ent.insert(0, prm["default"])
                ent.grid(row=i*2+2, column=0, padx=4, pady=(0, 2), sticky="ew")
                self._int_p_entries[prm["id"]] = ent

        # Ejemplos
        for w in self._int_ej_box.winfo_children():
            w.destroy()
        for ex in cfg["ejemplos"]:
            ctk.CTkButton(self._int_ej_box, text=ex, height=26,
                          font=ctk.CTkFont("Consolas", 11),
                          fg_color="#111827", hover_color=C_CARD,
                          border_width=1, border_color=C_BORDER, corner_radius=14,
                          command=lambda e=ex: self._int_set_expr(e)).pack(side="top", padx=3, pady=2, fill="x")

    def _int_set_expr(self, expr):
        self._int_expr.delete(0, "end")
        self._int_expr.insert(0, expr)

    def _int_get_params(self):
        return {pid: e.get().strip() for pid, e in self._int_p_entries.items()}

    def _int_calcular(self):
        if self._int_busy:
            return
        expr = self._int_expr.get().strip()
        if not expr:
            return
        p = self._int_get_params()
        self._int_busy = True
        self._int_btn_calc.configure(state="disabled", text="⏳ Calculando...")
        self._int_res_lbl.configure(text="Procesando con SymPy...", text_color=C_MUTED)
        self._int_res_frame.configure(fg_color=C_CARD)
        threading.Thread(target=self._int_calc_thread, args=(expr, p), daemon=True).start()

    def _int_calc_thread(self, expr, p):
        try:
            sys = self.current_system
            m = self._int_modo
            if m == "Indefinida":
                res = calc_indefinida(sys, expr, p.get("var", sys.var_names[0]))
            elif m == "Definida":
                res = calc_definida(sys, expr, p.get("var", sys.var_names[0]),
                                    p.get("from","0"), p.get("to","1"))
            elif m == "Doble":
                res = calc_doble(sys, expr,
                                 p.get("from1","0"), p.get("to1","1"),
                                 p.get("from2","0"), p.get("to2","2*pi"))
            elif m == "Triple":
                res = calc_triple(sys, expr,
                                  p.get("from1","0"), p.get("to1","1"),
                                  p.get("from2","0"), p.get("to2","2*pi"),
                                  p.get("from3","0"), p.get("to3","1"))
            elif m == "Línea":
                # Construir diccionario de funciones paramétricas
                param_funcs = {}
                if sys.name == "Cartesianas":
                    param_funcs['x'] = p.get("rx","cos(t)")
                    param_funcs['y'] = p.get("ry","sin(t)")
                elif sys.name == "Cilíndricas":
                    param_funcs['r'] = p.get("rt","1")
                    param_funcs['theta'] = p.get("tht","t")
                else:
                    param_funcs['r'] = p.get("rt","1")
                    param_funcs['theta'] = p.get("tht","pi/2")
                    param_funcs['phi'] = p.get("phit","t")
                res = calc_linea(sys, expr, param_funcs, p.get("from","0"), p.get("to","2*pi"))
            elif m == "Flujo":
                param_funcs = {}
                field_funcs = {}
                if sys.name == "Cartesianas":
                    param_funcs['x'] = p.get("rx","cos(t)")
                    param_funcs['y'] = p.get("ry","sin(t)")
                    field_funcs['Fx'] = p.get("Fx","-y")
                    field_funcs['Fy'] = p.get("Fy","x")
                elif sys.name == "Cilíndricas":
                    param_funcs['r'] = p.get("rt","1")
                    param_funcs['theta'] = p.get("tht","t")
                    field_funcs['Fr'] = p.get("Fr","-r*sin(theta)")
                    field_funcs['Fth'] = p.get("Fth","r*cos(theta)")
                else:
                    param_funcs['r'] = p.get("rt","1")
                    param_funcs['theta'] = p.get("tht","pi/2")
                    param_funcs['phi'] = p.get("phit","t")
                    field_funcs['Fr'] = p.get("Fr","r")
                    field_funcs['Fth'] = p.get("Fth","0")
                    field_funcs['Fphi'] = p.get("Fphi","0")
                res = calc_flujo(sys, param_funcs, field_funcs, p.get("from","0"), p.get("to","2*pi"))
            elif m == "Volumen":
                if sys.name == "Esféricas":
                    res = calc_volumen(sys, p.get("gsup","1"), p.get("ginf","0"),
                                       p.get("from1","0"), p.get("to1","1"),
                                       p.get("from2","0"), p.get("to2","pi"),
                                       p.get("from3","0"), p.get("to3","2*pi"))
                else:
                    res = calc_volumen(sys, p.get("gsup","1-x**2-y**2"), p.get("ginf","0"),
                                       p.get("from1","-1"), p.get("to1","1"),
                                       p.get("from2","-1"), p.get("to2","1"))
            else:
                res = "Modo desconocido"
            self.after(0, self._int_show_result, res, True)
        except (ValueError, TypeError, AttributeError) as exc:
            # Errores esperados: expresión inválida, tipo incorrecto, atributo faltante
            self.after(0, self._int_show_result, str(exc), False)
        except Exception as exc:
            # Error inesperado: registrar tipo y mensaje completo para diagnóstico
            traceback.print_exc()
            self.after(0, self._int_show_result, f"{type(exc).__name__}: {exc}", False)

    def _int_show_result(self, text, ok):
        self._int_res_lbl.configure(text=("  =  " if ok else "  ⚠  ") + text, text_color=C_TEXT)
        self._int_res_frame.configure(fg_color=C_OK_BG if ok else C_ERR_BG)
        self._int_btn_calc.configure(state="normal", text="▶  Calcular")
        self._int_busy = False

    def _int_limpiar(self):
        self._int_expr.delete(0, "end")
        self._int_res_lbl.configure(text="El resultado aparecerá aquí", text_color=C_MUTED)
        self._int_res_frame.configure(fg_color=C_CARD)

    def _int_graficar(self):
        expr = self._int_expr.get().strip()
        if not expr:
            messagebox.showwarning("Aviso", "Ingresa una expresión primero.")
            return
        p = self._int_get_params()
        def _t():
            self._graficar_integral(self._int_modo, expr, p, self._int_fig)
            self.after(0, self._int_canvas.draw)
        threading.Thread(target=_t, daemon=True).start()

    def _graficar_integral(self, modo, expr_str, params, fig):
        fig.clf()
        fig.patch.set_facecolor(C_BG)
        sys = self.current_system
        p = params

        try:
            # ---------- INDEFINIDA / DEFINIDA (curva 2D) ----------
            if modo in ("Indefinida", "Definida"):
                ax = fig.add_subplot(111)
                self._ax_style(ax)
                f_sym = parse(expr_str)
                # Determinar variable de integración
                var_name = p.get("var", sys.var_names[0])
                var_sym = sp.Symbol(var_name)
                f_np = to_numpy(f_sym, var_sym)

                # Rango de graficación
                if modo == "Definida":
                    av = float(parse(p.get("from", "0")).evalf())
                    bv = float(parse(p.get("to", "1")).evalf())
                    mg = max((bv - av) * 0.4, 0.5)
                    x0, x1 = max(av - mg, 0), bv + mg
                else:
                    x0, x1 = -5.0, 5.0

                xs = np.linspace(x0, x1, 900)
                ys = safe_eval1(f_np, xs)
                ax.plot(xs, ys, color=C_ACCENT2, lw=2, label=f"f({var_name})")

                if modo == "Definida":
                    av = float(parse(p.get("from", "0")).evalf())
                    bv = float(parse(p.get("to", "1")).evalf())
                    xf = np.linspace(av, bv, 600)
                    yf = safe_eval1(f_np, xf)
                    ax.fill_between(xf, yf, alpha=0.3, color=C_ACCENT)
                    ax.axvline(av, color=C_ACCENT, lw=1.2, ls="--", alpha=0.7)
                    ax.axvline(bv, color=C_ACCENT, lw=1.2, ls="--", alpha=0.7)

                # Antiderivada (solo si se puede integrar)
                try:
                    F_sym = sp.integrate(f_sym, var_sym)
                    F_np = to_numpy(F_sym, var_sym)
                    Fs = safe_eval1(F_np, xs)
                    ax.plot(xs, Fs, color="#f59e0b", lw=1.5, ls="--", alpha=0.75, label=f"F({var_name})")
                except Exception:
                    pass

                ax.axhline(0, color=C_BORDER, lw=0.7)
                ax.axvline(0, color=C_BORDER, lw=0.7)
                ax.legend(facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=9)
                ax.set_title(f"f({var_name}) = {expr_str}", color=C_TEXT, fontsize=10)

            # ---------- DOBLE (superficie 3D en coordenadas del sistema) ----------
            elif modo == "Doble":
                ax = fig.add_subplot(111, projection="3d")
                f_sym = parse(expr_str)
                # Obtener los dos primeros símbolos del sistema
                sym1, sym2 = sys.sym_vars[:2]
                f_np = to_numpy(f_sym, sym1, sym2)

                # Límites de integración
                a1 = float(parse(p.get("from1", "0")).evalf())
                b1 = float(parse(p.get("to1", "1")).evalf())
                a2 = float(parse(p.get("from2", "0")).evalf())
                b2 = float(parse(p.get("to2", "2*pi")).evalf())

                # Malla en coordenadas del sistema
                n1, n2 = 50, 50
                xs1 = np.linspace(a1, b1, n1)
                xs2 = np.linspace(a2, b2, n2)
                U, V = np.meshgrid(xs1, xs2)  # U = sym1, V = sym2
                Z = safe_eval2(f_np, U, V)

                # Convertir a cartesianas si es necesario (para cilíndricas y esféricas)
                if sys.name == "Cartesianas":
                    X, Y = U, V
                elif sys.name == "Cilíndricas":
                    X = U * np.cos(V)
                    Y = U * np.sin(V)
                else:  # Esféricas (doble solo usa r y theta, fijamos phi=0)
                    X = U * np.sin(V) * np.cos(0)
                    Y = U * np.sin(V) * np.sin(0)
                # Z ya es el valor de la función

                surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, edgecolor="none")
                fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
                self._ax3d_style(ax)
                ax.set_title(f"f({sys.var_names[0]}, {sys.var_names[1]}) = {expr_str}", color=C_TEXT, fontsize=9)

            # ---------- LÍNEA (curva paramétrica coloreada por f) ----------
            elif modo == "Línea":
                ax = fig.add_subplot(111)
                self._ax_style(ax)
                a = float(parse(p.get("from", "0")).evalf())
                b = float(parse(p.get("to", "2*pi")).evalf())
                # Obtener funciones paramétricas según sistema
                if sys.name == "Cartesianas":
                    x_t = parse(p.get("rx", "cos(t)"))
                    y_t = parse(p.get("ry", "sin(t)"))
                    x_np = to_numpy(x_t, t)
                    y_np = to_numpy(y_t, t)
                elif sys.name == "Cilíndricas":
                    r_t = parse(p.get("rt", "1"))
                    th_t = parse(p.get("tht", "t"))
                    r_np = to_numpy(r_t, t)
                    th_np = to_numpy(th_t, t)
                    # Convertir a cartesianas para graficar
                    def make_xy(tt):
                        rr = r_np(tt)
                        thth = th_np(tt)
                        return rr * np.cos(thth), rr * np.sin(thth)
                    x_np = lambda tt: make_xy(tt)[0]
                    y_np = lambda tt: make_xy(tt)[1]
                else:  # Esféricas: curva 3D, usar proyección 2D? Mejor 3D
                    ax = fig.add_subplot(111, projection="3d")
                    r_t = parse(p.get("rt", "1"))
                    th_t = parse(p.get("tht", "pi/2"))
                    ph_t = parse(p.get("phit", "t"))
                    r_np = to_numpy(r_t, t)
                    th_np = to_numpy(th_t, t)
                    ph_np = to_numpy(ph_t, t)
                    ts = np.linspace(a, b, 400)
                    rs = safe_eval1(r_np, ts)
                    ths = safe_eval1(th_np, ts)
                    phs = safe_eval1(ph_np, ts)
                    X = rs * np.sin(ths) * np.cos(phs)
                    Y = rs * np.sin(ths) * np.sin(phs)
                    Z = rs * np.cos(ths)
                    # Evaluar f sobre la curva
                    subs = {r: r_t, theta: th_t, phi: ph_t}
                    if sys.name == "Esféricas":
                        f_t = parse(expr_str).subs(subs)
                    else:
                        f_t = parse(expr_str).subs({x: x_t, y: y_t})
                    f_np = to_numpy(f_t, t)
                    f_val = safe_eval1(f_np, ts)
                    sc = ax.scatter(X, Y, Z, c=f_val, cmap="viridis", s=5)
                    ax.plot(X, Y, Z, color=C_MUTED, lw=1)
                    fig.colorbar(sc, ax=ax, label="f")
                    self._ax3d_style(ax)
                    ax.set_title(f"Línea en {sys.name} | f={expr_str}", color=C_TEXT)
                    fig.tight_layout()
                    return

                # Para sistemas 2D (Cartesiano y Cilíndrico)
                ts = np.linspace(a, b, 600)
                x_vals = safe_eval1(x_np, ts)
                y_vals = safe_eval1(y_np, ts)

                # Evaluar f sobre la curva
                if sys.name == "Cartesianas":
                    f_t = parse(expr_str).subs({x: x_t, y: y_t})
                else:  # Cilíndricas
                    f_t = parse(expr_str).subs({r: r_t, theta: th_t})
                f_np = to_numpy(f_t, t)
                f_vals = safe_eval1(f_np, ts)

                sc = ax.scatter(x_vals, y_vals, c=f_vals, cmap="viridis", s=5)
                ax.plot(x_vals, y_vals, color=C_MUTED, lw=1)
                ax.plot(x_vals[0], y_vals[0], "o", color=C_ACCENT, ms=8, label="inicio")
                ax.plot(x_vals[-1], y_vals[-1], "s", color=C_DANGER, ms=8, label="fin")
                fig.colorbar(sc, ax=ax, label="f")
                ax.legend(facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT)
                ax.set_title(f"Línea en {sys.name} | f={expr_str}", color=C_TEXT)
                ax.set_aspect("equal", adjustable="datalim")

            # ---------- VOLUMEN (superficie superior e inferior) ----------
            elif modo == "Volumen":
                ax = fig.add_subplot(111, projection="3d")
                gsup = parse(p.get("gsup", "1"))
                ginf = parse(p.get("ginf", "0"))
                # Límites en las dos primeras variables
                a1 = float(parse(p.get("from1", "-1")).evalf())
                b1 = float(parse(p.get("to1", "1")).evalf())
                a2 = float(parse(p.get("from2", "-1")).evalf())
                b2 = float(parse(p.get("to2", "1")).evalf())

                sym1, sym2 = sys.sym_vars[:2]
                fsup_np = to_numpy(gsup, sym1, sym2)
                finf_np = to_numpy(ginf, sym1, sym2)

                xs = np.linspace(a1, b1, 50)
                ys = np.linspace(a2, b2, 50)
                X, Y = np.meshgrid(xs, ys)

                # Convertir a coordenadas cartesianas si es necesario
                if sys.name == "Cartesianas":
                    Xc, Yc = X, Y
                elif sys.name == "Cilíndricas":
                    Xc = X * np.cos(Y)
                    Yc = X * np.sin(Y)
                else:  # Esféricas (fijar phi=0)
                    Xc = X * np.sin(Y) * np.cos(0)
                    Yc = X * np.sin(Y) * np.sin(0)

                Zsup = safe_eval2(fsup_np, X, Y)
                Zinf = safe_eval2(finf_np, X, Y)
                Zsup = np.where(Zsup >= Zinf, Zsup, np.nan)

                ax.plot_surface(Xc, Yc, Zsup, cmap="Blues", alpha=0.75, edgecolor="none")
                ax.plot_surface(Xc, Yc, Zinf, cmap="Oranges", alpha=0.5, edgecolor="none")
                self._ax3d_style(ax)
                ax.set_title(f"Volumen entre superficies en {sys.name}", color=C_TEXT, fontsize=9)

            # ---------- TRIPLE (corte o superficie coloreada) ----------
            elif modo == "Triple":
                ax = fig.add_subplot(111, projection="3d")
                f_sym = parse(expr_str)
                # Tomamos un corte en la tercera variable (su punto medio)
                var3 = sys.sym_vars[2]
                a3 = float(parse(p.get("from3", "0")).evalf())
                b3 = float(parse(p.get("to3", "1")).evalf())
                corte = (a3 + b3) / 2
                f_corte = f_sym.subs(var3, corte)

                sym1, sym2 = sys.sym_vars[:2]
                f_np = to_numpy(f_corte, sym1, sym2)

                a1 = float(parse(p.get("from1", "0")).evalf())
                b1 = float(parse(p.get("to1", "1")).evalf())
                a2 = float(parse(p.get("from2", "0")).evalf())
                b2 = float(parse(p.get("to2", "2*pi")).evalf())

                xs = np.linspace(a1, b1, 50)
                ys = np.linspace(a2, b2, 50)
                U, V = np.meshgrid(xs, ys)
                Z = safe_eval2(f_np, U, V)

                # Convertir a cartesianas
                if sys.name == "Cartesianas":
                    X, Y = U, V
                elif sys.name == "Cilíndricas":
                    X = U * np.cos(V)
                    Y = U * np.sin(V)
                else:
                    X = U * np.sin(V) * np.cos(0)
                    Y = U * np.sin(V) * np.sin(0)

                surf = ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.85, edgecolor="none")
                fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
                self._ax3d_style(ax)
                ax.set_title(f"Corte {var3}={corte:.2g} | f={expr_str}", color=C_TEXT, fontsize=9)

            # ---------- FLUJO (campo vectorial sobre curva) ----------
            elif modo == "Flujo":
                # Graficación simplificada: curva y campo en 2D (o 3D si es esférico)
                if sys.name == "Esféricas":
                    ax = fig.add_subplot(111, projection="3d")
                    # Similar a Línea en 3D pero con flechas de campo (omitimos flechas por simplicidad)
                    a = float(parse(p.get("from", "0")).evalf())
                    b = float(parse(p.get("to", "2*pi")).evalf())
                    r_t = parse(p.get("rt", "1"))
                    th_t = parse(p.get("tht", "pi/2"))
                    ph_t = parse(p.get("phit", "t"))
                    r_np = to_numpy(r_t, t)
                    th_np = to_numpy(th_t, t)
                    ph_np = to_numpy(ph_t, t)
                    ts = np.linspace(a, b, 400)
                    rs = safe_eval1(r_np, ts)
                    ths = safe_eval1(th_np, ts)
                    phs = safe_eval1(ph_np, ts)
                    X = rs * np.sin(ths) * np.cos(phs)
                    Y = rs * np.sin(ths) * np.sin(phs)
                    Z = rs * np.cos(ths)
                    ax.plot(X, Y, Z, color=C_GOLD, lw=2.5, label="curva C")
                    ax.plot([X[0]], [Y[0]], [Z[0]], "o", color=C_ACCENT, ms=8, label="inicio")
                    ax.plot([X[-1]], [Y[-1]], [Z[-1]], "s", color=C_DANGER, ms=8, label="fin")
                    self._ax3d_style(ax)
                    ax.legend(facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT)
                    ax.set_title(f"Flujo en {sys.name} (curva)", color=C_TEXT)
                else:
                    ax = fig.add_subplot(111)
                    self._ax_style(ax)
                    a = float(parse(p.get("from", "0")).evalf())
                    b = float(parse(p.get("to", "2*pi")).evalf())
                    if sys.name == "Cartesianas":
                        x_t = parse(p.get("rx", "cos(t)"))
                        y_t = parse(p.get("ry", "sin(t)"))
                        x_np = to_numpy(x_t, t)
                        y_np = to_numpy(y_t, t)
                        ts = np.linspace(a, b, 400)
                        xs = safe_eval1(x_np, ts)
                        ys = safe_eval1(y_np, ts)
                        ax.plot(xs, ys, color=C_ACCENT, lw=2.5, label="curva C")
                        ax.plot(xs[0], ys[0], "o", color=C_ACCENT, ms=8, label="inicio")
                        ax.plot(xs[-1], ys[-1], "s", color=C_DANGER, ms=8, label="fin")
                        # Añadir campo vectorial simple (flechas)
                        Xg, Yg = np.meshgrid(np.linspace(-2,2,10), np.linspace(-2,2,10))
                        Fx = parse(p.get("Fx", "-y"))
                        Fy = parse(p.get("Fy", "x"))
                        Fx_np = to_numpy(Fx, x, y)
                        Fy_np = to_numpy(Fy, x, y)
                        Ug = safe_eval2(Fx_np, Xg, Yg)
                        Vg = safe_eval2(Fy_np, Xg, Yg)
                        ax.quiver(Xg, Yg, Ug, Vg, alpha=0.5)
                    else:  # Cilíndricas
                        r_t = parse(p.get("rt", "1"))
                        th_t = parse(p.get("tht", "t"))
                        r_np = to_numpy(r_t, t)
                        th_np = to_numpy(th_t, t)
                        ts = np.linspace(a, b, 400)
                        rs = safe_eval1(r_np, ts)
                        ths = safe_eval1(th_np, ts)
                        xs = rs * np.cos(ths)
                        ys = rs * np.sin(ths)
                        ax.plot(xs, ys, color=C_ACCENT, lw=2.5, label="curva C")
                        ax.plot(xs[0], ys[0], "o", color=C_ACCENT, ms=8, label="inicio")
                        ax.plot(xs[-1], ys[-1], "s", color=C_DANGER, ms=8, label="fin")
                    ax.legend(facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT)
                    ax.set_title(f"Flujo en {sys.name}", color=C_TEXT)
                    ax.set_aspect("equal", adjustable="datalim")

        except Exception as exc:
            ax = fig.add_subplot(111)
            ax.set_facecolor(C_PANEL)
            ax.text(0.5, 0.5, f"No se pudo graficar:\n{exc}", ha="center", va="center",
                    color=C_DANGER, fontsize=10, transform=ax.transAxes, wrap=True)
            ax.axis("off")

        fig.tight_layout(pad=1.4)

    def _ax_style(self, ax):
        ax.set_facecolor(C_PANEL)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(C_BORDER)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.grid(color=C_BORDER, lw=0.5, ls="--", alpha=0.4)

    def _ax3d_style(self, ax):
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlabel("x", color=C_TEXT)
        ax.set_ylabel("y", color=C_TEXT)
        ax.set_zlabel("z", color=C_TEXT)

    # ==================================================================
    # PESTAÑA 2: DERIVADAS MIXTAS
    # ==================================================================
    def _build_tab_derivadas(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=10,
                             border_width=1, border_color=C_BORDER)
        frame.grid(row=0, column=0, sticky="nsew", padx=50, pady=20)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(frame, text="Derivadas Mixtas — Orden Superior",
                     font=ctk.CTkFont("Consolas", 16, "bold"), text_color=C_TEXT
                     ).grid(row=0, column=0, padx=20, pady=(20, 4), sticky="w")
        ctk.CTkLabel(frame, text=f"Variables disponibles: {', '.join(self.current_system.var_names)}  — Separa con comas",
                     font=ctk.CTkFont("Consolas", 11), text_color=C_MUTED
                     ).grid(row=1, column=0, padx=20, pady=(0, 14), sticky="w")

        f_row = ctk.CTkFrame(frame, fg_color="transparent")
        f_row.grid(row=2, column=0, padx=20, pady=(0, 8), sticky="ew")
        f_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f_row, text="Función  f :", width=130,
                     font=ctk.CTkFont("Consolas", 13), text_color=C_MUTED, anchor="w"
                     ).grid(row=0, column=0, padx=(0, 8))
        self._der_expr = ctk.CTkEntry(f_row, height=38, font=ctk.CTkFont("Consolas", 15),
                                      placeholder_text="Ej: r**2 * sin(theta) * z",
                                      fg_color="#080c14", border_color=C_BORDER, text_color=C_TEXT)
        self._der_expr.grid(row=0, column=1, sticky="ew")

        v_row = ctk.CTkFrame(frame, fg_color="transparent")
        v_row.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")
        v_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(v_row, text="Variables :", width=130,
                     font=ctk.CTkFont("Consolas", 13), text_color=C_MUTED, anchor="w"
                     ).grid(row=0, column=0, padx=(0, 8))
        self._der_vars = ctk.CTkEntry(v_row, height=38, font=ctk.CTkFont("Consolas", 15),
                                      placeholder_text="Ej: r, theta, r, z",
                                      fg_color="#080c14", border_color=C_BORDER, text_color=C_TEXT)
        self._der_vars.grid(row=0, column=1, sticky="ew")

        opt_row = ctk.CTkFrame(frame, fg_color="transparent")
        opt_row.grid(row=4, column=0, padx=20, pady=(0, 8), sticky="ew")
        opt_row.grid_columnconfigure(1, weight=1)
        self._der_simplify = ctk.CTkCheckBox(opt_row, text="Simplificar resultado",
                                             font=ctk.CTkFont("Consolas", 12), text_color=C_TEXT)
        self._der_simplify.grid(row=0, column=0, sticky="w")
        kb2 = ctk.CTkFrame(opt_row, fg_color="transparent")
        kb2.grid(row=0, column=1, padx=(20, 0), sticky="e")
        self._build_kb(kb2, self._der_expr)

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.grid(row=5, column=0, padx=20, pady=(0, 8), sticky="ew")
        btn_row.grid_columnconfigure((0,1), weight=1)
        ctk.CTkButton(btn_row, text="▶  Calcular Derivada", height=40,
                      font=ctk.CTkFont("Consolas", 13, "bold"),
                      fg_color=C_ACCENT, hover_color="#1a7a52",
                      command=self._der_calcular).grid(row=0, column=0, padx=(0,5), sticky="ew")
        ctk.CTkButton(btn_row, text="🗑  Limpiar", height=40,
                      font=ctk.CTkFont("Consolas", 13),
                      fg_color=C_DANGER, hover_color="#b91c1c",
                      command=self._der_limpiar).grid(row=0, column=1, padx=(5,0), sticky="ew")

        ctk.CTkLabel(frame, text="Resultado:", font=ctk.CTkFont("Consolas", 13, "bold"), text_color=C_TEXT
                     ).grid(row=6, column=0, padx=20, pady=(4,2), sticky="w")
        self._der_result = ctk.CTkTextbox(frame, font=ctk.CTkFont("Consolas", 14),
                                          fg_color=C_OK_BG, text_color=C_TEXT, height=120)
        self._der_result.grid(row=7, column=0, padx=20, pady=(0,20), sticky="ew")
        self._der_result.insert("0.0", "Aquí aparecerá el resultado...")
        self._der_result.configure(state="disabled")

    def _der_calcular(self):
        expr_str = self._der_expr.get().strip()
        vars_str = self._der_vars.get().strip()
        simplif = self._der_simplify.get()
        self._der_result.configure(state="normal")
        self._der_result.delete("0.0", "end")
        try:
            f = parse(expr_str)
            var_names = [v.strip() for v in vars_str.split(",") if v.strip()]
            # Mapear nombres a símbolos del sistema actual (más t, u, v)
            sym_map = {str(s): s for s in self.current_system.sym_vars}
            sym_map['t'] = t
            sym_map['u'] = u
            sym_map['v'] = v
            current = f
            for vname in var_names:
                sym = sym_map.get(vname, sp.Symbol(vname))
                current = sp.diff(current, sym)
            if simplif:
                current = sp.simplify(current)
            self._der_result.insert("0.0", str(current))
            self._der_result.configure(fg_color=C_OK_BG)
        except Exception as e:
            self._der_result.insert("0.0", f"Error: {e}")
            self._der_result.configure(fg_color=C_ERR_BG)
        self._der_result.configure(state="disabled")

    def _der_limpiar(self):
        self._der_expr.delete(0, "end")
        self._der_vars.delete(0, "end")
        self._der_result.configure(state="normal")
        self._der_result.delete("0.0", "end")
        self._der_result.insert("0.0", "Aquí aparecerá el resultado...")
        self._der_result.configure(fg_color=C_OK_BG, state="disabled")

    # ==================================================================
    # PESTAÑA 3: PRODUCTOS VECTORIALES (2 y 3 vectores)
    # ==================================================================
    def _build_tab_vectoriales(self, parent):
        # Sub-pestañas para 2 y 3 vectores
        subtabs = ctk.CTkTabview(parent, fg_color=C_CARD, corner_radius=10)
        subtabs.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        subtabs._segmented_button.configure(font=ctk.CTkFont("Consolas", 12, "bold"))

        tab2 = subtabs.add("  2 VECTORES  ")
        tab3 = subtabs.add("  3 VECTORES  ")
        for tab in (tab2, tab3):
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)

        self._build_2vec(tab2)
        self._build_3vec(tab3)

    def _build_2vec(self, parent):
        frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="PRODUCTOS VECTORIALES (2 VECTORES)",
                     font=ctk.CTkFont("Consolas", 18, "bold"), text_color=self.current_system.color
                     ).pack(pady=(18, 2))
        ctk.CTkLabel(frame, text="Producto Punto y Producto Cruz",
                     font=ctk.CTkFont("Consolas", 12), text_color=C_MUTED
                     ).pack(pady=(0, 14))

        # Entradas para vector A
        ctk.CTkLabel(frame, text="Vector A (componentes en la base del sistema)",
                     font=ctk.CTkFont("Consolas", 12, "bold"), text_color=self.current_system.color
                     ).pack(anchor="w", padx=20, pady=(8, 3))
        self._v2a = self._vec_input_row(frame, "A")

        # Entradas para vector B
        ctk.CTkLabel(frame, text="Vector B",
                     font=ctk.CTkFont("Consolas", 12, "bold"), text_color=self.current_system.color
                     ).pack(anchor="w", padx=20, pady=(15, 3))
        self._v2b = self._vec_input_row(frame, "B")

        ctk.CTkButton(frame, text="CALCULAR", height=44,
                      font=ctk.CTkFont("Consolas", 14, "bold"),
                      fg_color=self.current_system.color, text_color="#000000",
                      command=self._vec2_calcular).pack(fill="x", padx=20, pady=(8, 16))

        res = ctk.CTkFrame(frame, fg_color="#0d0d0d", border_width=2,
                           border_color=self.current_system.color, corner_radius=8)
        res.pack(fill="x", padx=20, pady=(0,20))

        ctk.CTkLabel(res, text="PRODUCTO PUNTO  A·B",
                     font=ctk.CTkFont("Consolas", 11, "bold"), text_color=self.current_system.color
                     ).pack(anchor="w", padx=14, pady=(10, 2))
        self._v2_res_punto = ctk.CTkLabel(res, text="0", font=ctk.CTkFont("Consolas", 18, "bold"),
                                          text_color="#00ff00", anchor="e")
        self._v2_res_punto.pack(anchor="e", padx=14, pady=(0,8), fill="x")

        ctk.CTkFrame(res, height=1, fg_color=self.current_system.color).pack(fill="x", padx=10)

        ctk.CTkLabel(res, text="PRODUCTO CRUZ  A×B",
                     font=ctk.CTkFont("Consolas", 11, "bold"), text_color=self.current_system.color
                     ).pack(anchor="w", padx=14, pady=(10, 2))
        self._v2_res_cruz = ctk.CTkLabel(res, text="[0, 0, 0]", font=ctk.CTkFont("Consolas", 18, "bold"),
                                         text_color="#00ff00", anchor="w")
        self._v2_res_cruz.pack(anchor="w", padx=14, pady=(0,12))

    def _build_3vec(self, parent):
        frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="PRODUCTOS TRIPLES (3 VECTORES)",
                     font=ctk.CTkFont("Consolas", 18, "bold"), text_color=self.current_system.color
                     ).pack(pady=(18, 2))
        ctk.CTkLabel(frame, text="Triple Escalar A·(B×C)  y  Triple Vectorial",
                     font=ctk.CTkFont("Consolas", 12), text_color=C_MUTED
                     ).pack(pady=(0, 14))

        self._v3a = self._vec_input_row(frame, "A")
        self._v3b = self._vec_input_row(frame, "B")
        self._v3c = self._vec_input_row(frame, "C")

        ctk.CTkLabel(frame, text="Orden del Triple Vectorial:",
                     font=ctk.CTkFont("Consolas", 11, "bold"), text_color=self.current_system.color
                     ).pack(anchor="w", padx=20, pady=(10,4))
        self._v3_orden = ctk.CTkSegmentedButton(frame, values=["A×(B×C)", "(A×B)×C"],
                                                font=ctk.CTkFont("Consolas", 11, "bold"),
                                                selected_color=self.current_system.color,
                                                unselected_color="#2a2a2a", text_color="#ffffff")
        self._v3_orden.pack(fill="x", padx=20, pady=(0,12))
        self._v3_orden.set("A×(B×C)")

        ctk.CTkButton(frame, text="CALCULAR", height=44,
                      font=ctk.CTkFont("Consolas", 14, "bold"),
                      fg_color="#00ff00", text_color="#000000", hover_color="#00cc00",
                      command=self._vec3_calcular).pack(fill="x", padx=20, pady=(4,14))

        res = ctk.CTkFrame(frame, fg_color="#0d0d0d", border_width=2,
                           border_color=self.current_system.color, corner_radius=8)
        res.pack(fill="x", padx=20, pady=(0,20))

        ctk.CTkLabel(res, text="TRIPLE ESCALAR  A·(B×C)",
                     font=ctk.CTkFont("Consolas", 11, "bold"), text_color=self.current_system.color
                     ).pack(anchor="w", padx=14, pady=(10,2))
        self._v3_res_esc = ctk.CTkLabel(res, text="0", font=ctk.CTkFont("Consolas", 18, "bold"),
                                        text_color="#00ff00", anchor="e")
        self._v3_res_esc.pack(anchor="e", padx=14, pady=(0,8), fill="x")

        ctk.CTkFrame(res, height=1, fg_color=self.current_system.color).pack(fill="x", padx=10)

        ctk.CTkLabel(res, text="TRIPLE VECTORIAL",
                     font=ctk.CTkFont("Consolas", 11, "bold"), text_color=self.current_system.color
                     ).pack(anchor="w", padx=14, pady=(10,2))
        self._v3_res_vec = ctk.CTkLabel(res, text="[0, 0, 0]", font=ctk.CTkFont("Consolas", 18, "bold"),
                                        text_color="#00ff00", anchor="w", wraplength=500)
        self._v3_res_vec.pack(anchor="w", padx=14, pady=(0,12))

    def _vec_input_row(self, parent, label):
        """Crea una fila con tres campos de entrada para las componentes del vector."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=4)
        ctk.CTkLabel(frame, text=f"{label}:", width=30,
                     font=ctk.CTkFont("Consolas", 12, "bold"), text_color=self.current_system.color
                     ).pack(side="left", padx=(0,10))
        entries = []
        # Usar nombres de componentes según sistema (r̂, θ̂, φ̂) o (i,j,k)
        if self.current_system.name == "Cartesianas":
            comp_names = ["i", "j", "k"]
        elif self.current_system.name == "Cilíndricas":
            comp_names = ["r̂", "θ̂", "ẑ"]
        else:
            comp_names = ["r̂", "θ̂", "φ̂"]
        for comp in comp_names:
            sub = ctk.CTkFrame(frame, fg_color="transparent")
            sub.pack(side="left", expand=True, fill="x", padx=4)
            ctk.CTkLabel(sub, text=f"{comp}:", font=ctk.CTkFont("Consolas", 10), text_color=C_MUTED
                         ).pack(side="left", padx=(0,4))
            ent = ctk.CTkEntry(sub, width=120, font=ctk.CTkFont("Consolas", 12),
                               fg_color="#2a2a2a", border_color=self.current_system.color,
                               border_width=1, corner_radius=4,
                               placeholder_text="0")
            ent.pack(side="left", fill="x", expand=True)
            entries.append(ent)
        return entries

    def _get_vec(self, entries):
        """Lee las componentes del vector como expresiones SymPy.
        Acepta tanto números (1, 2.5) como expresiones simbólicas (r*cos(theta))."""
        return [parse(e.get().strip() or "0") for e in entries]

    def _fmt_sym(self, expr):
        """Formatea una expresión SymPy: si es numérica muestra decimal, si no la simplifica."""
        simp = sp.simplify(expr)
        try:
            val = float(simp.evalf())
            return f"{val:.6g}"
        except (TypeError, ValueError):
            return str(simp)

    def _vec2_calcular(self):
        """Calcula producto punto y producto cruz de dos vectores con componentes simbólicas."""
        try:
            a = self._get_vec(self._v2a)
            b = self._get_vec(self._v2b)
            # Producto punto: A·B = a1*b1 + a2*b2 + a3*b3
            punto = sp.simplify(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
            # Producto cruz: A×B usando determinante 3x3
            cruz = [
                sp.simplify(a[1]*b[2] - a[2]*b[1]),
                sp.simplify(a[2]*b[0] - a[0]*b[2]),
                sp.simplify(a[0]*b[1] - a[1]*b[0]),
            ]
            self._v2_res_punto.configure(text=self._fmt_sym(punto))
            self._v2_res_cruz.configure(
                text=f"[{self._fmt_sym(cruz[0])},  {self._fmt_sym(cruz[1])},  {self._fmt_sym(cruz[2])}]"
            )
        except Exception as e:
            self._v2_res_punto.configure(text="ERROR")
            self._v2_res_cruz.configure(text=str(e))

    def _vec3_calcular(self):
        """Calcula triple escalar A·(B×C) y triple vectorial con componentes simbólicas."""
        try:
            a = self._get_vec(self._v3a)
            b = self._get_vec(self._v3b)
            c = self._get_vec(self._v3c)

            # Función auxiliar de producto cruz simbólico
            def cross(u, v):
                return [
                    sp.simplify(u[1]*v[2] - u[2]*v[1]),
                    sp.simplify(u[2]*v[0] - u[0]*v[2]),
                    sp.simplify(u[0]*v[1] - u[1]*v[0]),
                ]

            # Triple escalar: A·(B×C)
            bxc = cross(b, c)
            triple_esc = sp.simplify(a[0]*bxc[0] + a[1]*bxc[1] + a[2]*bxc[2])

            # Triple vectorial según orden seleccionado
            orden = self._v3_orden.get()
            if orden == "A×(B×C)":
                triple_vec = cross(a, bxc)
            else:
                axb = cross(a, b)
                triple_vec = cross(axb, c)

            self._v3_res_esc.configure(text=self._fmt_sym(triple_esc))
            self._v3_res_vec.configure(
                text=f"[{self._fmt_sym(triple_vec[0])},  {self._fmt_sym(triple_vec[1])},  {self._fmt_sym(triple_vec[2])}]"
            )
        except Exception as e:
            self._v3_res_esc.configure(text="ERROR")
            self._v3_res_vec.configure(text=str(e))

    # ==================================================================
    # PESTAÑA 4: OPERADORES NABLA (GRADIENTE, DIVERGENCIA, ROTACIONAL, LAPLACIANO)
    # ==================================================================
    def _build_tab_nabla(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=10,
                             border_width=1, border_color=C_BORDER)
        frame.grid(row=0, column=0, sticky="nsew", padx=40, pady=16)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(4, weight=1)

        sys = self.current_system
        ctk.CTkLabel(frame, text=f"Operadores Vectoriales — {sys.name} (∇)",
                     font=ctk.CTkFont("Consolas", 16, "bold"), text_color=sys.color
                     ).grid(row=0, column=0, padx=20, pady=(18,4), sticky="w")
        ctk.CTkLabel(frame, text="Gradiente · Divergencia · Rotacional · Laplaciano",
                     font=ctk.CTkFont("Consolas", 11), text_color=C_MUTED
                     ).grid(row=1, column=0, padx=20, pady=(0,14), sticky="w")

        # Entradas para componentes
        inp_frame = ctk.CTkFrame(frame, fg_color="transparent")
        inp_frame.grid(row=2, column=0, padx=20, pady=(0,10), sticky="ew")
        inp_frame.grid_columnconfigure(1, weight=1)

        comp_labels = []
        if sys.name == "Cartesianas":
            comp_labels = ["Componente i (Fx) o función escalar f:", "Componente j (Fy):", "Componente k (Fz):"]
        elif sys.name == "Cilíndricas":
            comp_labels = ["Componente Fr (r̂) o f:", "Componente Fθ (θ̂):", "Componente Fz (ẑ):"]
        else:
            comp_labels = ["Componente Fr (r̂) o f:", "Componente Fθ (θ̂):", "Componente Fφ (φ̂):"]

        self._nab_entries = []
        for i, lbl in enumerate(comp_labels):
            ctk.CTkLabel(inp_frame, text=lbl, width=260,
                         font=ctk.CTkFont("Consolas", 12), text_color=C_MUTED, anchor="w"
                         ).grid(row=i, column=0, padx=(0,10), pady=5, sticky="w")
            ent = ctk.CTkEntry(inp_frame, height=34, font=ctk.CTkFont("Consolas", 14),
                               fg_color="#080c14", border_color=C_BORDER, text_color=C_TEXT)
            ent.grid(row=i, column=1, pady=5, sticky="ew")
            self._nab_entries.append(ent)

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=3, column=0, padx=20, pady=(4,10), sticky="ew")
        for c in range(4):
            btn_frame.grid_columnconfigure(c, weight=1)

        ops = [
            ("Gradiente  ∇f",   C_ACCENT2,  self._nab_gradiente),
            ("Divergencia  ∇·F", C_ACCENT,  self._nab_divergencia),
            ("Rotacional  ∇×F", "#f59e0b", self._nab_rotacional),
            ("Laplaciano  ∇²f", C_PURPLE,  self._nab_laplaciano),
        ]
        for c, (txt, color, cmd) in enumerate(ops):
            ctk.CTkButton(btn_frame, text=txt, height=40,
                          font=ctk.CTkFont("Consolas", 12, "bold"),
                          fg_color=color, hover_color="#444",
                          command=cmd).grid(row=0, column=c, padx=4, sticky="ew")

        ctk.CTkLabel(frame, text="Desarrollo paso a paso:",
                     font=ctk.CTkFont("Consolas", 12, "bold"), text_color=C_TEXT
                     ).grid(row=4, column=0, padx=20, pady=(4,2), sticky="w")

        self._nab_result = ctk.CTkTextbox(frame, font=ctk.CTkFont("Consolas", 13),
                                          fg_color=C_PANEL, text_color=C_TEXT)
        self._nab_result.grid(row=5, column=0, padx=20, pady=(0,20), sticky="nsew")
        frame.grid_rowconfigure(5, weight=1)
        self._nab_result.insert("end", "Ingresa las componentes y pulsa una operación.\n\n"
                                       "• Gradiente y Laplaciano: usa solo la primera componente (función escalar).\n"
                                       "• Divergencia y Rotacional: usa las tres componentes del campo vectorial.\n\n"
                                       f"Sistema actual: {sys.name}")

    def _nab_show(self, texto):
        self._nab_result.delete("1.0", "end")
        self._nab_result.insert("end", texto)

    def _nab_gradiente(self):
        sys = self.current_system
        f_str = self._nab_entries[0].get().strip()
        if not f_str:
            self._nab_show("Error: Ingresa una función escalar en la primera componente.")
            return
        try:
            f = parse(f_str)
            grad = sys.grad(f)
            names = sys.var_names
            result = f"--- GRADIENTE en {sys.name} ---\n\n"
            result += f"f = {f}\n\n"
            result += f"∇f = (∂f/∂{names[0]}) {names[0]}̂ + "
            result += f"(1/{names[0] if names[0]=='r' else '1'}·∂f/∂{names[1]}) {names[1]}̂ + "
            result += f"(∂f/∂{names[2]}) {names[2]}̂\n\n"
            result += f"Componente {names[0]}̂: {sp.simplify(grad[0])}\n"
            result += f"Componente {names[1]}̂: {sp.simplify(grad[1])}\n"
            result += f"Componente {names[2]}̂: {sp.simplify(grad[2])}\n"
            self._nab_show(result)
        except Exception as e:
            self._nab_show(f"Error: {e}")

    def _nab_divergencia(self):
        sys = self.current_system
        try:
            f1 = parse(self._nab_entries[0].get().strip() or "0")
            f2 = parse(self._nab_entries[1].get().strip() or "0")
            f3 = parse(self._nab_entries[2].get().strip() or "0")
            div = sys.div(f1, f2, f3)
            names = sys.var_names
            result = f"--- DIVERGENCIA en {sys.name} ---\n\n"
            result += f"F = (F_{names[0]}) {names[0]}̂ + (F_{names[1]}) {names[1]}̂ + (F_{names[2]}) {names[2]}̂\n"
            result += f"F_{names[0]} = {f1}\nF_{names[1]} = {f2}\nF_{names[2]} = {f3}\n\n"
            result += f"∇·F = {sp.simplify(div)}"
            self._nab_show(result)
        except Exception as e:
            self._nab_show(f"Error: {e}")

    def _nab_rotacional(self):
        sys = self.current_system
        try:
            f1 = parse(self._nab_entries[0].get().strip() or "0")
            f2 = parse(self._nab_entries[1].get().strip() or "0")
            f3 = parse(self._nab_entries[2].get().strip() or "0")
            rot = sys.rot(f1, f2, f3)
            names = sys.var_names
            result = f"--- ROTACIONAL en {sys.name} ---\n\n"
            result += f"F = (F_{names[0]}) {names[0]}̂ + (F_{names[1]}) {names[1]}̂ + (F_{names[2]}) {names[2]}̂\n\n"
            result += f"Componente {names[0]}̂: {sp.simplify(rot[0])}\n"
            result += f"Componente {names[1]}̂: {sp.simplify(rot[1])}\n"
            result += f"Componente {names[2]}̂: {sp.simplify(rot[2])}"
            self._nab_show(result)
        except Exception as e:
            self._nab_show(f"Error: {e}")

    def _nab_laplaciano(self):
        sys = self.current_system
        f_str = self._nab_entries[0].get().strip()
        if not f_str:
            self._nab_show("Error: Ingresa una función escalar en la primera componente.")
            return
        try:
            f = parse(f_str)
            lap = sys.lap(f)
            result = f"--- LAPLACIANO en {sys.name} ---\n\n"
            result += f"f = {f}\n\n"
            result += f"∇²f = {sp.simplify(lap)}"
            self._nab_show(result)
        except Exception as e:
            self._nab_show(f"Error: {e}")

    # ==================================================================
    # TECLADO CIENTÍFICO COMPARTIDO
    # ==================================================================
    def _build_kb(self, parent, target_entry):
        # Botones comunes
        common_keys = [
            ("sin","sin()"), ("cos","cos()"), ("tan","tan()"),
            ("ln","log()"), ("√","sqrt()"), ("eˣ","exp()"),
            ("π","pi"), ("xⁿ","**"), ("(","("), (")",")"), ("⌫","__del__"),
        ]
        # Variables del sistema actual
        var_keys = [(btn_label, var_name) for btn_label, var_name in self.current_system.var_buttons]
        # Organizar en 2 filas superiores para variables del sistema
        cols = 8
        for c in range(cols):
            parent.grid_columnconfigure(c, weight=0)

        row = 0
        col = 0
        for lbl, val in var_keys:
            btn = ctk.CTkButton(parent, text=lbl, width=52, height=30,
                                font=ctk.CTkFont("Consolas", 11),
                                fg_color=self.current_system.color,
                                hover_color="#444",
                                border_width=1, border_color=C_BORDER, corner_radius=6,
                                command=lambda v=val, e=target_entry: self._kb_press(v, e))
            btn.grid(row=row, column=col, padx=2, pady=2)
            col += 1
            if col >= cols:
                col = 0
                row += 1

        # Resto de botones comunes
        for lbl, val in common_keys:
            is_del = val == "__del__"
            btn = ctk.CTkButton(parent, text=lbl, width=52, height=30,
                                font=ctk.CTkFont("Consolas", 11),
                                fg_color=C_DANGER if is_del else C_CARD,
                                hover_color="#7f1d1d" if is_del else C_BORDER,
                                border_width=1, border_color=C_BORDER, corner_radius=6,
                                command=lambda v=val, e=target_entry: self._kb_press(v, e))
            btn.grid(row=row, column=col, padx=2, pady=2)
            col += 1
            if col >= cols:
                col = 0
                row += 1

    def _kb_press(self, val, entry):
        if val == "__del__":
            pos = entry.index("insert")
            if pos > 0:
                cur = entry.get()
                entry.delete(0, "end")
                entry.insert(0, cur[:pos-1] + cur[pos:])
                entry.icursor(pos - 1)
            return
        pos = entry.index("insert")
        if val.endswith("()"):
            entry.insert(pos, val[:-1])
            entry.icursor(pos + len(val) - 1)
        else:
            entry.insert(pos, val)


# ======================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    app = SuiteCalculoUnificada()
    app.mainloop()