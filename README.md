# 🧮 Suite de Cálculo Vectorial — Sistemas Unificados

Una suite completa de cálculo vectorial que integra múltiples sistemas de coordenadas en una interfaz moderna y fácil de usar. Realiza integrales, derivadas, productos vectoriales y operadores nabla en coordenadas Cartesianas, Cilíndricas y Esféricas.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Local](https://img.shields.io/badge/Status-100%25%20Local-success)

## ✨ Características Principales

- **🎯 Múltiples Sistemas de Coordenadas**: Soporte completo para Cartesianas (x,y,z), Cilíndricas (r,θ,z) y Esféricas (r,θ,φ)
- **📊 7 Tipos de Integrales**: Indefinida, Definida, Doble, Triple, Línea, Flujo y Volumen
- **📈 Visualización Gráfica**: Generación de gráficas 2D y 3D interactivas con Matplotlib
- **🧮 Operadores Nabla**: Gradiente, Divergencia, Rotacional y Laplaciano
- **🔢 Productos Vectoriales**: Producto punto, producto cruz y productos triples
- **📝 Derivadas Mixtas**: Cálculo de derivadas parciales de orden superior
- **🔒 100% Local**: Funciona completamente sin conexión a internet, sin APIs externas
- **🎨 Interfaz Moderna**: GUI oscura y responsiva construida con CustomTkinter
- **⚡ Cálculo Simbólico**: Potente motor matemático usando SymPy

## 🛠 Tecnologías Utilizadas

- **Python** - Lenguaje de programación principal
- **CustomTkinter** - Interfaz gráfica moderna y responsiva
- **SymPy** - Cálculo simbólico y matemático
- **NumPy** - Computación numérica
- **Matplotlib** - Visualización gráfica 2D/3D
- **Tkinter** - Framework de interfaz gráfica estándar

## 🚀 Cómo Ejecutarlo Localmente

### Prerrequisitos

Asegúrate de tener Python 3.8 o superior instalado en tu sistema.

### Instalación de Dependencias

```bash
pip install customtkinter numpy matplotlib sympy
```

### Ejecución

1. Navega al directorio del proyecto:
```bash
cd "c:\Users\ritog\Documents\Materias de 3er semestre\Calculo vectorial\Calculadora"
```

2. Ejecuta el archivo principal:
```bash
python suite_vectorial_unificada.py
```

O alternativamente:
```bash
python3 suite_vectorial_unificada.py
```

### Uso

Una vez iniciada la aplicación:
- Selecciona el sistema de coordenadas deseado (Cartesianas, Cilíndricas o Esféricas)
- Elige entre las 4 pestañas principales: Integrales, Derivadas Mixtas, Productos Vectoriales u Operadores Nabla
- Ingresa las expresiones matemáticas usando la sintaxis estándar
- Utiliza los botones de acceso rápido para insertar variables y constantes
- Presiona "Calcular" para obtener el resultado simbólico
- Presiona "Graficar" para visualizar la función

## 📚 Sintaxis de Expresiones

- Variables: `x`, `y`, `z`, `r`, `theta`, `phi`, `t`
- Constantes: `pi`, `e`
- Funciones: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`
- Operadores: `+`, `-`, `*`, `/`, `^` (potencia)
- Ejemplos: `x**2 + sin(y)`, `r*cos(theta)`, `exp(-x**2)`

## 👨‍💻 Autor

**Oscar Avendaño Morales**

[![GitHub](https://img.shields.io/badge/Github-oscaravendanomorales--wq-lightgrey)](https://github.com/oscaravendanomorales-wq)

---

Desarrollado con ❤️ para facilitar el aprendizaje y la práctica del cálculo vectorial.
