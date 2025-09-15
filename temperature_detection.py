import matplotlib.figure
import numpy as np
import csv
import matplotlib
import sklearn

from sklearn.linear_model import LinearRegression
import sklearn.linear_model
import sklearn.pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt
from typing import Literal

import sklearn.preprocessing

def get_float(prompt: str) -> float:
    while True:
        value = input(prompt)
        try:
            value = float(value)
            return value
        except ValueError:
            print("Por favor, ingrese un número válido")

def get_initial_data(max_x: float, max_y: float):
    initial_points = [
        [ 0, 0, "inferior izquierda" ],
        [ max_x, 0, "inferior derecha" ],
        [ 0, max_y, "superior izquierda" ],
        [ max_x, max_y, "superior derecha" ]
    ]

    points = []
    for i, (x_0, y_0, name) in enumerate(initial_points):
        index = i + 1
        print(f"==Sensado número {index}==")
        point = [ x_0, y_0, z0 := get_float(prompt=f'Ingrese la temperatura de la esquina {name}: ')]
        points.append(point)
        print()

    return points

def main():
    print("Bienvenido a la utilidad de registro de datos xyz.")
    
    max_x = get_float(prompt="Ingrese el ancho de la charola: ")
    max_y = get_float(prompt="Ingrese el alto de la charola: ")
    
    raw_xyzs = get_initial_data(max_x=max_x, max_y=max_y)
    chartable_xyzs = raw_xyzs
    predicted_xyz_tuple = ...
    polynomial: sklearn.preprocessing.PolynomialFeatures = ...
    model: sklearn.linear_model.LinearRegression = ...
    latex_equation: str = ...
    
    index = 4
    while True:
        index += 1
        *predicted_xyz_tuple, _, polynomial, model = predict_xyzs(raw_xyzs=raw_xyzs)
        latex_equation = build_latex_equation(polynomial=polynomial, model=model)

        show_dashboard(
            xyz_tuple=separate_xyzs(xyzs=chartable_xyzs), 
            predicted_xyz_tuple=predicted_xyz_tuple, 
            latex_equation=latex_equation
            )
        
        _continue = input("¿Desea agregar más datos? (S/n): ").strip().lower()

        if _continue == 'n':
            break
        print(f"==Sensado número {index}==")

        x = get_float(prompt=f"Ingrese el valor de x: ")
        y = get_float(prompt=f"Ingrese el valor de y: ")
        z = get_float(prompt="Ingrese el valor de temperatura: ")
        
        raw_xyzs.append([x, y, z])
        missing_points = get_missing_points(xyzs=raw_xyzs)
        if missing_points.__len__() > 0:
            print('Parece que has sensado datos de manera irregular. Para obtener un gráfico más completo, intenta sensar los siguientes puntos: ')
            print_missing_points(missing_points=missing_points)
            chartable_xyzs = fill_missing_points(xyzs=raw_xyzs, missing_points=missing_points)
        
    print("Datos registrados:")
    for data in raw_xyzs:
        print(data)
    
    save_to_csv(raw_xyzs, "datos.csv")
    print("Datos originales guardados en 'datos.csv'.")

    save_to_csv(chartable_xyzs, 'datos_corregidos.csv')
    print("Datos corregidos guardados en 'datos_corregidos.csv'.")

    save_latex_equation(polynomial=polynomial, model=model, save_as='equation.txt')
    print("Ecuación guardada en 'equation.txt'.")
    
    save_dashboard(
        xyz_tuple=separate_xyzs(xyzs=chartable_xyzs), 
        predicted_xyz_tuple=predicted_xyz_tuple, 
        latex_equation=latex_equation, 
        save_as="dashboard_final.png"
        )
    print("Dashboard guardado como 'dashboard_final.png'.")

    save_contour( xyz_tuple=separate_xyzs(xyzs=chartable_xyzs), save_as='contour_final.png' )
    print("Gráfico de contour guardado como 'contour_final.png'.")

    save_contour( xyz_tuple=predicted_xyz_tuple, save_as='contour_prediccion_final.png' )
    print("Gráfico de contour de predicción guardado como 'contour_prediccion_final.png'.")

    save_3d( xyz_tuple=separate_xyzs(xyzs=chartable_xyzs), save_as='3d_final.png' )
    print("Gráfico de 3D de guardado como '3d_final.png'.")

    save_3d( xyz_tuple=predicted_xyz_tuple, save_as='3d_prediccion_final.png' )
    print("Gráfico de 3D de predicción guardado como '3d_prediccion_final.png'.")


def print_missing_points(missing_points: list[tuple[float, float]]):
    unique_ys = list(set(y for (_, y) in missing_points))
    unique_ys.sort(reverse=True)
    for y in unique_ys:
        coordinates = [ (x0, y0) for (x0, y0) in missing_points if y0 == y ]
        coordinates.sort(key=lambda _tuple: _tuple[0])
        print('\t'.join([ f'{repr(coordinate):<15}' for coordinate in coordinates ]))

def fill_missing_points(xyzs: list[list[float]], missing_points: list[tuple[float, float]], strategy: Literal['zeroes'] | Literal['average'] | Literal['nan'] = 'average') -> list[list[float]]:
    clean_xyzs = [*xyzs]
    match strategy:
        case 'zeroes':
            clean_xyzs.extend([[x, y, 0] for (x, y) in missing_points])

        case 'average':
            mean_value = np.mean([z for (_, __, z) in xyzs]).__float__()
            clean_xyzs.extend([[x, y, mean_value] for (x, y) in missing_points])

        case 'nan':
            clean_xyzs.extend([[x, y, np.nan] for (x, y) in missing_points])

    return clean_xyzs

def get_missing_points(xyzs) -> list[tuple[float, float]]:
    def find_coordinate_value(x0: int, y0: int) -> int:
        return next((z for (x, y, z) in xyzs if x == x0 and y == y0), np.nan)

    xs = list(set(x for (x, _, __) in xyzs))
    ys = list(set(y for (_, y, __) in xyzs))
    xs.sort()
    ys.sort()
    
    return [ (x, y) for x in xs for y in ys if find_coordinate_value(x0=x, y0=y) is np.nan ]

def separate_xyzs(xyzs: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def find_coordinate_value(x0: int, y0: int) -> int:
        return next(z for (x, y, z) in xyzs if x == x0 and y == y0)
    
    xs = list(set(x for (x, _, __) in xyzs))
    ys = list(set(y for (_, y, __) in xyzs))
    xs.sort()
    ys.sort()

    zs = np.array([ [ find_coordinate_value(x, y) for x in xs ] for y in ys ])
    
    xs, ys = np.array(xs), np.array(ys)
    xs, ys = np.meshgrid(xs, ys)
    return (xs, ys, zs)

def get_dashboard_figure():
    aspect_ratio = 16 / 9
    height = 6
    width = height * aspect_ratio

    figure = plt.figure(figsize=(width, height))
    return figure

def show_dashboard(
        xyz_tuple: tuple[np.ndarray, np.ndarray, np.ndarray], 
        predicted_xyz_tuple: tuple[np.ndarray, np.ndarray, np.ndarray], 
        latex_equation: str
        ):
    figure = get_dashboard_figure()
    plot_dashboard(
        figure=figure, 
        xyz_tuple=xyz_tuple, 
        predicted_xyz_tuple=predicted_xyz_tuple, 
        latex_equation=latex_equation
        )
    plt.show()
    plt.close(figure)

def save_dashboard(xyz_tuple, predicted_xyz_tuple, latex_equation, save_as: str | None):
    figure = get_dashboard_figure()
    plot_dashboard(
        figure=figure, 
        xyz_tuple=xyz_tuple, 
        predicted_xyz_tuple=predicted_xyz_tuple, 
        latex_equation=latex_equation
        )
    figure.savefig(save_as)
    plt.close(figure)

def save_contour(xyz_tuple, save_as: str):
    xs, ys, zs = xyz_tuple
    figure = plt.figure()
    axes = figure.add_subplot()
    plot_contour(axes=axes, xs=xs, ys=ys, zs=zs)
    figure.savefig(save_as)
    plt.close(figure)

def save_3d(xyz_tuple, save_as: str):
    xs, ys, zs = xyz_tuple
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    plot_3d(axes=axes, xs=xs, ys=ys, zs=zs)
    figure.savefig(save_as)
    plt.close(figure)

def save_latex_equation(polynomial: sklearn.preprocessing.PolynomialFeatures, model: sklearn.linear_model.LinearRegression, save_as: str):
    feature_names = polynomial.get_feature_names_out(["x", "y"])
    coefficients = model.coef_
    intercept = model.intercept_

    terms = [ str(intercept) ]
    for name, coefficient in zip(feature_names, coefficients):
        latex_name = name.replace(" ", "")
        latex_name = latex_name.replace("x^", "x^{").replace("y^", "y^{")
        latex_name = latex_name.replace("x y", "xy").replace("}", "}")  # limpiar
        if "{" in latex_name:
            latex_name += "}"
        terms.append(f"{coefficient}{latex_name}")
    
    equation = r"$z = " + " + ".join(terms) + r"$"
    with open(save_as, 'w') as equation_file:
        equation_file.write(equation)

def build_latex_equation(polynomial: sklearn.preprocessing.PolynomialFeatures, model: sklearn.linear_model.LinearRegression):
    """Construye la ecuación polinomial en formato LaTeX."""
    feature_names = polynomial.get_feature_names_out(["x", "y"])
    coefficients = model.coef_
    intercept = model.intercept_

    terms = [f"{intercept:.2f}"]
    for name, coefficient in zip(feature_names, coefficients):
        if abs(coefficient) < 1e-6:  # omitir coeficientes casi cero
            continue
        # Formato LaTeX: manejar x^2, xy, etc.
        name_latex = name.replace(" ", "")
        name_latex = name_latex.replace("x^", "x^{").replace("y^", "y^{")
        name_latex = name_latex.replace("x y", "xy").replace("}", "}")  # limpiar
        if "{" in name_latex:
            name_latex += "}"
        terms.append(f"{coefficient:.2f}{name_latex}")

    return r"$z = " + " + ".join(terms) + r"$"

def plot_dashboard(
        figure: matplotlib.figure.Figure, 
        latex_equation: str,
        xyz_tuple: tuple[np.ndarray, np.ndarray, np.ndarray],
        predicted_xyz_tuple: tuple[np.ndarray, np.ndarray, np.ndarray],
        ):
    xs, ys, zs = xyz_tuple
    predicted_xs, predicted_ys, predicted_zs = predicted_xyz_tuple

    axes_for_contour = figure.add_subplot(221)
    axes_for_3d = figure.add_subplot(222, projection='3d')

    axes_for_predicted_contour = figure.add_subplot(223)
    axes_for_predicted_3d = figure.add_subplot(224, projection='3d')

    plot_contour(axes=axes_for_contour, xs=xs, ys=ys, zs=zs)
    plot_3d(axes=axes_for_3d, xs=xs, ys=ys, zs=zs)

    plot_contour(
        axes=axes_for_predicted_contour, 
        xs=predicted_xs, 
        ys=predicted_ys, 
        zs=predicted_zs,
        title="Contorno de temperatura (predicción)"
        )
    plot_3d(
        axes=axes_for_predicted_3d, 
        xs=predicted_xs, 
        ys=predicted_ys, 
        zs=predicted_zs,
        title="Gráfico 3D de temperatura (predicción)"
        )
    
    figure.suptitle(latex_equation, fontsize=12)

def plot_contour(axes, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, title: str = "Contorno de la temperatura"):
    contour = axes.contourf(xs, ys, zs, levels=15, cmap="plasma")
    plt.colorbar(contour)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_title(title)

def plot_3d(axes, xs, ys, zs, title = "Gráfico 3D de temperatura"):
    from matplotlib.ticker import LinearLocator
    
    surface = axes.plot_surface(xs, ys, zs, antialiased=True, cmap='plasma')
    axes.zaxis.set_major_locator(LinearLocator(10))
    axes.zaxis.set_major_formatter('{x:.02f}')
    plt.colorbar(surface, shrink=0.5, aspect=5)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.set_title(title)

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])  # Encabezados
        writer.writerows(data)

def train_polynomial_regression_model(raw_xyzs, degree = 3):
    """Entrena un modelo de regresión polinomial multivariable."""
    X = np.array([[x, y] for x, y, _ in raw_xyzs])
    y = np.array([z for _, _, z in raw_xyzs])
    
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    model = LinearRegression()
    pipeline = make_pipeline(polynomial, model)
    pipeline.fit(X, y)

    return pipeline, polynomial, model

def separate_xyzs_from_model(model: sklearn.linear_model.LinearRegression, max_x, max_y, resolution: int = 50):
    """Genera una malla (xs, ys) y predice zs usando el modelo entrenado."""
    x_range = np.linspace(0, max_x, resolution)
    y_range = np.linspace(0, max_y, resolution)

    xs, ys = np.meshgrid(x_range, y_range)
    X_pred = np.c_[xs.ravel(), ys.ravel()]
    z_pred = model.predict(X_pred).reshape(xs.shape)

    return xs, ys, z_pred

def predict_xyzs(raw_xyzs) -> tuple[np.ndarray, np.ndarray, np.ndarray, sklearn.pipeline.Pipeline, sklearn.preprocessing.PolynomialFeatures, sklearn.linear_model.LinearRegression]:
    """Crea gráficos basados en el modelo de regresión."""
    pipeline, polynomial, model = train_polynomial_regression_model(raw_xyzs)

    max_x = max(x for (x, _, __) in raw_xyzs)
    max_y = max(y for (_, y, __) in raw_xyzs)
    xs, ys, zs = separate_xyzs_from_model(model=pipeline, max_x=max_x, max_y=max_y)

    return (xs, ys, zs, pipeline, polynomial, model)


if __name__ == "__main__":
    main()


