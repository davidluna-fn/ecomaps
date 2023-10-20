import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.2c'  # Muy importante, deberéis ir cambiando la versión de vuestra librería según incluyáis nuevas funcionalidades
PACKAGE_NAME = 'ecomaps'  # Debe coincidir con el nombre de la carpeta
AUTHOR = 'David Luna'  # Modificar con vuestros datos
AUTHOR_EMAIL = 'david.lunan@udea.edu.co'  # Modificar con vuestros datos
URL = 'https://github.com/davidluna-fn'  # Modificar con vuestros datos

LICENSE = 'MIT'  # Tipo de licencia
DESCRIPTION = 'Librería para analisis ecoacutisto de larga duracion'  # Descripción corta
# Referencia al documento README con una descripción más elaborada
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"


# Paquetes necesarios para que funcione la libreía. Se instalarán a la vez si no lo tuvieras ya instalado
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'scikit-maad',
    'scikit-learn',
    'matplotlib',
    'mat73',
    'suntime'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)
