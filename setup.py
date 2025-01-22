from setuptools import setup, find_packages

def load_requirements(filename):
    """Upload requirements.txt."""
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='Digital-Assistant',
    use_scm_version=True,  
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    install_requires=load_requirements('requirements.txt'),  # Загружаем зависимости из requirements.txt
    entry_points={
        'console_scripts': [
            'your_command=your_module:main_function',
        ],
    },
    description='This project involves the creation of a Digital Twin LLM Assistant designed to help manage the portfolios of high-net-worth clients. The assistant is powered by a series of cutting-edge language models (LLM) capable of analyzing financial data, proposing investment strategies, and providing recommendations tailored to the unique goals and risk profile of each client.',
    author='Askar Embulatov',
    author_email='comradefrunze@gmail.com',
    url='https://github.com/ZakatZakat/Digital-Assistant/tree/main',
)