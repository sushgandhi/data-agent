from setuptools import setup, find_packages

setup(
    name="data-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pandas",
        "numpy",
        "plotly",
        "openai",
        "python-dotenv",
        "psutil",
    ],
    python_requires=">=3.8",
) 